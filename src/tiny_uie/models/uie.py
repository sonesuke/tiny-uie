"""UIE (Universal Information Extractor) model implementation.

This model handles span extraction, type classification, and relation extraction.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import AutoModel

from .relation import BiaffineRel


class SpanHead(nn.Module):
    """Span head module for extracting text spans.

    This module predicts start and end positions of text spans.
    """

    def __init__(self, hidden_size: int) -> None:
        """Initialize the SpanHead module.

        Args:
            hidden_size: Hidden size of the model
        """
        super().__init__()
        self.start = nn.Linear(hidden_size, 1)
        self.end = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:  # [B,T,H]
        """Forward pass for span head.

        Args:
            hidden_states: Hidden states tensor of shape [B, T, H]

        Returns:
            Tuple of start and end logits tensors
        """
        s = self.start(hidden_states).squeeze(-1)  # [B,T]
        e = self.end(hidden_states).squeeze(-1)  # [B,T]
        return s, e


@dataclass
class UIEOutputs:
    """Output dataclass for UIE model.

    Attributes:
        start_logits: Start position logits tensor
        end_logits: End position logits tensor
    """

    start_logits: torch.Tensor
    end_logits: torch.Tensor


class UIE(nn.Module):
    """Universal Information Extractor model.

    This model handles span extraction, type classification, and relation extraction.
    """

    def __init__(self, model_name: str, n_rel: int = 0) -> None:
        """Initialize the UIE model.

        Args:
            model_name: Name of the pretrained encoder model
            n_rel: Number of relation types (default: 0)
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.span = SpanHead(hidden)
        self.rel = BiaffineRel(hidden, n_rel) if n_rel > 0 else None

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, span_indices: list | None = None
    ) -> tuple[UIEOutputs, Tensor, Tensor | None]:
        """Forward pass for UIE model.

        Args:
            input_ids: Input token IDs tensor
            attention_mask: Attention mask tensor
            span_indices: List of span index pairs for relation extraction (optional)

        Returns:
            Tuple of UIEOutputs, hidden states, and relation scores (if available)
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B,T,H]
        s, e = self.span(h)
        rel_scores = None
        if self.rel is not None and span_indices is not None:
            head_emb, tail_emb = [], []
            for (h_st, h_ed), (t_st, t_ed) in span_indices:
                head_emb.append(h[0, h_st : h_ed + 1].mean(dim=0))
                tail_emb.append(h[0, t_st : t_ed + 1].mean(dim=0))
            if head_emb:
                head_emb = torch.stack(head_emb, dim=0)
                tail_emb = torch.stack(tail_emb, dim=0)
                rel_scores = self.rel(head_emb, tail_emb)
        return UIEOutputs(s, e), h, rel_scores
