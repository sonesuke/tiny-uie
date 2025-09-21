"""Biaffine relation extraction module for UIE model.

This module implements biaffine scoring for relation classification.
"""

import torch
from torch import Tensor, nn


class BiaffineRel(nn.Module):
    """Biaffine relation scoring module.

    This module computes relation scores between head and tail entity embeddings
    using biaffine scoring function.
    """

    def __init__(self, hidden_size: int, n_rel: int) -> None:
        """Initialize the BiaffineRel module.

        Args:
            hidden_size: Hidden size of the model
            n_rel: Number of relation types
        """
        super().__init__()
        self.U = nn.Parameter(torch.randn(n_rel, hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.U)

    def forward(self, head_emb: Tensor, tail_emb: Tensor) -> Tensor:
        """Compute relation scores between head and tail embeddings.

        Args:
            head_emb: Head entity embeddings of shape [N, H]
            tail_emb: Tail entity embeddings of shape [N, H]

        Returns:
            Relation scores tensor of shape [N, n_rel]
        """
        scores = []
        for r in range(self.U.size(0)):
            u_r = self.U[r]
            s = (head_emb @ u_r * tail_emb).sum(dim=-1)  # [N]
            scores.append(s)
        return torch.stack(scores, dim=-1)
