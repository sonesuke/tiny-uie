"""Dataset module for UIE model training.

This module provides dataset loading and preprocessing functionality for UIE models.
It handles span extraction, type classification, and relation extraction tasks.
"""

import json
from pathlib import Path

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SpanTok:
    """Represents a token-level span with start and end positions."""

    def __init__(self, start_tok: int, end_tok_inclusive: int) -> None:
        """Initialize SpanTok with start and end token positions."""
        self.start_tok = start_tok
        self.end_tok_inclusive = end_tok_inclusive


class UIEDataset(Dataset):
    """Dataset class for UIE model training.

    Loads and preprocesses data for span extraction, type classification,
    and relation extraction tasks.
    """

    def __init__(self, data_path: str, model_name: str, max_length: int) -> None:
        """Initialize UIEDataset with data path, model name, and max length."""
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        with Path(data_path).open(encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset by index."""
        item = self.data[idx]
        text = item["text"]
        spans = item["spans"]

        # Tokenize the text
        enc = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        # Keep the tokenizer output as is for char_to_token method
        enc_dict = {
            "input_ids": enc["input_ids"].squeeze().tolist(),
            "attention_mask": enc["attention_mask"].squeeze().tolist(),
        }

        # Convert character-level spans to token-level spans
        spans_tok = []
        for span in spans:
            start_char = span["start"]
            end_char = span["end"]

            # Get token-level positions
            start_tok = enc.char_to_token(start_char)
            end_tok = enc.char_to_token(end_char - 1)  # -1 because end is exclusive in Python slicing

            if start_tok is not None and end_tok is not None:
                spans_tok.append(SpanTok(start_tok, end_tok_inclusive=end_tok))

        return {"enc": enc_dict, "spans": spans, "spans_tok": spans_tok, "relations": item.get("relations", [])}
