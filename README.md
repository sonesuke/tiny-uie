# UIE Minimal Starter (ModernBERT-Ja-30m)

A minimal implementation kit for span extraction + type classification + relation extraction using ModernBERT-Ja 30M as the backbone (PyTorch + Hugging Face).

## Features

- **Span Extraction**: Extract text spans using a simple head on top of ModernBERT
- **Type Classification**: Classify spans into predefined types using InfoNCE loss
- **Relation Extraction**: Extract relations between spans using Biaffine scoring
- **Post-processing**: NMS and other techniques to improve inference quality
- **UV Package Management**: Fast dependency management and project setup

## Quick Start

### Development Setup

1. Set up a virtual environment and install development dependencies:
   ```
   uv sync
   ```


### Data Preparation

Generate dummy training data:
```
uv run python tools/make_dummy_data.py
```

This creates:
- `data/train.jsonl` and `data/dev.jsonl`: Training and validation data
- `data/schema/types.json` and `data/schema/relations.json`: Type and relation definitions

### Training Pipeline

1. **Span Extraction Training**:
   ```
   uv run train-span
   ```

2. **Type Classification Training**:
   ```
   uv run train-type
   ```

3. **Relation Extraction Training**:
   ```
   uv run train-rel
   ```

All models are saved to the `checkpoints/` directory.

### Inference

Run inference with the full pipeline:
```
uv run infer "ABC株式会社は1999年に山田太郎が設立した。"
```

## Project Structure

```
.
├── config.yaml             # Model and training configuration
├── src/
│   └── tiny_uie/          # Main package
│       ├── __init__.py     # Package initializer
│       ├── py.typed        # Type checking marker
│       ├── dataset.py      # Dataset loading and preprocessing
│       ├── train_span.py   # Span extraction training script
│       ├── train_type.py   # Type classification training script
│       ├── train_rel.py    # Relation extraction training script
│       ├── infer.py        # Inference script with post-processing
│       ├── infer_rel.py    # Relation-specific inference script
│       ├── postprocess.py  # Post-processing functions (NMS, etc.)
│       ├── models/        # Model components
│       │   ├── __init__.py
│       │   ├── uie.py     # Main UIE model with span and relation heads
│       │   └── relation.py  # Biaffine relation extraction module
│       └── utils/         # Utility functions
│           ├── __init__.py
│           └── rel_pairs.py  # Relation pair utilities
├── data/
│   ├── train.jsonl         # Training data
│   ├── dev.jsonl          # Validation data
│   └── schema/            # Schema definitions
│       ├── types.json
│       └── relations.json
├── tools/
│   └── make_dummy_data.py # Data generation script
├── checkpoints/           # Saved model checkpoints
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml
└── README.md
```

## Configuration

### config.yaml

Contains all model and training configuration:
- `model_name`: Pretrained model identifier
- `max_length`: Maximum sequence length
- `train`: Training hyperparameters
- `loss_weights`: Weighting for different loss components
- `inference`: Inference parameters (thresholds, NMS settings)
- `paths`: File paths for data and checkpoints

### Model Architecture

The UIE model consists of three components:
1. **Span Head**: Detects start and end positions of text spans
2. **Type Head**: Classifies spans into semantic types
3. **Relation Head**: Biaffine scorer for relations between spans

### Post-processing

Inference includes several post-processing steps:
- Threshold-based filtering of span candidates
- Length limitations on extracted spans
- Non-Maximum Suppression (NMS) to remove duplicates
- Character offset mapping for readable outputs

## Using UV for Dependency Management

[UV](https://github.com/astral-sh/uv) manages dependencies quickly:
- Install dependencies: `uv sync`
- Add a dependency: `uv add package-name`
- Run scripts: `uv run python script.py`

## License

This project is available as open source under the terms of the MIT License.

## Contributing

Contributions are welcome! Please submit a Pull Request.