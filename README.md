# TRM Text-to-SQL Implementation

This repository implements a Tiny Recursive Model (TRM) for the text-to-SQL task using the Spider dataset.

## Overview

The TRM model uses a recursive attention mechanism with transformer architecture to translate natural language questions into SQL queries. The implementation includes:

- **TRM Model**: Encoder-decoder architecture with recursive attention layers
- **Dataset Preprocessing**: Automated preprocessing of Spider dataset
- **Training Pipeline**: Complete training loop with validation
- **Evaluation Metrics**: Exact match and execution accuracy

## Installation

1. Clone the repository or ensure all files are in the same directory

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

### Option 1: Manual Download (Recommended)

1. Download the Spider dataset from:
   - Official repository: https://github.com/taoyds/spider
   - Kaggle: https://www.kaggle.com/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset

2. Place the following files in `data/spider/`:
   - `train_spider.json`
   - `train_others.json` (optional)
   - `dev.json`
   - `tables.json`

### Option 2: Automated Download (if implemented)

```bash
python main.py download --data_dir data
```

## Usage

### Complete Pipeline

1. **Download dataset** (if not done manually):
```bash
python main.py download
```

2. **Preprocess the dataset**:
```bash
python main.py preprocess --data_dir data/spider --output_dir data/processed
```

3. **Train the model**:
```bash
python main.py train \
    --data_dir data/spider \
    --processed_dir data/processed \
    --output_dir checkpoints \
    --batch_size 16 \
    --epochs 10 \
    --lr 1e-4 \
    --d_model 512 \
    --n_heads 8 \
    --n_layers 6
```

4. **Evaluate the model**:
```bash
python main.py evaluate \
    --checkpoint checkpoints/best_model.pt \
    --output_file evaluation_results.json
```

### Individual Scripts

You can also run individual components:

```bash
# Preprocessing only
python preprocess.py --data_dir data/spider --output_dir data/processed

# Training only
python train.py --data_dir data/spider --processed_dir data/processed --epochs 10

# Evaluation only
python evaluate.py --checkpoint checkpoints/best_model.pt
```

## Model Architecture

The TRM model consists of:

1. **Encoder**: Transformer encoder that processes the question and schema context
2. **Decoder**: Transformer decoder with recursive attention layers that generates SQL
3. **Recursive Attention**: Multi-head attention mechanism with residual connections

Key features:
- Positional encoding for sequence understanding
- Recursive attention layers for better context modeling
- Schema-aware encoding that includes database structure

## Configuration

Default hyperparameters:
- `d_model`: 512 (model dimension)
- `n_heads`: 8 (attention heads)
- `n_layers`: 6 (number of layers)
- `max_len`: 512 (maximum sequence length)
- `batch_size`: 16
- `learning_rate`: 1e-4

## Evaluation Metrics

The evaluation script computes:
- **Exact Match (EM)**: Percentage of generated SQL queries that exactly match the gold SQL
- **Execution Accuracy**: Percentage of queries that produce correct results (requires database connection)

## File Structure

```
.
├── main.py              # Main entry point
├── download_spider.py   # Dataset download utility
├── preprocess.py        # Data preprocessing
├── trm_model.py         # TRM model implementation
├── train.py             # Training script
├── evaluate.py          # Evaluation script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Notes

- The current implementation uses a simple tokenizer. For better performance, consider using BERT or GPT tokenizers.
- Execution accuracy requires database connections to Spider's databases, which is not included in this implementation.
- The model architecture is based on the TRM principles described in the implementation plan PDF.

## References

- Spider Dataset: https://github.com/taoyds/spider
- TRM Architecture: Based on Tiny Recursive Model principles

## License

This implementation is for research and educational purposes.

