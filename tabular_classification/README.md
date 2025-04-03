# Tabular Classification with Transformers

This project implements a tabular classification model using transformer-based architectures (BERT, RoBERTa, or DeBERTa) with positional encoding. It's designed to handle structured data and convert it into a format suitable for transformer models.

## Features

- Support for multiple transformer architectures (BERT, RoBERTa, DeBERTa)
- Positional encoding for tabular data
- Integration with Weights & Biases for experiment tracking
- Early stopping and model checkpointing
- Comprehensive evaluation metrics

## Requirements

- Python 3.6+
- PyTorch 1.12.1
- Transformers 4.23.1
- Scikit-learn 1.1.3
- Pandas 1.3.5
- Weights & Biases (wandb) 0.12.16

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure

```
tabular_classification/
├── README.md
├── requirements.txt
└── tabular_classification.py
```

## Usage

The main script can be run using the following command:

```bash
python tabular_classification.py \
    --model_base [bert|roberta|deberta] \
    --data_path path/to/your/data.xlsx \
    --output_dir path/to/save/models \
    --num_epochs 100 \
    --batch_size 128 \
    --lr 1e-5 \
    --early_stop 10 \
    --wandb_key your_wandb_key
```

### Parameters

- `model_base`: Base transformer model to use (default: "roberta")
- `data_path`: Path to the Excel file containing your data
- `output_dir`: Directory to save trained models
- `num_epochs`: Number of training epochs
- `batch_size`: Training batch size
- `lr`: Learning rate
- `early_stop`: Number of epochs to wait before early stopping
- `wandb_key`: Weights & Biases API key (optional)

## Data Format

The input data should be in Excel format (.xlsx) with the following structure:
- Each row represents a sample
- The last column should contain the target labels
- All other columns should contain numerical features

## Model Architecture

The model consists of:
1. Positional encoding layer for tabular data
2. Transformer backbone (BERT/RoBERTa/DeBERTa)
3. Classification head

## Evaluation Metrics

The model reports:
- Accuracy
- F1 Score
- Classification Report