# T5 Encoder for Sequence Classification

This project implements a sequence classification model using the T5 encoder architecture. It leverages the powerful T5 transformer model for text classification tasks, using only the encoder part of the T5 architecture.

## Overview

The project implements a custom `T5EncoderForSequenceClassification` model that:
- Uses the T5 encoder for feature extraction
- Adds a classification head on top of the encoder
- Supports fine-tuning on custom classification tasks

## Features

- Custom T5 encoder-based classification model
- Support for various input formats (articles, summaries, etc.)
- Comprehensive training pipeline with metrics tracking
- Early stopping and checkpointing
- Support for custom datasets

## Requirements

- Python 3.6+
- PyTorch 1.12.1
- Transformers 4.23.1
- Other dependencies as listed in `requirements.txt`

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

The main training script can be run using:

```bash
python main.py \
    --train_file path/to/train.json \
    --validation_file path/to/validation.json \
    --test_file path/to/test.json \
    --model_name_or_path t5-base \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 2e-5
```

### Key Arguments

- `--train_file`: Path to training data file
- `--validation_file`: Path to validation data file
- `--test_file`: Path to test data file
- `--model_name_or_path`: Pre-trained model to use (default: "t5-base")
- `--num_epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate for training

### Data Format

The input data should be in JSON format with the following structure:
```json
{
    "article": "text of the article",
    "summary_text": "summary of the article",
    "label": 0
}
```

## Model Architecture

The model consists of:
1. T5 encoder for feature extraction
2. Classification head (linear layer) on top of the encoder
3. Cross-entropy loss for training

## Evaluation Metrics

The model tracks:
- Accuracy
- F1 score
- Precision
- Recall
- Confusion matrix