# T5-based Text Summarization

This project implements a text summarization system using the T5 model, fine-tuned on the CNN DailyMail dataset. The implementation uses the Hugging Face Transformers library to create a sequence-to-sequence model for abstractive text summarization.

## Overview

The project uses:
- T5-base model as the base architecture
- CNN DailyMail dataset for training and evaluation
- Hugging Face Transformers library for model implementation
- ROUGE metrics for evaluation

## Features

- Fine-tuning of T5 model for text summarization
- Support for both training and evaluation
- Early stopping callback to prevent overfitting
- Custom dataset handling for CNN DailyMail
- ROUGE metric computation for evaluation

## Setup

1. Install the required dependencies:
```bash
pip install transformers datasets nltk torch pandas numpy
```

2. Download NLTK data (required for evaluation):
```python
import nltk
nltk.download('punkt')
```

## Usage

### Training

To train the model:
```bash
python seq2seq.py --model_name_or_path t5-base --do_train --do_eval --do_predict
```

### Available Arguments

- `--model_name_or_path`: Path to pretrained model or model identifier from huggingface.co/models
- `--do_train`: Whether to run training
- `--do_eval`: Whether to run evaluation
- `--do_predict`: Whether to run predictions
- `--max_source_length`: Maximum input sequence length
- `--max_target_length`: Maximum output sequence length
- `--num_train_epochs`: Total number of training epochs
- `--per_device_train_batch_size`: Batch size per device during training
- `--per_device_eval_batch_size`: Batch size per device during evaluation
- `--learning_rate`: Initial learning rate
- `--output_dir`: Output directory for model checkpoints and predictions

## Model Architecture

The implementation uses:
- T5ForConditionalGeneration as the base model
- Seq2SeqTrainingArguments for training configuration
- DataCollatorForSeq2Seq for batch preparation
- EarlyStoppingCallback for preventing overfitting

## Evaluation

The model is evaluated using ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L) which measure the quality of the generated summaries against reference summaries.