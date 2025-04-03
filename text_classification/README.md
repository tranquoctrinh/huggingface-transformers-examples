# Text Classification with HuggingFace Transformers

This repository contains a custom implementation for text classification using the HuggingFace Transformers library. It provides a flexible framework to fine-tune pre-trained language models (like BART) for text classification tasks.

## Features

- Custom model architecture built on pre-trained transformers
- Configurable classification setup with support for any number of labels
- Easy integration with HuggingFace's Trainer API
- Dataset handling for text classification tasks
- Support for model saving and loading through HuggingFace's standard interfaces

## Installation

```bash
pip install transformers datasets torch pandas
```

## Usage

### Basic Example

```python
from transformers import AutoTokenizer, Trainer, TrainingArguments
from model import ModelCustom, ConfigCustom
from dataset import ClassificationDataset

# Initialize config and model
config = ConfigCustom(
    model_type="bart",
    pretrained_model="facebook/bart-base",
    num_labels=2,
    max_length=128
)
model = ModelCustom(config)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)

# Prepare datasets
train_dataset = ClassificationDataset(
    split="train",
    tokenizer=tokenizer,
    max_length=config.max_length
)
eval_dataset = ClassificationDataset(
    split="test",
    tokenizer=tokenizer,
    max_length=config.max_length
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Save model
model.save_pretrained("./saved_model")
config.save_pretrained("./saved_model")

# Load model
loaded_model = ModelCustom.from_pretrained("./saved_model")
```

## Components

### Custom Configuration (ConfigCustom)

The `ConfigCustom` class extends HuggingFace's `PretrainedConfig` to define the configuration for our custom model:

```python
class ConfigCustom(PretrainedConfig):
    def __init__(
        self,
        model_type: str = "bart",
        pretrained_model: str = "facebook/bart-base",
        num_labels: int = 2,
        dropout: float = 0.1,
        inner_dim: int = 1024,
        max_length: int = 128,
        **kwargs
    ):
        super(ConfigCustom, self).__init__(num_labels=num_labels, **kwargs)
        self.model_type = model_type
        self.pretrained_model = pretrained_model
        self.dropout = dropout
        self.inner_dim = inner_dim
        self.max_length = max_length

        encoder_config = AutoConfig.from_pretrained(
            self.pretrained_model,
        )
        self.vocab_size = encoder_config.vocab_size
        self.eos_token_id = encoder_config.eos_token_id
        # self.encoder_config = self.encoder_config.to_dict()
```

### Custom Model (ModelCustom)

The `ModelCustom` class implements a text classification model on top of a pre-trained transformer:

- Uses a pre-trained encoder (e.g., BART)
- Adds classification layers on top of the encoder
- Supports standard HuggingFace model interfaces

### Classification Dataset

The `ClassificationDataset` class handles data preprocessing for text classification:

- Supports loading data from CSV files or HuggingFace datasets
- Handles tokenization and conversion to model inputs
- Supports label mapping for classification tasks

## Customization

You can customize the model by:

1. Changing the pre-trained model (`pretrained_model` parameter)
2. Adjusting the number of labels (`num_labels` parameter)
3. Modifying the architecture parameters (dropout, inner dimension)
4. Changing the maximum sequence length (`max_length` parameter)