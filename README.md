# HuggingFace Transformers Examples

A collection of example implementations showcasing various applications of HuggingFace's Transformers library for different NLP tasks.

## Project Overview

This repository contains four different implementations that demonstrate the flexibility and power of transformer models for various natural language processing and tabular data tasks:

1. **Text Classification**: Custom implementation for fine-tuning transformer models on text classification tasks.
2. **Seq2Seq (Summarization)**: T5-based implementation for text summarization tasks.
3. **T5 Encoder for Classification**: Using only the encoder component of T5 for efficient sequence classification.
4. **Tabular Classification**: Adapting transformer architectures for structured tabular data.

Each implementation includes detailed documentation, code examples, and configuration options to help you understand and adapt these models to your own use cases.

## Repository Structure

```
huggingface-transformers-examples/
├── text_classification/         # Text classification implementation
├── seq2seq/                     # Sequence-to-sequence (summarization) implementation
├── t5encoder_classification/    # T5 encoder-based classification
├── tabular_classification/      # Transformer models for tabular data
├── LICENSE                      # License information
└── README.md                    # This file
```

## Implementation Details

### Text Classification

A custom implementation for text classification using pre-trained transformer models like BART.

**Key Features:**
- Custom model architecture built on pre-trained transformers
- Configurable classification setup with support for any number of labels
- Easy integration with HuggingFace's Trainer API

[Learn more about Text Classification](./text_classification/README.md)

### Seq2Seq (Summarization)

A T5-based implementation for text summarization, fine-tuned on the CNN DailyMail dataset.

**Key Features:**
- T5 model fine-tuning for abstractive summarization
- ROUGE metrics for evaluation
- Support for both training and inference

[Learn more about Seq2Seq Summarization](./seq2seq/README.md)

### T5 Encoder for Classification

A custom implementation that uses only the encoder component of the T5 model for sequence classification tasks.

**Key Features:**
- Lightweight classification using T5 encoder
- Custom classification head
- Comprehensive training pipeline with metrics tracking

[Learn more about T5 Encoder Classification](./t5encoder_classification/README.md)

### Tabular Classification

An implementation that adapts transformer architectures (BERT, RoBERTa, DeBERTa) for structured tabular data classification.

**Key Features:**
- Positional encoding for tabular data
- Support for multiple transformer backbones
- Integration with Weights & Biases for experiment tracking

[Learn more about Tabular Classification](./tabular_classification/README.md)

## Installation

Each implementation has its own requirements, but they all rely on HuggingFace Transformers and PyTorch. Refer to the individual README files in each directory for specific installation instructions.

## Usage

Each implementation includes detailed usage instructions in its respective README file. The implementations follow a similar pattern of loading pre-trained models, configuring them for specific tasks, and fine-tuning them on custom datasets.

## License

This project is licensed under the terms included in the [LICENSE](./LICENSE) file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
