import os
import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
from torch.utils.data import dataset
from datasets import load_dataset, load_metric, DatasetDict, Dataset
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple, List

from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
    AutoTokenizer,
    T5Config,
    T5Tokenizer
)

from transformers.trainer_utils import get_last_checkpoint

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def main():
    # Arguments
    import argparse
    parser = argparse.ArgumentParser()
    # Input
    parser.add_argument("--train_file", type=str, default=None, help="Path to train file.")
    parser.add_argument("--validation_file", type=str, default=None, help="Path to valid file.")
    parser.add_argument("--test_file", type=str, default=None, help="Path to test file.")
    parser.add_argument("--article_column", type=str, default="article", help="Name of article column.")
    parser.add_argument("--summary_column", type=str, default="summary_text", help="Name of summary column.")
    parser.add_argument("--label_column", type=str, default="label", help="Name of label column.")

    # Model and Training
    parser.add_argument("--model_name_or_path", default="t5-base", type=str, help="Path to pre-trained model or name of huggingface/transformers model")
    parser.add_argument("--num_epochs", default=10, type=int, help="Number of epochs to train for")
    parser.add_argument("--max_length", default=512, type=int, help="Maximum length of the sequence")
    parser.add_argument("--do_train", default=True, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", default=False, action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Run evaluation every X steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--batch_size_val", type=int, default=10, help="Batch size per GPU/CPU for validation.")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Linear warmup over warmup_steps.")
    parser.add_argument("--accum_steps", type=int, default=2, help="Accumulate gradients across X steps.")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam size.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="The initial learning rate for Adam.")
    parser.add_argument("--fp16", default=False, action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--overwrite_output_dir", default=False, action="store_true", help="Overwrite the content of the output directory")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience.")
    # For debugging
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum number of training samples to use.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Maximum number of evaluation samples to use.")
    parser.add_argument("--max_predict_samples", type=int, default=None, help="Maximum number of prediction samples to use.")
    # Output
    parser.add_argument("--output_dir", default="imdb_classification", type=str, help="The output directory where the model predictions and checkpoints will be written.")
    # Setting save 2 last checkpoint and save best checkpoint
    parser.add_argument("--save_total_limit", type=int, default=2, help="If we save total_limit checkpoints, delete the older checkpoints")
    parser.add_argument("--load_best_model_at_end", default=True, action="store_true", help="Save only the best checkpoint in the output directory")
    config = parser.parse_args()

    print(config)

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config.model_name_or_path)

    # Load dataset
    def convert_text_for_input(article, summary):
        return f"for the following article: {article}\nsummary of the article: {summary}"

    train_df = pd.read_csv(config.train_file)
    train_df["text_for_input"] = train_df.apply(lambda x: convert_text_for_input(x[config.article_column], x[config.summary_column]), axis=1)
    eval_df = pd.read_csv(config.validation_file)
    eval_df["text_for_input"] = eval_df.apply(lambda x: convert_text_for_input(x[config.article_column], x[config.summary_column]), axis=1)
    test_df = pd.read_csv(config.test_file)
    test_df["text_for_input"] = test_df.apply(lambda x: convert_text_for_input(x[config.article_column], x[config.summary_column]), axis=1)

    raw_datasets = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "val": Dataset.from_pandas(eval_df),
            "test": Dataset.from_pandas(test_df),
        }
    )

    # Preprocessing the datasets
    def preprocess_function(examples):
        args = (
            (examples["text_for_input"], )
        )
        result = tokenizer(*args, padding="max_length", truncation=True, max_length=config.max_length)
        result["label"] = [int(i)-1 for i in examples[config.label_column]]
        return result

    ds = raw_datasets.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )

    train_dataset = ds["train"].select(range(config.max_train_samples if config.max_train_samples is not None else len(ds["train"])))
    eval_dataset = ds["val"].select(range(config.max_eval_samples if config.max_eval_samples is not None else len(ds["val"])))
    predict_dataset = ds["test"].select(range(config.max_predict_samples if config.max_predict_samples is not None else len(ds["test"])))

    # Load model and tokenizer
    # model = AutoModelForSequenceClassification.from_pretrained(config.model_name_or_path, num_labels=len(set(train_df[config.label_column])))
    from t5encoder import T5EncoderForSequenceClassification

    config_model = T5Config.from_pretrained(config.model_name_or_path, num_labels=len(set(train_df[config.label_column])))
    model = T5EncoderForSequenceClassification(config_model)
    model.load_pretrained_weights(config.model_name_or_path)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        do_train=config.do_train,
        do_eval=config.do_eval,
        do_predict=config.do_predict,
        num_train_epochs=config.num_epochs,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size_val,
        logging_steps=config.logging_steps,
        warmup_steps=config.warmup_steps,
        gradient_accumulation_steps=config.accum_steps,
        overwrite_output_dir=config.overwrite_output_dir,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model="f1_score",
    )
    print("------- Training arguments -------")
    print(training_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Metric
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        acc = (preds == p.label_ids).astype(np.float32).mean().item()
        f1 = f1_score(p.label_ids, preds, average="macro")
        return {"accuracy": acc, "f1_score": f1}


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params}")

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            config.max_train_samples if config.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = (
            config.max_eval_samples if config.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results.csv")
        predict_df = predict_dataset.to_pandas().copy()
        predict_df["prediction"] = [p+1 for p in predictions]
        predict_df.to_csv(output_predict_file, index=False)

    print(classification_report(predict_df[config.label_column], predictions))

if __name__ == "__main__":
    main()