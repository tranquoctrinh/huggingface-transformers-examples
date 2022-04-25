import os
import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
from torch.utils.data import dataset
from datasets import load_dataset, load_metric
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple, List

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
    AutoModel, 
    AutoTokenizer, 
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel,
    default_data_collator,
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


class ClassificationDataset(dataset.Dataset):
    def __init__(
        self, 
        path_df=None, 
        tokenizer=None, 
        max_length=512, 
        prefix=None, 
        ignore_pad_token_for_loss=True, 
        padding="max_length", 
        max_samples=None, 
        label_to_id=None, 
        predict=False, 
        split="train"
    ):

        self.text_column = "text"
        self.label_column = "label"
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix = prefix if prefix is not None else ""
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.padding = padding
        self.predict = predict
        if path_df is None:
            self.df = load_dataset('imdb', split=split).to_pandas()
        else:
            self.df = pd.read_csv(path_df)

        if max_samples is not None:
            self.df = self.df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
        if label_to_id is not None:
            self.label_to_id = label_to_id
        else:
            self.label_to_id = {v: i for i,v in enumerate(df[self.label_column].unique())}
        
        self.df["label_id"] = [self.label_to_id.get(label, -1) for label in self.df[self.label_column]]


    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        input_text = self.df.loc[index, self.text_column]
        label = self.df.loc[index, "label_id"]
        
        input_text = self.prefix + input_text
        model_inputs = self.tokenizer(input_text, max_length=self.max_length, padding=self.padding, truncation=True)
        if not self.predict:
            model_inputs["label"] = label
        return model_inputs



@dataclass
class OutputCustom(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class ConfigCustom(PretrainedConfig):
    def __init__(
        self,
        model_type: str = "bert",
        pretrained_model: str = "bert-base-uncased",
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
        # self.num_labels = num_labels

        encoder_config = AutoConfig.from_pretrained(
            self.pretrained_model,
        )
        self.vocab_size = encoder_config.vocab_size
        self.eos_token_id = encoder_config.eos_token_id
        # self.encoder_config = self.encoder_config.to_dict()


class ModelCustom(PreTrainedModel):
    config_class = ConfigCustom # This help we load the model using the .from_pretrained() method

    def __init__(self, config: ConfigCustom):
        super(ModelCustom, self).__init__(config)
        # self.num_labels = config.num_labels
        self.config = config
        self.encoder = AutoModel.from_pretrained(self.config.pretrained_model)
        self.encoder.resize_token_embeddings(self.config.vocab_size)
        self.dense_1 = nn.Linear(
            self.encoder.config.hidden_size,
            self.config.inner_dim,
            bias=False
        )
        self.dense_2 = nn.Linear(
            self.config.inner_dim,
            self.config.num_labels,
            bias=False
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.encoder._init_weights(self.dense_1)
        self.encoder._init_weights(self.dense_2)

    def resize_token_embeddings(self, new_num_tokens):
        self.encoder.resize_token_embeddings(new_num_tokens)

    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=None, **kwargs):
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # labels=labels,
            return_dict=return_dict,
        )
        hidden_states = encoded.last_hidden_state[:, 0, :]
        x = self.dropout(hidden_states)
        x = torch.relu(self.dense_1(x))
        logits = self.dense_2(self.dropout(x))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return OutputCustom(
            loss=loss,
            logits=logits
        )
    
    def predict(self, input_ids=None, attention_mask=None, return_dict=None, threshold=0.5, **kwargs):
        logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict, **kwargs).logits
        if self.config.num_labels == 2:
            logits = torch.sigmoid(logits)
            predict_labels = (logits > threshold).long()
        else:
            logits = torch.softmax(logits, dim=-1)
            predict_labels = logits.argmax(dim=-1)
        return predict_labels



def main():
    # Arguments
    import argparse
    parser = argparse.ArgumentParser()
    # Input
    parser.add_argument("--train_file", type=str, default=None, help="Path to train file.")
    parser.add_argument("--validation_file", type=str, default=None, help="Path to valid file.")
    parser.add_argument("--test_file", type=str, default=None, help="Path to test file.")
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
    
    # Load model and tokenizer
    label_to_id = {1: 1, 0: 0}

    model_config = ConfigCustom(
        num_labels=2,
        max_length=config.max_length,
    )
    model_config.label2id = label_to_id
    model_config.id2label = {v: k for k, v in label_to_id.items()}

    model = ModelCustom(model_config)
    model.label2id = label_to_id
    model.id2label = {v: k for k, v in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model.config.pretrained_model)

    # special_tokens = ["[language]", "[\language]", "[correct]", "[\correct]", "[problem]", "[\problem]", "[incorrect]", "[\incorrect]"]
    # tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    # model.resize_token_embeddings(len(tokenizer))
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        do_train=config.do_train,
        do_eval=config.do_eval,
        do_predict=config.do_predict,
        num_train_epochs=config.num_epochs,
        evaluation_strategy="steps",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size_val,
        fp16=config.fp16,
        # half_precision_backend="apex",
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        warmup_steps=config.warmup_steps,
        gradient_accumulation_steps=config.accum_steps,
        overwrite_output_dir=config.overwrite_output_dir,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model="accuracy",
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
        
        tn, fp, fn, tp = confusion_matrix(p.label_ids, preds).ravel()
        return {
            "accuracy": accuracy_score(p.label_ids, preds),
            "f1": f1_score(p.label_ids, preds),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, # confusion matrix
            }
    
    # Load dataset
    train_dataset = ClassificationDataset(
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_samples=config.max_train_samples,
        label_to_id=model.label2id,
        split="train",
    )
    eval_dataset = ClassificationDataset(
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_samples=config.max_eval_samples,
        label_to_id=model.label2id,
        split="test",
    )
    predict_dataset = ClassificationDataset(
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_samples=config.max_predict_samples,
        label_to_id=model.label2id,
        predict=True,
        split="test",
    )

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
        predict_df = predict_dataset.df.copy()
        predict_df["prediction"] = [model.id2label[p] for p in predictions]
        predict_df.to_csv(output_predict_file, index=False)

if __name__ == "__main__":
    main()