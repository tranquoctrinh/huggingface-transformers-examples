Custom Model for Text Classification with Trainer from HuggingFace transformers library


# Custom Config

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

# Custom Model

```python

@dataclass
class OutputCustom(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class ModelCustom(PreTrainedModel):
    config_class = ConfigCustom # This help we load the model using the .from_pretrained() method

    def __init__(self, config: ConfigCustom):
        super(ModelCustom, self).__init__(config)
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
        hidden_states = encoded.last_hidden_state

        eos_mask = input_ids.eq(self.config.eos_token_id)

        encoded_rep = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]

        x = self.dropout(encoded_rep)
        x = torch.tanh(self.dense_1(x))
        x = self.dropout(x)
        logits = self.dense_2(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return OutputCustom(
            loss=loss,
            logits=logits
        )
```


# Custom Dataset

```python
class ClassificationDataset(dataset.Dataset):
    def __init__(self, path_df=None, tokenizer=None, max_length=512, prefix=None, ignore_pad_token_for_loss=True, padding="max_length", max_samples=None, label_to_id=None, predict=False, split="train"):
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
```