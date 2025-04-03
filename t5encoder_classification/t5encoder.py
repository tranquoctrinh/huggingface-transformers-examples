
from transformers import (
    T5EncoderModel,
    T5Config,
    PreTrainedModel,
)

from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn


class T5EncoderForSequenceClassification(PreTrainedModel):
    
    config_class = T5Config

    def __init__(self, config: T5Config):
        super(T5EncoderForSequenceClassification, self).__init__(config)
        self.config = config
        self.encoder = T5EncoderModel(config)
        self.classifier = nn.Linear(config.d_model, config.num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, input_type_ids=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, -1, :])
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def load_pretrained_weights(self, model_name):
        self.encoder.from_pretrained(model_name)

    

def main():
    config = T5Config.from_pretrained("t5-base", num_labels=5)
    model = T5EncoderForSequenceClassification(config)
    model.load_pretrained_weights("t5-base")
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    main()