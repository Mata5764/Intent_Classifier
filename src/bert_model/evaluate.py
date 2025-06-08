import torch
from torch import nn
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from transformers.modeling_outputs import SequenceClassifierOutput

# ----- Custom Model Class -----
class CustomBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.intermediate = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.intermediate(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# ----- Dataset Class -----
class IntentDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

# ----- Load Model & Tokenizer -----
def load_model_and_tokenizer(model_path, label2id, id2label):
    config = BertConfig.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = CustomBertForSequenceClassification(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model, tokenizer

# ----- Predict -----
def predict(model, tokenizer, sentences, id2label):
    dataset = IntentDataset(sentences, tokenizer)
    loader = DataLoader(dataset, batch_size=16)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    return [id2label[p] for p in all_preds]

# ----- Run Inference -----
if __name__ == "__main__":
    # Load label mappings from JSON files (consistent with training)
    with open('src/bert_model/label2id.json', 'r') as f:
        label2id = json.load(f)
    with open('src/bert_model/id2label.json', 'r') as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    model_path = "src/bert_model/best_model_three_layers.pt"
    model, tokenizer = load_model_and_tokenizer(model_path, label2id, id2label)

    test_sentences = [
        "I need a mattress for back pain.",
        "Where can I buy a soft bed?"
    ]
    predictions = predict(model, tokenizer, test_sentences, id2label)
    
    for sent, pred in zip(test_sentences, predictions):
        print(f"Query: {sent}\nPredicted Intent: {pred}\n")
