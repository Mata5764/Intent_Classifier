import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
import json
import os
from transformers.modeling_outputs import SequenceClassifierOutput

class CustomBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # Add intermediate layer
        self.intermediate = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]  # [CLS] token output
        
        # Single intermediate layer
        pooled_output = self.intermediate(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            # Apply label smoothing
            num_classes = logits.size(-1)
            smooth_factor = 0.1
            smooth_value = smooth_factor / (num_classes - 1)
            
            # Create one-hot encoding
            one_hot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)
            
            # Create soft labels
            soft_labels = one_hot * (1.0 - smooth_factor) + smooth_value
            
            # Calculate loss with soft labels and class weights
            log_probs = F.log_softmax(logits, dim=-1)
            if class_weights is not None:
                loss = -(soft_labels * log_probs * class_weights.unsqueeze(0)).sum(dim=-1).mean()
            else:
                loss = -(soft_labels * log_probs).sum(dim=-1).mean()
        else:
            loss = None

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def save_label_mappings(label2id, id2label, save_dir='data/json'):
    """Save label mappings to JSON files"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save label2id mapping
    with open(os.path.join(save_dir, 'label2id.json'), 'w') as f:
        json.dump(label2id, f, indent=2)
    
    # Save id2label mapping
    with open(os.path.join(save_dir, 'id2label.json'), 'w') as f:
        json.dump(id2label, f, indent=2)
    
    print(f"✅ Label mappings saved to {save_dir}/")

def print_confusion_metrics(y_true, y_pred, epoch):
    """Print confusion matrix metrics for multi-class classification."""
    print(f'\nOverall Confusion Matrix Metrics - Epoch {epoch}')
    print('-' * 50)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    n_samples = len(y_true)
    n_classes = len(np.unique(y_true))
    
    # Calculate metrics for each class
    class_metrics = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        tn = n_samples - (tp + fp + fn)
        
        accuracy = (tp + tn) / n_samples
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics.append({
            'class': i,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # Calculate overall metrics
    total_tp = sum(m['tp'] for m in class_metrics)
    total_fp = sum(m['fp'] for m in class_metrics)
    total_fn = sum(m['fn'] for m in class_metrics)
    total_tn = sum(m['tn'] for m in class_metrics)
    
    overall_accuracy = (total_tp + total_tn) / (n_samples * n_classes)
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print(f'Number of samples: {n_samples}')
    print(f'Number of classes: {n_classes}')
    print(f'True Positives (TP): {total_tp} (correct predictions)')
    print(f'False Positives (FP): {total_fp} (incorrect predictions)')
    print(f'False Negatives (FN): {total_fn} (missed predictions)')
    print(f'True Negatives (TN): {total_tn} (correct rejections)')
    print(f'Overall Accuracy: {overall_accuracy:.3f}')
    print(f'Overall Precision: {overall_precision:.3f}')
    print(f'Overall Recall: {overall_recall:.3f}')
    print(f'Overall F1-score: {overall_f1:.3f}')
    print('-' * 50)
    
    print('\nPer-class accuracy:')
    for metric in class_metrics:
        class_samples = sum(cm[metric['class'], :])  # Total samples for this class
        print(f"Class {metric['class']}: {metric['tp']}/{class_samples} = {metric['accuracy']:.3f}")

def load_data():
    # Read the CSV file
    df = pd.read_csv('data/sofmattress_train.csv')
    
    # Create label mappings
    unique_labels = df['label'].unique()
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # Convert labels to IDs
    df['label_id'] = df['label'].map(label2id)
    
    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df['label_id']),
        y=df['label_id']
    )
    
    # Save label mappings
    save_label_mappings(label2id, id2label)
    
    return df, label2id, id2label, class_weights

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data
    df, label2id, id2label, class_weights = load_data()
    
    # Convert class weights to tensor and move to device
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['sentence'].values,
        df['label_id'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label_id']
    )

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    model = CustomBertForSequenceClassification(config)
    
    # Freeze all BERT layers except the last 3 encoder layers
    for i, layer in enumerate(model.bert.encoder.layer[:-3]):  # All layers except last 3
        for param in layer.parameters():
            param.requires_grad = False
    
    print("\nModel architecture:")
    print("BERT -> Linear(768,768) -> ReLU -> Dropout -> Linear(768,num_labels)")
    print("\nTraining status:")
    print("BERT layers: All frozen except last 3 encoder layers")
    print("Classification head: Trainable")
    print("Label smoothing: Enabled (factor=0.1)")
    print("Learning rate: Constant 2e-5")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nTotal parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)')
    
    model.to(device)

    # Create datasets
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer)
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Training loop
    num_epochs = 80
    best_val_accuracy = 0
    
    # Initialize optimizer with constant learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print(f'Learning rate: 2e-5 (constant)')
        
        # Training phase
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                class_weights=class_weights
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            progress_bar.set_description(f'Training (loss: {loss.item():.4f})')
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        all_val_preds = []
        all_val_labels = []
        progress_bar = tqdm(val_loader, desc='Validation')
        
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    class_weights=class_weights
                )
                
                loss = outputs.loss
                total_val_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_description(f'Validation (loss: {loss.item():.4f})')
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        print(f'Validation accuracy: {val_accuracy:.4f}')
        
        # Print confusion matrix metrics
        print_confusion_metrics(all_val_labels, all_val_preds, epoch + 1)
        
        # Save model if it's the best so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'src/bert_model/best_model_three_layers.pt')
            print('Best model saved!')

if __name__ == '__main__':
    main() 