# BERT Model Predictor for Streamlit App

import torch
import json
import time
from transformers import BertTokenizer, BertConfig
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import importlib.util
spec = importlib.util.spec_from_file_location("bert_model", "src/bert_model/1l.py")
bert_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bert_module)
CustomBertForSequenceClassification = bert_module.CustomBertForSequenceClassification
from config import BERT_MODEL_PATH, LABEL_MAPPINGS_PATH

class BERTPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """Load the saved BERT model and tokenizer"""
        try:
            # Load label mappings
            with open(LABEL_MAPPINGS_PATH, 'r') as f:
                label2id = json.load(f)
            
            self.id2label = {v: k for k, v in label2id.items()}
            
            # Initialize tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            # Initialize model
            config = BertConfig.from_pretrained(
                'bert-base-uncased',
                num_labels=len(label2id),
                id2label=self.id2label,
                label2id=label2id
            )
            
            self.model = CustomBertForSequenceClassification(config)
            
            # Load saved weights
            self.model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ BERT model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading BERT model: {e}")
            raise
    
    def predict(self, text):
        """Predict intent for given text"""
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded"}
        
        try:
            start_time = time.time()
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)
            
            # Make prediction (filter out token_type_ids if present)
            model_inputs = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }
            with torch.no_grad():
                outputs = self.model(**model_inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_id = torch.argmax(logits, dim=-1).item()
                confidence = probabilities[0][predicted_id].item()
            
            prediction_time = time.time() - start_time
            
            return {
                "intent": self.id2label[predicted_id],
                "confidence": round(confidence * 100, 2),
                "prediction_time": round(prediction_time * 1000, 2),  # ms
                "model_info": "BERT (Fine-tuned)",
                "training_examples": 262
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

# Create singleton instance
bert_predictor = None

def get_bert_predictor():
    """Get or create BERT predictor instance"""
    global bert_predictor
    if bert_predictor is None:
        bert_predictor = BERTPredictor()
    return bert_predictor 