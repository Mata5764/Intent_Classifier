# Few-Shot LLM Predictor for Streamlit App

import time
import sys
import os
import streamlit as st

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.llm_model.few_shot import load_data, select_examples, predict
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from config import FEW_SHOT_EXAMPLES

class IntentClassification(BaseModel):
    intent: str = Field(description="The exact intent category that best matches the customer query. Must be one of the categories shown in the examples.")

class FewShotPredictor:
    def __init__(self):
        self.structured_llm = None
        self.examples = None
        self.load_model()
    
    def load_model(self):
        """Load the few-shot examples and initialize LLM"""
        try:
            # Load training data and examples
            train_texts, val_texts, train_labels, val_labels = load_data()
            self.examples = select_examples(train_texts, train_labels, examples_per_class=4)
            
            # Get OpenAI API key from Streamlit secrets
            try:
                api_key = st.secrets["OPENAI_API_KEY"]
            except KeyError:
                try:
                    # Try alternative format
                    api_key = st.secrets["secrets"]["OPENAI_API_KEY"]
                except KeyError:
                    raise ValueError("OpenAI API key not found in Streamlit secrets")
            
            # Initialize LangChain structured LLM
            llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.0, api_key=api_key)
            self.structured_llm = llm.with_structured_output(IntentClassification)
            
            print("✅ Few-Shot model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading Few-Shot model: {e}")
            raise
    
    def predict(self, text):
        """Predict intent for given text using few-shot learning"""
        if not self.structured_llm or not self.examples:
            return {"error": "Model not loaded"}
        
        try:
            start_time = time.time()
            
            # Make prediction using few-shot learning
            intent = predict(self.examples, text, self.structured_llm)
            
            prediction_time = time.time() - start_time
            
            return {
                "intent": intent,
                "confidence": "N/A",  # LLMs don't provide confidence scores
                "prediction_time": round(prediction_time * 1000, 2),  # ms
                "model_info": "GPT-4-turbo (Few-Shot)",
                "training_examples": FEW_SHOT_EXAMPLES
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

# Create singleton instance
few_shot_predictor = None

def get_few_shot_predictor():
    """Get or create Few-Shot predictor instance"""
    global few_shot_predictor
    if few_shot_predictor is None:
        few_shot_predictor = FewShotPredictor()
    return few_shot_predictor 