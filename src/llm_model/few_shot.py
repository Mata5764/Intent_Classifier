# Simple Few-Shot Intent Classification
# Clean functions without verbose printing

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json
import numpy as np
import random
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

class IntentClassification(BaseModel):
    intent: str = Field(description="The exact intent category that best matches the customer query. Must be one of the categories shown in the examples.")

def set_seed(seed=42):
    """Set seed for reproducibility - same as BERT model"""
    np.random.seed(seed)
    random.seed(seed)

def load_data():
    """Load and split data same as BERT model"""
    df = pd.read_csv('data/sofmattress_train.csv')
    
    # Load existing label mappings from JSON folder
    with open('data/json/label2id.json', 'r') as f:
        label2id = json.load(f)
    
    # Create label IDs for stratified split
    df['label_id'] = df['label'].map(label2id)
    
    # Split with same parameters as BERT
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['sentence'].values,
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label_id']
    )
    
    return train_texts, val_texts, train_labels, val_labels

def select_examples(train_texts, train_labels, examples_per_class=4):
    """Select few examples from each class"""
    df_train = pd.DataFrame({'text': train_texts, 'label': train_labels})
    examples = {}
    
    print(f"Selecting examples (max {examples_per_class} per class):")
    
    for label in df_train['label'].unique():
        label_samples = df_train[df_train['label'] == label]['text'].tolist()
        available_examples = len(label_samples)
        
        # Take minimum of available examples and requested examples_per_class
        take_count = min(available_examples, examples_per_class)
        examples[label] = label_samples[:take_count]
        
        print(f"  {label}: {take_count}/{available_examples} examples")
    
    return examples

def create_prompt(examples, query):
    """Create few-shot prompt for LLM with chain-of-thought"""
    prompt = """Classify this SOF Mattress customer query into the exact intent category shown in the examples.

Examples:
"""
    
    # Add examples
    for intent, texts in examples.items():
        for text in texts:
            prompt += f'Customer: "{text}" → Intent: {intent}\n'
    
    # Add query to classify with chain-of-thought
    prompt += f"""
Customer: "{query}"

Think step by step:
1. What is the customer asking about?
2. Which intent category best matches this query?
3. Provide your reasoning and the exact intent category."""
    
    return prompt

def predict(examples, query, structured_llm):
    """Predict intent using LangChain structured output"""
    prompt = create_prompt(examples, query)
    
    try:
        result = structured_llm.invoke(prompt)
        return result.intent
    except Exception as e:
        print(f"Error: {e}")
        return "UNKNOWN"

def evaluate(examples, val_texts, val_labels, structured_llm):
    """Evaluate on validation set"""
    correct = 0
    total = len(val_texts)
    
    for i, (text, true_label) in enumerate(zip(val_texts, val_labels)):
        if i % 10 == 0:
            print(f"Progress: {i+1}/{total}")
        
        prediction = predict(examples, text, structured_llm)
        if prediction == true_label:
            correct += 1
    
    accuracy = correct / total
    return accuracy

def main():
    # Set seed for reproducibility - same as BERT model
    set_seed(42)
    
    # Reload environment variables to get the corrected API key
    load_dotenv(override=True)
    
    print("Loading data...")
    train_texts, val_texts, train_labels, val_labels = load_data()
    
    print("Selecting examples...")
    examples = select_examples(train_texts, train_labels, examples_per_class=4)
    
    print(f"Train: {len(train_texts)}, Validation: {len(val_texts)}")
    print(f"Using {sum(len(texts) for texts in examples.values())} examples for few-shot")
    
    # Initialize LangChain structured LLM
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.0)
    structured_llm = llm.with_structured_output(IntentClassification)
    
    # Quick test for sanity check
    test_query = "Do you offer EMI?"
    prediction = predict(examples, test_query, structured_llm)
    print(f"\nQuick test: '{test_query}' → {prediction}")
    
    # Run full evaluation on entire validation set (for BERT comparison)
    print(f"\nRunning full evaluation on {len(val_texts)} validation samples...")
    accuracy = evaluate(examples, val_texts, val_labels, structured_llm)
    
    print(f"\n" + "="*50)
    print("FINAL RESULTS:")
    print(f"Few-Shot Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"BERT Accuracy:     0.9242 (92.42%)")
    
    difference = accuracy - 0.9242
    if difference > 0:
        print(f"🎉 Few-Shot wins by {difference*100:.2f}%!")
    else:
        print(f"📉 BERT wins by {abs(difference)*100:.2f}%")
    print("="*50)

if __name__ == '__main__':
    main() 