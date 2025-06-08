# LLM Intent Classification Evaluator
# Simple evaluation script that takes queries and returns predictions

import pandas as pd
import json
import numpy as np
import random
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

class IntentClassification(BaseModel):
    intent: str = Field(description="The exact intent category that best matches the customer query. Must be one of the categories shown in the examples.")

def set_seed(seed=42):
    """Set seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)

def load_examples():
    """Load few-shot examples from training data"""
    df = pd.read_csv('data/sofmattress_train.csv')
    
    # Load existing label mappings
    with open('data/json/label2id.json', 'r') as f:
        label2id = json.load(f)
    
    # Create label IDs for stratified split (same as training)
    df['label_id'] = df['label'].map(label2id)
    
    # Use same split as training to get consistent examples
    from sklearn.model_selection import train_test_split
    train_texts, _, train_labels, _ = train_test_split(
        df['sentence'].values,
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label_id']
    )
    
    # Select 4 examples per class (same as few_shot.py)
    df_train = pd.DataFrame({'text': train_texts, 'label': train_labels})
    examples = {}
    
    for label in df_train['label'].unique():
        label_samples = df_train[df_train['label'] == label]['text'].tolist()
        examples[label] = label_samples[:4]  # Take first 4 examples
    
    return examples

def create_prompt(examples, query):
    """Create few-shot prompt for LLM"""
    prompt = """Classify this SOF Mattress customer query into the exact intent category shown in the examples.

Examples:
"""
    
    # Add examples
    for intent, texts in examples.items():
        for text in texts:
            prompt += f'Customer: "{text}" → Intent: {intent}\n'
    
    # Add query to classify
    prompt += f"""
Customer: "{query}"

Think step by step:
1. What is the customer asking about?
2. Which intent category best matches this query?
3. Provide your reasoning and the exact intent category."""
    
    return prompt

def load_model():
    """Initialize the LLM model"""
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.0)
    structured_llm = llm.with_structured_output(IntentClassification)
    return structured_llm

def predict(examples, queries, structured_llm):
    """Predict intents for a list of queries"""
    predictions = []
    
    for query in queries:
        prompt = create_prompt(examples, query)
        
        try:
            result = structured_llm.invoke(prompt)
            predictions.append(result.intent)
        except Exception as e:
            print(f"Error predicting for query '{query}': {e}")
            predictions.append("UNKNOWN")
    
    return predictions

def predict_single(examples, query, structured_llm):
    """Predict intent for a single query"""
    return predict(examples, [query], structured_llm)[0]

# ----- Main Evaluation Function -----
def evaluate_queries(queries):
    """
    Main function to evaluate a list of queries
    
    Args:
        queries (list): List of query strings
        
    Returns:
        list: List of predicted intent labels
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Load examples and model
    examples = load_examples()
    structured_llm = load_model()
    
    # Get predictions
    predictions = predict(examples, queries, structured_llm)
    
    return predictions

# ----- Run Inference -----
if __name__ == "__main__":
    # Test with sample queries
    test_queries = [
        "I need a mattress for back pain.",
        "Where can I buy a soft bed?",
        "Can I return this mattress?",
        "Do you offer EMI payment options?",
        "What is your warranty policy?"
    ]
    
    print("GPT-4 Few-Shot Intent Classification")
    print("=" * 50)
    
    predictions = evaluate_queries(test_queries)
    
    print("\nResults:")
    for query, prediction in zip(test_queries, predictions):
        print(f"Query: {query}")
        print(f"Predicted Intent: {prediction}")
        print("-" * 30) 