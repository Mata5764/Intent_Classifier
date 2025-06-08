# Intent Classification for Customer Support

This project classifies customer queries into different intent categories (like EMI, WARRANTY, etc.) using two different approaches: a fine-tuned BERT model and a few-shot learning approach with GPT.

## Results
- **BERT Model**: 92.42% accuracy
- **Few-Shot Model**: 95.45% accuracy

## What's Included
- Trained BERT model for intent classification
- Few-shot learning approach using OpenAI API  
- Web app to compare both models side-by-side
- Dataset: 329 customer queries across 21 intent categories

## Quick Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Add OpenAI API key** (create `.env` file):
```
OPENAI_API_KEY=your_api_key_here
```

3. **Run the demo:**
```bash
streamlit run src/streamlit_app/app.py
```

Open http://localhost:8501 in your browser.

## Live Demo
🚀 **[Try the live demo here](https://your-app-url.streamlit.app)** (Temporary - for evaluation period)

*Note: The live demo includes both BERT and Few-Shot models for comparison.*

## How to Test
1. Enter a customer query like "Do you offer EMI?"
2. Click "Classify Intent" 
3. See predictions from both models with confidence scores

## Reproducing Training Results

**Train BERT model:**
```bash
python src/bert_model/1l.py
```

**Test few-shot approach:**
```bash
python src/llm_model/few_shot.py
```

## Files Structure
```
├── data/sofmattress_train.csv          # Training data
├── src/bert_model/1l.py                # BERT training
├── src/llm_model/few_shot.py           # Few-shot evaluation  
├── src/streamlit_app/                  # Web demo
└── requirements.txt                    # Dependencies
```

## Key Features
- **Two approaches compared**: Traditional ML vs Modern LLM
- **Production ready**: Clean web interface for testing
- **Reproducible**: Fixed random seeds, clear instructions
- **Real dataset**: Actual customer support queries

Both models work independently - you can test with just BERT if you don't have an OpenAI API key. 