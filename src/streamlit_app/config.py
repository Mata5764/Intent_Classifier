# Configuration for Streamlit App

import os

# Paths
BERT_MODEL_PATH = "src/bert_model/best_model_three_layers.pt"
LABEL_MAPPINGS_PATH = "data/json/label2id.json"
TRAINING_DATA_PATH = "data/sofmattress_train.csv"

# Model Information
BERT_ACCURACY = 92.42
FEW_SHOT_ACCURACY = 95.45
BERT_TRAINING_EXAMPLES = 262
FEW_SHOT_EXAMPLES = 84

# App Settings
APP_TITLE = "🛏️ SOF Mattress Intent Classification"
APP_SUBTITLE = "Compare BERT vs Few-Shot Learning approaches" 