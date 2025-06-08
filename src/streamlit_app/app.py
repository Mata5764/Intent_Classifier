"""
SOF Mattress Intent Classification App
Compare BERT vs Few-Shot Learning approaches
"""

import streamlit as st
import time
from config import *
from bert_predictor import get_bert_predictor
from few_shot_predictor import get_few_shot_predictor

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_models():
    """Load both models with caching"""
    if 'models_loaded' not in st.session_state:
        with st.spinner("Loading models... This may take a moment."):
            try:
                st.session_state['bert_predictor'] = get_bert_predictor()
                st.session_state['few_shot_predictor'] = get_few_shot_predictor()
                st.session_state['models_loaded'] = True
                st.success("✅ Both models loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading models: {e}")
                st.stop()

def main():
    st.title(APP_TITLE)
    st.markdown(f"**{APP_SUBTITLE}**")
    
    # Load models
    load_models()
    
    # Sidebar with model information
    with st.sidebar:
        st.header("📊 Model Comparison")
        
        st.subheader("🤖 BERT Model")
        st.write(f"• **Accuracy:** {BERT_ACCURACY}%")
        st.write(f"• **Training Examples:** {BERT_TRAINING_EXAMPLES}")
        st.write(f"• **Type:** Fine-tuned")
        
        st.subheader("🧠 Few-Shot Model")
        st.write(f"• **Accuracy:** {FEW_SHOT_ACCURACY}%")
        st.write(f"• **Training Examples:** {FEW_SHOT_EXAMPLES}")
        st.write(f"• **Type:** GPT-4 + Prompting")
    
    # Main input section
    st.header("💬 Enter Customer Query")
    
    # Text input
    user_input = st.text_area(
        "Customer query:",
        placeholder="Enter a customer query here...",
        height=100
    )
    
    if st.button("🔍 Classify Intent", type="primary") and user_input.strip():
        
        # Create two columns for side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 BERT Prediction")
            with st.spinner("BERT is thinking..."):
                bert_result = st.session_state['bert_predictor'].predict(user_input)
                
            if 'error' in bert_result:
                st.error(f"Error: {bert_result['error']}")
            else:
                st.success(f"**Intent:** {bert_result['intent']}")
                st.info(f"**Confidence:** {bert_result['confidence']}%")
                st.caption(f"**Time:** {bert_result['prediction_time']}ms")
                st.caption(f"**Model:** {bert_result['model_info']}")
        
        with col2:
            st.subheader("🧠 Few-Shot Prediction")
            with st.spinner("GPT-4 is thinking..."):
                few_shot_result = st.session_state['few_shot_predictor'].predict(user_input)
                
            if 'error' in few_shot_result:
                st.error(f"Error: {few_shot_result['error']}")
            else:
                st.success(f"**Intent:** {few_shot_result['intent']}")
                st.info(f"**Confidence:** {few_shot_result['confidence']}")
                st.caption(f"**Time:** {few_shot_result['prediction_time']}ms")
                st.caption(f"**Model:** {few_shot_result['model_info']}")
        
        # Comparison section
        if 'error' not in bert_result and 'error' not in few_shot_result:
            st.header("⚖️ Comparison")
            
            # Agreement check
            if bert_result['intent'] == few_shot_result['intent']:
                st.success("✅ **Both models agree!**")
            else:
                st.warning("⚠️ **Models disagree!**")
            
            # Performance comparison
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("BERT Speed", f"{bert_result['prediction_time']}ms")
            
            with col2:
                st.metric("Few-Shot Speed", f"{few_shot_result['prediction_time']}ms")
            
            with col3:
                faster_model = "BERT" if bert_result['prediction_time'] < few_shot_result['prediction_time'] else "Few-Shot"
                st.metric("Faster Model", faster_model)
    
    # Instructions
    elif not user_input.strip():
        st.info("👆 Enter a customer query above and click 'Classify Intent' to see predictions from both models.")
    
    # Footer
    st.markdown("---")
    st.markdown("**About:** This app compares traditional fine-tuned BERT vs modern few-shot GPT-4 for intent classification on SOF Mattress customer queries.")

if __name__ == "__main__":
    main() 