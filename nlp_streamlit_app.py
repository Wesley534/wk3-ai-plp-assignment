import streamlit as st
import spacy
import os
import pandas as pd

# --- Configuration and Setup ---
st.set_page_config(page_title="NLP Analysis with spaCy", layout="wide")
st.title("🤖 Product Review NLP Analyzer (spaCy)")
st.markdown("---")

# --- Load spaCy Model ---
# Use st.cache_resource to load the spaCy model only once
@st.cache_resource
def load_spacy_model():
    """Load the pre-installed spaCy model."""
    model_name = "en_core_web_sm"
    try:
        # Load the model, which is now installed as a Python package via requirements.txt
        nlp = spacy.load(model_name)
        st.success(f"spaCy model '{model_name}' loaded successfully.")
        return nlp
    except Exception as e:
        st.error(f"Failed to load spaCy model '{model_name}'. Check that it is correctly added to requirements.txt. Error: {e}")
        return None

nlp = load_spacy_model()

# --- Analysis Functions (from nlp_analysis.py) ---

def perform_ner(text, nlp_model):
    """Performs NER using spaCy and filters for relevant entities."""
    doc = nlp_model(text)
    entities = []
    
    # Filter for entities that are likely products, brands, or tech
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "NORP", "GPE"]:
            # Simple check for product names with numbers
            if any(char.isdigit() for char in ent.text) or ent.label_ == "ORG":
                entities.append((ent.text, ent.label_))
            # Catch common words that are often brand/product names
            elif ent.text.lower() in ["iphone", "galaxy", "echo dot", "sony", "amazon"]:
                 entities.append((ent.text, "BRAND_NAME"))
        # Add generic entities for completeness
        elif ent.label_ in ["PERSON", "DATE", "MONEY"]:
            entities.append((ent.text, ent.label_))
            
    return entities

def analyze_sentiment(text, nlp_model):
    """Analyzes sentiment using a simple keyword/rule-based approach."""
    
    # Predefined lists of positive and negative keywords
    positive_words = ["love", "amazing", "great", "fantastic", "crisp", "perfectly", "excellent", "best", "good", "happy"]
    negative_words = ["overheats", "regret", "high", "died", "unacceptable", "terrible", "bad", "worse", "poor", "slow"]
    
    positive_score = 0
    negative_score = 0
    
    doc = nlp_model(text.lower())
    
    for token in doc:
        if token.text in positive_words:
            positive_score += 1
        elif token.text in negative_words:
            negative_score += 1
            
    if positive_score > negative_score:
        return "Positive", positive_score, negative_score
    elif negative_score > positive_score:
        return "Negative", positive_score, negative_score
    else:
        return "Neutral/Mixed", positive_score, negative_score

# --- Main App Logic ---

if nlp is not None:
    
    st.sidebar.header("Instructions")
    st.sidebar.markdown(
        """
        1. Paste an Amazon product review into the text area.
        2. Click 'Analyze Review'.
        3. The app will extract Named Entities (product/brand names) and calculate sentiment.
        """
    )

    default_review = "I love the new Sony WH-1000XM5 headphones! The crisp sound quality is amazing, but the price is a bit high. Still, a fantastic purchase."
    
    user_input = st.text_area(
        "Paste an Amazon Product Review Here:",
        default_review,
        height=150
    )

    if st.button('Analyze Review', type="primary"):
        if user_input:
            
            # --- Perform Analysis ---
            entities = perform_ner(user_input, nlp)
            sentiment, pos_score, neg_score = analyze_sentiment(user_input, nlp)
            
            # --- Display NER Results ---
            st.subheader("1. Named Entity Recognition (NER)")
            if entities:
                df_entities = pd.DataFrame(entities, columns=['Entity', 'Label'])
                st.table(df_entities)
            else:
                st.info("No relevant product/brand entities were found.")

            # --- Display Sentiment Results ---
            st.subheader("2. Rule-Based Sentiment Analysis")
            
            if sentiment == "Positive":
                st.success(f"**Result: {sentiment}**")
            elif sentiment == "Negative":
                st.error(f"**Result: {sentiment}**")
            else:
                st.warning(f"**Result: {sentiment}**")
                
            st.markdown(f"**Scores:** Positive Keywords Found: `{pos_score}`, Negative Keywords Found: `{neg_score}`")
            
        else:
            st.warning("Please enter some text to analyze.")