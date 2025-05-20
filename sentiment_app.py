import streamlit as st
from transformers import pipeline
import time  # Import the time module

# First Streamlit command must be set_page_config
st.set_page_config(page_title="🌍 Multilingual Sentiment Detector", page_icon="🔍")

# Load the multilingual sentiment analysis model
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

sentiment_pipeline = load_model()

# Streamlit UI
st.title("Multilingual Sentiment Detection App")
st.write("Analyze the sentiment of text in multiple languages!")

# Text input
user_input = st.text_area("Enter text here:", height=150)

# Analyze button
if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        with st.spinner('Analyzing...'):
            start_time = time.time()  # Time before prediction
            result = sentiment_pipeline(user_input)[0]
            end_time = time.time()    # Time after prediction

            label = result['label']
            score = result['score']

            # Display sentiment result
            if "positive" in label.lower():
                st.success(f"😄 Wonderful!I sense some positive vibes (Confidence: {score:.2f})")
            elif "negative" in label.lower():
                st.error(f"😔 Oh shoot! That's certainly very bad(Confidence: {score:.2f})")
            else:
                st.info(f"🙂 Hmm!I'll Save this info for later (Confidence: {score:.2f})")

            # Display time taken
            elapsed_time = end_time - start_time
            st.markdown(f"⏱️ **Time taken for analysis:** {elapsed_time:.2f} seconds")

    else:
        st.warning("Please enter some text first!")

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align:center;">Built using Streamlit and Hugging Face Transformers</p>
    """,
    unsafe_allow_html=True
)
