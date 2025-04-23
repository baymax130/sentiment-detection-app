import streamlit as st
from transformers import pipeline

# Load the multilingual sentiment analysis model
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

sentiment_pipeline = load_model()

# Streamlit UI setup
st.set_page_config(page_title="🌍 Multilingual Sentiment Detector", page_icon="🔍")

st.title("🔍 Multilingual Sentiment Detection App")
st.write("Analyze the sentiment of text in multiple languages!")

# Text input
user_input = st.text_area("Enter text here:", height=150)

# Analyze button
if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        with st.spinner('Analyzing...'):
            result = sentiment_pipeline(user_input)[0]
            label = result['label']
            score = result['score']

            # Display result
            if "positive" in label.lower():
                st.success(f"😄 Positive Sentiment! (Confidence: {score:.2f})")
            elif "negative" in label.lower():
                st.error(f"😔 Negative Sentiment! (Confidence: {score:.2f})")
            else:
                st.info(f"🙂 Neutral Sentiment! (Confidence: {score:.2f})")
    else:
        st.warning("Please enter some text first!")

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align:center;">Built with ❤️ using Streamlit and Hugging Face Transformers</p>
    """,
    unsafe_allow_html=True
)
