import streamlit as st
from src.analyzer import SentimentAnalyzer

st.set_page_config(page_title="Text & Emoji Sentiment", page_icon="🎭", layout="centered")

@st.cache_resource
def get_analyzer():
    return SentimentAnalyzer()

analyzer = get_analyzer()

st.title("🎭 Text and Emoji Sentiment Analyzer")
st.markdown("Analyze the sentiment of text and emojis using our Custom Trained Model. "
            "This tool provides a combined sentiment score that reflects the overall polarity of your input based on a custom Emoji Lexicon and TF-IDF Scikit-Learn Text Model.")

user_input = st.text_area("Enter your text (with emojis!):", "I love this product! 😍 But the shipping was slow 😞", height=150)

if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            result = analyzer.analyze(user_input)
            
            st.subheader("Extracted Components")
            col1, col2 = st.columns(2)
            with col1:
                text_content = result['text_only'] if result['text_only'] else "No text detected."
                st.info(f"**Text Component:**\n\n{text_content}")
            with col2:
                emoji_content = result['emojis'] if result['emojis'] else "No emojis detected."
                st.info(f"**Emoji Component:**\n\n{emoji_content}")
                
            st.subheader("Combined Sentiment Scores")
            st.markdown("These scores are calculated using a custom ML Pipeline separating text and emojis.")
            
            scores = result['custom_scores']
            
            # Display metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Compound", f"{scores['compound']:.3f}", help="Overall normalized sentiment score between -1 and 1")
            m2.metric("Positive", f"{scores['pos']:.3f}", help="Proportion of positive sentiment")
            m3.metric("Neutral", f"{scores['neu']:.3f}", help="Proportion of neutral sentiment")
            m4.metric("Negative", f"{scores['neg']:.3f}", help="Proportion of negative sentiment")
            
            st.markdown("---")
            
            # Sentiment interpretation based on compound score
            compound = scores['compound']
            if compound >= 0.05:
                st.success("🌟 Overall Sentiment: **Positive**")
            elif compound <= -0.05:
                st.error("💔 Overall Sentiment: **Negative**")
            else:
                st.warning("⚖️ Overall Sentiment: **Neutral**")
                
            # Sarcasm detection info
            if result['is_sarcastic']:
                st.error(f"🚨 **Potential Sarcasm Detected!**\n\nReason: {result['sarcasm_reason']}")
            else:
                st.info("✅ No obvious sarcasm detected.")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.caption("Developed for Sentiment Analysis evaluating words and emojis. Powered by a Custom Scikit-Learn Model and Streamlit.")
