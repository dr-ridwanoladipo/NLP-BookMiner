import streamlit as st
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import spacy
import en_core_web_sm
from nltk.corpus import stopwords
import nltk
import PyPDF2
import io
import string
import textstat

# Download stopwords
nltk.download('stopwords', quiet=True)

# Load spaCy model
nlp = en_core_web_sm.load()

class EnhancedTextAnalyzer:
    def __init__(self, text):
        self.text = text
        self.words = self.preprocess_text(self.text)
        self.sentences = re.findall(r'\w+[.!?]', self.text)
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        # Remove punctuation and convert to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        return text.split()

    def word_frequency(self, n=10, include_stopwords=False):
        if include_stopwords:
            return Counter(self.words).most_common(n)
        else:
            return Counter(word for word in self.words if word not in self.stop_words).most_common(n)

    def generate_wordcloud(self):
        try:
            if not self.text.strip():
                raise ValueError("The text is empty")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(self.text)
            return wordcloud
        except ValueError as e:
            st.warning(f"Unable to generate word cloud: {str(e)}")
            return None

    def sentiment_analysis(self):
        return TextBlob(self.text).sentiment.polarity

    def named_entity_recognition(self):
        doc = nlp(self.text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def readability_score(self):
        return textstat.flesch_reading_ease(self.text) / 100  # Normalize to 0-1 range

    def find_sentences_with_word(self, word):
        pattern = re.compile(fr'\b{re.escape(word)}\b', re.IGNORECASE)
        sentences = re.split(r'(?<=[.!?])\s+', self.text)
        return [sentence for sentence in sentences if pattern.search(sentence)]

    def find_paragraphs_with_word(self, word):
        pattern = re.compile(fr'\b{re.escape(word)}\b', re.IGNORECASE)
        paragraphs = re.split(r'\n\s*\n', self.text)
        return [para for para in paragraphs if pattern.search(para)]

    def word_count(self, word):
        pattern = re.compile(fr'\b{re.escape(word)}\b', re.IGNORECASE)
        return len(pattern.findall(self.text))

def interpret_sentiment(score):
    if score > 0.5:
        return "Very Positive"
    elif score > 0:
        return "Positive"
    elif score == 0:
        return "Neutral"
    elif score > -0.5:
        return "Negative"
    else:
        return "Very Negative"

def interpret_readability(score):
    if score > 0.8:
        return "The text is very easy to read and understand."
    elif score > 0.6:
        return "The text is easy to read and understand."
    elif score > 0.4:
        return "The text has a moderate level of readability."
    elif score > 0.2:
        return "The text may be somewhat difficult to read and understand."
    else:
        return "The text may be very difficult to read and understand."

def read_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            st.warning("The PDF appears to be empty or contains only scanned images. Text extraction may be limited.")

        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def main():
    st.set_page_config(page_title="NLP Text Analyzer", layout="wide")

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stExpander {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #2c3e50;
        color: white;
        text-align: center;
        padding: 10px 0;
        font-weight: bold;
    }
    .user-guide {
        border-left: 5px solid #3498db;
        padding-left: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧠 NLP Text Analyzer")

    # Add space after the title
    st.markdown("<br>", unsafe_allow_html=True)

    # User Guide
    with st.expander("📘 User Guide"):
        st.markdown("""
        <div class="user-guide">
        Welcome to the NLP Text Analyzer! Here's a quick guide to get you started:

        1. **Input**: Choose to either upload a file (TXT or PDF) or paste text directly.
        2. **Text Overview**: View basic statistics and a word cloud of your text.
        3. **Word Frequency**: Analyze the most common words, with an option to include/exclude stop words.
        4. **Sentiment Analysis**: Understand the overall sentiment of your text.
           - Sentiment Score ranges from -1 (very negative) to +1 (very positive).
           - Scores interpretation:
             * -1.0 to -0.6: Very Negative
             * -0.6 to -0.1: Negative
             * -0.1 to +0.1: Neutral
             * +0.1 to +0.6: Positive
             * +0.6 to +1.0: Very Positive
        5. **Named Entity Recognition**: Identify key entities in your text.
        6. **Word Search**: Look for specific words and their context within the text.
        7. **Readability Analysis**: Get insights into how easy or difficult your text is to read.
           - Readability Score ranges from 0 (very difficult) to 1 (very easy).
           - Scores interpretation:
             * 0.0 to 0.2: Very difficult to read
             * 0.2 to 0.4: Difficult to read
             * 0.4 to 0.6: Moderately easy to read
             * 0.6 to 0.8: Easy to read
             * 0.8 to 1.0: Very easy to read

        Explore each section to gain valuable insights into your text!
        </div>
        """, unsafe_allow_html=True)

    # Add space after the user guide
    st.markdown("<br>", unsafe_allow_html=True)

    # File upload or text input
    upload_option = st.radio("Choose input method:", ["Upload File", "Paste Text"])

    if upload_option == "Upload File":
        uploaded_file = st.file_uploader("Choose a text or PDF file", type=["txt", "pdf"])
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "application/pdf":
                    text = read_pdf(uploaded_file)
                else:
                    text = uploaded_file.getvalue().decode("utf-8")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                text = ""
        else:
            text = ""
    else:
        text = st.text_area("Enter your text here", height=200)

    if text:
        try:
            analyzer = EnhancedTextAnalyzer(text)

            st.header("📝 Text Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Words", len(analyzer.words))
            col2.metric("Unique Words", len(set(analyzer.words)))
            col3.metric("Sentences", len(analyzer.sentences))

            st.subheader("Word Cloud")
            wordcloud = analyzer.generate_wordcloud()
            if wordcloud:
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            else:
                st.info("Word cloud could not be generated due to insufficient text content.")

            st.header("📊 Word Frequency Analysis")
            n = st.slider("Select number of top words", 5, 50, 10)
            include_stopwords = st.checkbox("Include stop words")
            word_freq = analyzer.word_frequency(n, include_stopwords)
            if word_freq:
                df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
                fig = px.bar(df, x='Word', y='Frequency',
                             title=f"Top {n} Most Frequent Words {'(Excluding Stop Words)' if not include_stopwords else ''}",
                             color='Frequency', color_continuous_scale='Viridis')
                st.plotly_chart(fig)
            else:
                st.info("No words found for frequency analysis.")

            st.header("😊😐😠 Sentiment Analysis")
            sentiment = analyzer.sentiment_analysis()
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sentiment Score"},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "#3498db"},
                    'steps': [
                        {'range': [-1, -0.5], 'color': "#e74c3c"},
                        {'range': [-0.5, 0.5], 'color': "#f1c40f"},
                        {'range': [0.5, 1], 'color': "#2ecc71"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': sentiment}}))
            st.plotly_chart(fig)
            st.write(f"Interpretation: {interpret_sentiment(sentiment)}")

            st.header("🏷️ Named Entity Recognition")
            entities = analyzer.named_entity_recognition()
            if entities:
                df = pd.DataFrame(entities, columns=['Entity', 'Label'])
                with st.expander("View Named Entities"):
                    st.table(df)
            else:
                st.info("No named entities found in the text.")

            st.header("🔍 Word Search")
            search_word = st.text_input("Enter a word to search in the text:")
            if search_word:
                word_count = analyzer.word_count(search_word)
                st.write(f"Frequency of '{search_word}': {word_count}")

                sentences = analyzer.find_sentences_with_word(search_word)
                paragraphs = analyzer.find_paragraphs_with_word(search_word)

                with st.expander(f"Sentences containing '{search_word}' ({len(sentences)})"):
                    for sentence in sentences:
                        st.write(f"- {sentence}")

                with st.expander(f"Paragraphs containing '{search_word}' ({len(paragraphs)})"):
                    for para in paragraphs:
                        st.write(f"- {para}\n")

            st.header("📖 Readability Analysis")
            readability_score = analyzer.readability_score()
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=readability_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Readability Score"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#3498db"},
                    'steps': [
                        {'range': [0, 0.2], 'color': "#e74c3c"},
                        {'range': [0.2, 0.4], 'color': "#e67e22"},
                        {'range': [0.4, 0.6], 'color': "#f1c40f"},
                        {'range': [0.6, 0.8], 'color': "#2ecc71"},
                        {'range': [0.8, 1], 'color': "#27ae60"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': readability_score}}))
            st.plotly_chart(fig)
            st.write(f"Interpretation: {interpret_readability(readability_score)}")

        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {str(e)}")
            st.info("Please try again with different text or contact support if the issue persists.")

    else:
        st.info("Please upload a file or enter some text to begin the analysis.")

    st.markdown(
        """
        <div class="footer">
        © 2024 All Rights Reserved | Dr. Ridwan Oladipo
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()