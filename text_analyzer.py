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

# Download stopwords
nltk.download('stopwords', quiet=True)

# Load spaCy model
nlp = en_core_web_sm.load()

class EnhancedTextAnalyzer:
    def __init__(self, text):
        self.text = text
        self.words = self.text.lower().split()
        self.sentences = re.findall(r'\w+[.!?]', self.text)
        self.stop_words = set(stopwords.words('english'))

    def word_frequency(self, n=10, include_stopwords=False):
        if include_stopwords:
            return Counter(self.words).most_common(n)
        else:
            return Counter(word for word in self.words if word not in self.stop_words).most_common(n)

    def generate_wordcloud(self):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(self.text)
        return wordcloud

    def sentiment_analysis(self):
        return TextBlob(self.text).sentiment.polarity

    def named_entity_recognition(self):
        doc = nlp(self.text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def readability_score(self):
        return TextBlob(self.text).sentiment.polarity

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
    if score > 0.5:
        return "The text is relatively easy to read and understand."
    elif score > 0:
        return "The text is somewhat easy to read and understand."
    elif score == 0:
        return "The text has a moderate level of readability."
    elif score > -0.5:
        return "The text may be somewhat difficult to read and understand."
    else:
        return "The text may be very difficult to read and understand."

def main():
    st.set_page_config(page_title="Improved NLP Text Analyzer", layout="wide")

    st.title("📊 Improved NLP Text Analyzer")

    # File upload or text input
    upload_option = st.radio("Choose input method:", ["Upload File", "Paste Text"])

    if upload_option == "Upload File":
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
        if uploaded_file is not None:
            text = uploaded_file.getvalue().decode("utf-8")
        else:
            text = ""
    else:
        text = st.text_area("Enter your text here", height=200)

    if text:
        analyzer = EnhancedTextAnalyzer(text)

        st.header("📝 Text Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Words", len(analyzer.words))
        col2.metric("Unique Words", len(set(analyzer.words)))
        col3.metric("Sentences", len(analyzer.sentences))

        st.subheader("Word Cloud")
        wordcloud = analyzer.generate_wordcloud()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        st.header("📊 Word Frequency Analysis")
        n = st.slider("Select number of top words", 5, 50, 10)
        include_stopwords = st.checkbox("Include stop words")
        word_freq = analyzer.word_frequency(n, include_stopwords)
        df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
        fig = px.bar(df, x='Word', y='Frequency', title=f"Top {n} Most Frequent Words {'(Excluding Stop Words)' if not include_stopwords else ''}")
        st.plotly_chart(fig)

        st.header("😊😐😠 Sentiment Analysis")
        sentiment = analyzer.sentiment_analysis()
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sentiment,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Score"},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [-1, -0.5], 'color': "red"},
                    {'range': [-0.5, 0.5], 'color': "yellow"},
                    {'range': [0.5, 1], 'color': "green"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': sentiment}}))
        st.plotly_chart(fig)
        st.write(f"Interpretation: {interpret_sentiment(sentiment)}")

        st.header("🏷️ Named Entity Recognition")
        st.write("Named Entity Recognition (NER) is the process of identifying and categorizing key information (entities) in text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.")
        entities = analyzer.named_entity_recognition()
        df = pd.DataFrame(entities, columns=['Entity', 'Label'])
        st.table(df)

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
            mode = "gauge+number",
            value = readability_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Readability Score"},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [-1, -0.5], 'color': "red"},
                    {'range': [-0.5, 0.5], 'color': "yellow"},
                    {'range': [0.5, 1], 'color': "green"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': readability_score}}))
        st.plotly_chart(fig)
        st.write(f"Interpretation: {interpret_readability(readability_score)}")

    else:
        st.info("Please upload a file or enter some text to begin the analysis.")

    st.markdown("---")
    st.markdown("Created with ❤️ by Your Name")
    st.markdown("Connect with me on [LinkedIn](https://www.linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)")

if __name__ == "__main__":
    main()