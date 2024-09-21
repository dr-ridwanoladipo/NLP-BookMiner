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

# Load spaCy model
nlp = en_core_web_sm.load()

class SimpleTextAnalyzer:
    def __init__(self, text):
        self.text = text
        self.words = self.text.lower().split()
        self.sentences = re.findall(r'\w+[.!?]', self.text)

    def word_frequency(self, n=10):
        return Counter(self.words).most_common(n)

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

def main():
    st.set_page_config(page_title="Simple NLP Text Analyzer", layout="wide")

    st.title("📊 Simple NLP Text Analyzer")

    text = st.text_area("Enter your text here", height=200)

    if text:
        analyzer = SimpleTextAnalyzer(text)

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
        word_freq = analyzer.word_frequency(n)
        df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
        fig = px.bar(df, x='Word', y='Frequency', title=f"Top {n} Most Frequent Words")
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

        st.header("🏷️ Named Entity Recognition")
        entities = analyzer.named_entity_recognition()
        df = pd.DataFrame(entities, columns=['Entity', 'Label'])
        st.table(df)

    else:
        st.info("Please enter some text to begin the analysis.")

    st.markdown("---")
    st.markdown("Created with ❤️ by Your Name")
    st.markdown("Connect with me on [LinkedIn](https://www.linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)")

if __name__ == "__main__":
    main()