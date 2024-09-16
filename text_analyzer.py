import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from collections import Counter


class TextAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.book_content = self.load_book()
        self.chapters = self.split_chapters()

    def load_book(self):
        """Load the book content from the file."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' not found.")
            return None
        except IOError:
            print(f"Error: Unable to read file '{self.file_path}'.")
            return None

    def split_chapters(self):
        """Split the book into chapters."""
        if not self.book_content:
            return []
        pattern = re.compile(r"Chapter \d+")
        return re.split(pattern, self.book_content)[1:]  # Exclude content before first chapter

    def count_chapters(self):
        """Count the number of chapters in the book."""
        return len(self.chapters)

    def find_sentences_with_word(self, word):
        """Find sentences containing a specific word."""
        pattern = re.compile(fr"[A-Z][^.]*\b{word}\b[^.]*\.", re.IGNORECASE)
        return re.findall(pattern, self.book_content)

    def find_paragraphs_with_word(self, word):
        """Find paragraphs containing a specific word."""
        pattern = re.compile(fr"[^\n]+\b{word}\b[^\n]+", re.IGNORECASE)
        return re.findall(pattern, self.book_content)

    def get_chapter_titles(self):
        """Extract chapter titles from the book."""
        pattern = re.compile(r"Chapter \d+\s+([a-zA-Z ,]+)")
        return re.findall(pattern, self.book_content)

    def most_common_words(self, n=10, exclude_stopwords=True):
        """Find the n most common words in the book."""
        words = re.findall(r"\b[a-zA-Z]+\b", self.book_content.lower())
        if exclude_stopwords:
            stop_words = set(stopwords.words("english"))
            words = [word for word in words if word not in stop_words]
        return Counter(words).most_common(n)

    def word_frequency(self, word):
        """Find the frequency of a specific word in the book."""
        words = re.findall(r"\b[a-zA-Z]+\b", self.book_content.lower())
        return Counter(words)[word.lower()]

    def sentiment_analysis(self):
        """Perform sentiment analysis on each chapter."""
        analyzer = SentimentIntensityAnalyzer()
        sentiments = []
        for i, chapter in enumerate(self.chapters, 1):
            sentiment = analyzer.polarity_scores(chapter)
            sentiments.append((i, sentiment))
        return sentiments

    def visualize_sentiment(self):
        """Visualize the sentiment analysis results."""
        sentiments = self.sentiment_analysis()
        chapters, compounds = zip(*[(c, s['compound']) for c, s in sentiments])

        plt.figure(figsize=(12, 6))
        plt.bar(chapters, compounds)
        plt.title('Sentiment Analysis by Chapter')
        plt.xlabel('Chapter')
        plt.ylabel('Compound Sentiment Score')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.show()

    def visualize_word_frequency(self, n=10):
        """Visualize the n most common words."""
        words, counts = zip(*self.most_common_words(n))

        plt.figure(figsize=(12, 6))
        plt.bar(words, counts)
        plt.title(f'Top {n} Most Common Words')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# Usage example
if __name__ == "__main__":
    analyzer = TextAnalyzer("miracle_in_the_andes.txt")

    print(f"Number of chapters: {analyzer.count_chapters()}")

    love_sentences = analyzer.find_sentences_with_word("love")
    print(f"Number of sentences containing 'love': {len(love_sentences)}")

    chapter_titles = analyzer.get_chapter_titles()
    print("Chapter titles:", chapter_titles)

    print("Most common words:", analyzer.most_common_words())

    print("Frequency of 'mountain':", analyzer.word_frequency("mountain"))

    analyzer.visualize_sentiment()
    analyzer.visualize_word_frequency()