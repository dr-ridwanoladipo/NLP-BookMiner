# NLP Text Analyzer

This project is a Natural Language Processing (NLP) tool designed to analyze text documents, specifically focused on books. It demonstrates various text analysis techniques including word frequency analysis, sentiment analysis, and basic text mining using regular expressions.

## Features

- Chapter counting and title extraction
- Word frequency analysis
- Sentence and paragraph search for specific words
- Sentiment analysis by chapter
- Visualization of word frequency and sentiment analysis results

## Requirements

- Python 3.12
- NLTK
- Matplotlib

## Installation

1. Clone this repository:
 

2. Install the required packages:
   ```
   pip install nltk matplotlib
   ```

3. Download necessary NLTK data:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('vader_lexicon')
   ```

## Usage

1. Place your text file (e.g., "miracle_in_the_andes.txt") in the project directory.

2. Run the script:
   ```
   python text_analyzer.py
   ```

3. The script will output various analyses and generate visualizations.

## Example Output

```
Number of chapters: 10
Number of sentences containing 'love': 15
Chapter titles: ['The Crash', 'Aftermath', 'Hope Fades', ...]
Most common words: [('the', 1420), ('and', 982), ('to', 725), ...]
Frequency of 'mountain': 67
```

The script will also generate two plots:
1. A bar chart showing the sentiment analysis results by chapter.
2. A bar chart showing the frequency of the most common words.

## Customization

You can modify the `TextAnalyzer` class in `text_analyzer.py` to add more analysis methods or change existing ones. The main script at the bottom of the file can be adjusted to run different analyses or create custom visualizations.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/yourusername/nlp-text-analyzer/issues) if you want to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.