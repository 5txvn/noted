# Text Vectorization Project

A comprehensive system for creating word embeddings from Wikipedia data and visualizing word relationships using PCA.

## Features

- **Wikipedia Data Collection**: Automatically collects and cleans text from Wikipedia pages
- **Word Embeddings**: Creates meaningful word vectors using co-occurrence matrices and SVD
- **Word Similarity**: Finds semantically similar words using cosine similarity
- **Visualization**: Visualizes word relationships in 2D space using PCA
- **Interactive Exploration**: Allows users to explore word similarities interactively

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. The system will automatically download required NLTK data on first run.

## Usage

### Quick Start

Run the main script to see the complete workflow:

```bash
python main.py
```

This will:
1. Collect Wikipedia data from various pages
2. Train word embeddings
3. Demonstrate word similarity
4. Create visualizations
5. Allow interactive exploration

### Individual Components

#### Wikipedia Data Collection

```python
from wikipedia_collector import WikipediaCollector

collector = WikipediaCollector()
pages = ["Artificial intelligence", "Machine learning", "Computer science"]
corpus = collector.collect_corpus(pages)
collector.save_corpus(corpus)
```

#### Word Embeddings

```python
from vectorizer import TextVectorizer

# Initialize vectorizer
vectorizer = TextVectorizer(min_word_freq=3, window_size=5, vector_dim=50)

# Train embeddings
vectorizer.train_embeddings(corpus)

# Find similar words
similar_words = vectorizer.find_similar_words("computer", top_k=10)

# Visualize words
words_to_plot = ["computer", "science", "learning", "data"]
vectorizer.visualize_words(words_to_plot)
```

## How It Works

### 1. Text Preprocessing
- Removes Wikipedia markup, parentheses, and special characters
- Tokenizes text into sentences and words
- Filters out stop words and short words
- Applies lemmatization to normalize word forms

### 2. Co-occurrence Matrix
- Builds a matrix where each cell (i,j) represents how often word i appears near word j
- Uses a sliding window approach to capture word context
- Applies log transformation to reduce the impact of high-frequency words

### 3. Word Embeddings
- Uses Singular Value Decomposition (SVD) to reduce the co-occurrence matrix
- Creates dense word vectors that capture semantic relationships
- Vectors are much smaller than one-hot encodings while preserving meaning

### 4. Visualization
- Uses Principal Component Analysis (PCA) to reduce vectors to 2D
- Plots words in 2D space where similar words appear closer together
- Provides insights into word relationships and semantic clusters

## Configuration

You can customize the system by modifying these parameters:

- `min_word_freq`: Minimum frequency for a word to be included in vocabulary
- `window_size`: Size of context window for co-occurrence calculation
- `vector_dim`: Dimensionality of word vectors
- `delay`: Delay between Wikipedia API requests (to be respectful)

## Example Output

```
Words similar to 'computer':
  programming: 0.892
  software: 0.845
  system: 0.823
  algorithm: 0.801
  data: 0.789
```

## Files

- `main.py`: Main script demonstrating the complete workflow
- `wikipedia_collector.py`: Wikipedia data collection and cleaning
- `vectorizer.py`: Word embedding creation and visualization
- `requirements.txt`: Python dependencies
- `wikipedia_corpus.json`: Collected Wikipedia sentences (generated)
- `word_embeddings.pkl`: Trained word embeddings (generated)

## Extending the System

### Adding New Data Sources
You can extend the system to work with other text sources by:
1. Creating a new data collector class
2. Implementing the same text cleaning pipeline
3. Using the existing `TextVectorizer` class

### Custom Visualizations
The `visualize_words` method can be extended to create different types of visualizations:
- 3D plots using PCA with 3 components
- Interactive plots using libraries like Plotly
- Network graphs showing word relationships

### Advanced Embeddings
For better performance, consider:
- Using pre-trained embeddings like Word2Vec or GloVe
- Implementing neural network-based approaches
- Adding subword information for better handling of rare words

## Troubleshooting

### Common Issues

1. **NLTK Data Not Found**: The system will automatically download required NLTK data on first run.

2. **Wikipedia API Errors**: If you encounter rate limiting, increase the delay parameter in the collector.

3. **Memory Issues**: For large corpora, reduce the vocabulary size by increasing `min_word_freq`.

4. **No Words Found**: Ensure your corpus contains enough text and adjust `min_word_freq` if needed.

## License

This project is open source and available under the MIT License.