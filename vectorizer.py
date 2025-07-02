import re
import nltk
import numpy as np
import json
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class TextVectorizer:
    def __init__(self, min_word_freq=5, window_size=5, vector_dim=100):
        self.min_word_freq = min_word_freq
        self.window_size = window_size
        self.vector_dim = vector_dim
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_vectors = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove special characters and normalize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_and_clean(self, sentences):
        """Tokenize sentences and clean words"""
        all_words = []
        for sentence in sentences:
            # Tokenize
            words = word_tokenize(sentence.lower())
            
            # Clean and filter words
            cleaned_words = []
            for word in words:
                # Remove non-alphabetic words and stop words
                if word.isalpha() and word not in self.stop_words and len(word) > 2:
                    # Lemmatize
                    lemma = self.lemmatizer.lemmatize(word)
                    cleaned_words.append(lemma)
            
            all_words.extend(cleaned_words)
        
        return all_words
    
    def build_vocabulary(self, sentences):
        """Build vocabulary from sentences"""
        print("Building vocabulary...")
        all_words = self.tokenize_and_clean(sentences)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Filter by minimum frequency
        vocab = [word for word, count in word_counts.items() 
                if count >= self.min_word_freq]
        
        # Create word to index mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"Vocabulary size: {len(vocab)}")
        return vocab
    
    def build_cooccurrence_matrix(self, sentences):
        """Build co-occurrence matrix from sentences"""
        print("Building co-occurrence matrix...")
        vocab_size = len(self.word_to_idx)
        cooccurrence_matrix = np.zeros((vocab_size, vocab_size))
        
        for sentence in sentences:
            words = self.tokenize_and_clean([sentence])
            word_indices = [self.word_to_idx[word] for word in words 
                          if word in self.word_to_idx]
            
            # Build co-occurrence matrix
            for i, word_idx in enumerate(word_indices):
                # Look at words within window_size
                start = max(0, i - self.window_size)
                end = min(len(word_indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        cooccurrence_matrix[word_idx][word_indices[j]] += 1
        
        return cooccurrence_matrix
    
    def train_embeddings(self, sentences):
        """Train word embeddings using SVD on co-occurrence matrix"""
        print("Training word embeddings...")
        
        # Build vocabulary
        vocab = self.build_vocabulary(sentences)
        
        # Build co-occurrence matrix
        cooccurrence_matrix = self.build_cooccurrence_matrix(sentences)
        
        # Apply SVD to get word vectors
        from sklearn.decomposition import TruncatedSVD
        
        # Add small constant to avoid log(0)
        cooccurrence_matrix = cooccurrence_matrix + 1
        
        # Apply log transformation
        log_cooccurrence = np.log(cooccurrence_matrix)
        
        # Apply SVD
        svd = TruncatedSVD(n_components=self.vector_dim, random_state=42)
        self.word_vectors = svd.fit_transform(log_cooccurrence)
        
        print(f"Word vectors shape: {self.word_vectors.shape}")
        return self.word_vectors
    
    def get_word_vector(self, word):
        """Get vector for a specific word"""
        if word in self.word_to_idx:
            return self.word_vectors[self.word_to_idx[word]]
        else:
            return None
    
    def find_similar_words(self, word, top_k=10):
        """Find most similar words to given word"""
        word_vector = self.get_word_vector(word)
        if word_vector is None:
            return []
        
        # Calculate cosine similarities
        similarities = cosine_similarity([word_vector], self.word_vectors)[0]
        
        # Get top k similar words
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]  # Skip the word itself
        
        similar_words = []
        for idx in similar_indices:
            similar_words.append({
                'word': self.idx_to_word[idx],
                'similarity': similarities[idx]
            })
        
        return similar_words
    
    def visualize_words(self, words, figsize=(12, 8)):
        """Visualize word relationships using PCA"""
        # Get vectors for the specified words
        word_vectors = []
        valid_words = []
        
        for word in words:
            vector = self.get_word_vector(word)
            if vector is not None:
                word_vectors.append(vector)
                valid_words.append(word)
        
        if len(word_vectors) < 2:
            print("Need at least 2 valid words to visualize")
            return
        
        word_vectors = np.array(word_vectors)
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        word_vectors_2d = pca.fit_transform(word_vectors)
        
        # Create visualization
        plt.figure(figsize=figsize)
        plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], alpha=0.7)
        
        # Add word labels
        for i, word in enumerate(valid_words):
            plt.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.title('Word Embeddings Visualization (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print explained variance
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")
    
    def save_model(self, filename='word_embeddings.pkl'):
        """Save the trained model"""
        model_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'word_vectors': self.word_vectors,
            'min_word_freq': self.min_word_freq,
            'window_size': self.window_size,
            'vector_dim': self.vector_dim
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='word_embeddings.pkl'):
        """Load a trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.word_to_idx = model_data['word_to_idx']
            self.idx_to_word = model_data['idx_to_word']
            self.word_vectors = model_data['word_vectors']
            self.min_word_freq = model_data['min_word_freq']
            self.window_size = model_data['window_size']
            self.vector_dim = model_data['vector_dim']
            
            print(f"Model loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Model file {filename} not found")
            return False

def clean_text(text):
    """Legacy function for backward compatibility"""
    text = remove_parentheses(text)
    text = remove_section_headers(text)
    text = normalize_whitespace(text)
    sentences = split_into_sentences(text)
    return sentences

def remove_parentheses(text):
    return re.sub(r'\([^)]*\)', '', text)

def remove_section_headers(text):
    return re.sub(r'={2,}.*?={2,}', '', text)

def normalize_whitespace(text):
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_sentences(text):
    return sent_tokenize(text)

# Example usage
if __name__ == "__main__":
    # Load corpus
    try:
        with open('wikipedia_corpus.json', 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        print(f"Loaded corpus with {len(corpus)} sentences")
    except FileNotFoundError:
        print("No corpus found. Please run wikipedia_collector.py first.")
        exit(1)
    
    # Initialize and train vectorizer
    vectorizer = TextVectorizer(min_word_freq=3, window_size=5, vector_dim=50)
    
    # Train embeddings
    vectorizer.train_embeddings(corpus)
    
    # Save model
    vectorizer.save_model()
    
    # Example: find similar words
    test_words = ['computer', 'science', 'learning', 'data']
    for word in test_words:
        similar = vectorizer.find_similar_words(word, top_k=5)
        print(f"\nWords similar to '{word}':")
        for item in similar:
            print(f"  {item['word']}: {item['similarity']:.3f}")
    
    # Visualize some words
    words_to_visualize = ['computer', 'science', 'learning', 'data', 'algorithm', 
                         'machine', 'intelligence', 'programming', 'software', 'database']
    vectorizer.visualize_words(words_to_visualize)
