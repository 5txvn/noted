#!/usr/bin/env python3
"""
Keyword extraction using trained word embeddings.
This script demonstrates how to extract important keywords from text
using the trained word embeddings and similarity analysis.
"""

import re
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from vectorizer import TextVectorizer

class KeywordExtractor:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text):
        """Clean and preprocess input text"""
        # Remove special characters and normalize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_candidates(self, text, min_length=3):
        """Extract potential keyword candidates from text"""
        # Tokenize and clean
        words = word_tokenize(text.lower())
        
        # Filter candidates
        candidates = []
        for word in words:
            if (word.isalpha() and 
                word not in self.stop_words and 
                len(word) >= min_length):
                # Lemmatize
                lemma = self.lemmatizer.lemmatize(word)
                candidates.append(lemma)
        
        return candidates
    
    def calculate_word_importance(self, candidates, text_sentences):
        """Calculate importance score for each candidate word"""
        word_scores = {}
        
        for candidate in set(candidates):
            if candidate not in self.vectorizer.word_to_idx:
                continue
                
            # Get word vector
            candidate_vector = self.vectorizer.get_word_vector(candidate)
            if candidate_vector is None:
                continue
            
            # Calculate TF (Term Frequency)
            tf = candidates.count(candidate) / len(candidates)
            
            # Calculate semantic centrality (average similarity to other words in text)
            similarities = []
            for other_candidate in set(candidates):
                if (other_candidate != candidate and 
                    other_candidate in self.vectorizer.word_to_idx):
                    other_vector = self.vectorizer.get_word_vector(other_candidate)
                    if other_vector is not None:
                        similarity = cosine_similarity([candidate_vector], [other_vector])[0][0]
                        similarities.append(similarity)
            
            # Average similarity (semantic centrality)
            semantic_centrality = np.mean(similarities) if similarities else 0
            
            # Combined score (you can adjust weights)
            importance_score = 0.6 * tf + 0.4 * semantic_centrality
            word_scores[candidate] = importance_score
        
        return word_scores
    
    def extract_keywords(self, text, top_k=10):
        """Extract top keywords from text"""
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Split into sentences for better processing
        sentences = sent_tokenize(cleaned_text)
        
        # Extract candidates
        all_candidates = []
        for sentence in sentences:
            candidates = self.extract_candidates(sentence)
            all_candidates.extend(candidates)
        
        if not all_candidates:
            return []
        
        # Calculate importance scores
        word_scores = self.calculate_word_importance(all_candidates, sentences)
        
        # Sort by importance score
        sorted_keywords = sorted(word_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # Return top k keywords
        return sorted_keywords[:top_k]
    
    def extract_keywords_with_similarity(self, text, top_k=10):
        """Extract keywords and find similar words for each"""
        keywords = self.extract_keywords(text, top_k)
        
        results = []
        for keyword, score in keywords:
            # Find similar words
            similar_words = self.vectorizer.find_similar_words(keyword, top_k=5)
            
            results.append({
                'keyword': keyword,
                'score': score,
                'similar_words': similar_words
            })
        
        return results

def main():
    # Load the trained vectorizer
    vectorizer = TextVectorizer()
    if not vectorizer.load_model():
        print("Please train the model first by running main.py")
        return
    
    # Initialize keyword extractor
    extractor = KeywordExtractor(vectorizer)
    
    # Example texts
    example_texts = [
        """
        Machine learning is a subset of artificial intelligence that focuses on 
        algorithms and statistical models that enable computers to perform tasks 
        without explicit programming. Deep learning, a subset of machine learning, 
        uses neural networks with multiple layers to analyze various factors of data.
        """,
        
        """
        Natural language processing (NLP) is a field of artificial intelligence 
        that gives machines the ability to read, understand, and derive meaning 
        from human languages. It combines computational linguistics with machine 
        learning, deep learning, and statistical models.
        """,
        
        """
        Computer science is the study of computers and computational systems. 
        Unlike electrical and computer engineers, computer scientists deal mostly 
        with software and software systems, including their theory, design, 
        development, and application.
        """
    ]
    
    print("=== Keyword Extraction Demo ===\n")
    
    for i, text in enumerate(example_texts, 1):
        print(f"Text {i}:")
        print(text.strip())
        print("\nExtracted Keywords:")
        
        keywords = extractor.extract_keywords(text, top_k=8)
        for keyword, score in keywords:
            print(f"  {keyword}: {score:.3f}")
        
        print("\nKeywords with Similar Words:")
        results = extractor.extract_keywords_with_similarity(text, top_k=5)
        for result in results:
            print(f"\n  {result['keyword']} (score: {result['score']:.3f}):")
            for similar in result['similar_words'][:3]:
                print(f"    - {similar['word']}: {similar['similarity']:.3f}")
        
        print("\n" + "="*50 + "\n")
    
    # Interactive mode
    print("Interactive Keyword Extraction")
    print("Enter text to extract keywords (or 'quit' to exit):")
    
    while True:
        user_text = input("\nEnter text: ").strip()
        
        if user_text.lower() == 'quit':
            break
        
        if user_text:
            print("\nExtracted Keywords:")
            keywords = extractor.extract_keywords(user_text, top_k=10)
            for keyword, score in keywords:
                print(f"  {keyword}: {score:.3f}")

if __name__ == "__main__":
    main() 