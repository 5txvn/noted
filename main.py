#!/usr/bin/env python3
"""
Main script for the text vectorization project.
This script demonstrates the complete workflow:
1. Collect Wikipedia data
2. Train word embeddings
3. Visualize word relationships
"""

import json
import os
from wikipedia_collector import WikipediaCollector
from vectorizer import TextVectorizer

def main():
    print("=== Text Vectorization Project ===\n")
    
    # Step 1: Collect Wikipedia data
    print("Step 1: Collecting Wikipedia data...")
    
    # Define Wikipedia pages to collect from
    pages = [
        "Artificial intelligence",
        "Machine learning", 
        "Natural language processing",
        "Computer science",
        "Mathematics",
        "Physics",
        "Biology",
        "Chemistry",
        "History",
        "Geography",
        "Programming",
        "Data science",
        "Statistics",
        "Algorithm",
        "Database"
    ]
    
    collector = WikipediaCollector()
    
    # Check if corpus already exists
    if os.path.exists('wikipedia_corpus.json'):
        print("Loading existing corpus...")
        corpus = collector.load_corpus()
    else:
        print("Collecting new corpus from Wikipedia...")
        corpus = collector.collect_corpus(pages, delay=1)
        collector.save_corpus(corpus)
    
    if not corpus:
        print("Error: No corpus available. Exiting.")
        return
    
    print(f"Corpus contains {len(corpus)} sentences\n")
    
    # Step 2: Train word embeddings
    print("Step 2: Training word embeddings...")
    
    # Check if model already exists
    vectorizer = TextVectorizer(min_word_freq=3, window_size=5, vector_dim=50)
    
    if os.path.exists('word_embeddings.pkl'):
        print("Loading existing model...")
        if not vectorizer.load_model():
            print("Error loading model. Training new one...")
            vectorizer.train_embeddings(corpus)
            vectorizer.save_model()
    else:
        print("Training new model...")
        vectorizer.train_embeddings(corpus)
        vectorizer.save_model()
    
    print(f"Model trained with {len(vectorizer.word_to_idx)} words\n")
    
    # Step 3: Demonstrate word similarity
    print("Step 3: Finding similar words...")
    
    test_words = ['computer', 'science', 'learning', 'data', 'algorithm']
    
    for word in test_words:
        similar = vectorizer.find_similar_words(word, top_k=5)
        if similar:
            print(f"\nWords similar to '{word}':")
            for item in similar:
                print(f"  {item['word']}: {item['similarity']:.3f}")
        else:
            print(f"\nWord '{word}' not found in vocabulary")
    
    # Step 4: Visualize word relationships
    print("\nStep 4: Creating word visualization...")
    
    # Words to visualize (mix of different topics)
    words_to_visualize = [
        # Computer science
        'computer', 'programming', 'algorithm', 'software', 'database',
        # Science
        'science', 'physics', 'chemistry', 'biology', 'mathematics',
        # AI/ML
        'learning', 'intelligence', 'data', 'machine', 'neural',
        # General
        'system', 'method', 'analysis', 'research', 'technology'
    ]
    
    # Filter to only include words that exist in our vocabulary
    valid_words = [word for word in words_to_visualize 
                  if word in vectorizer.word_to_idx]
    
    if len(valid_words) >= 2:
        print(f"Visualizing {len(valid_words)} words...")
        vectorizer.visualize_words(valid_words)
    else:
        print("Not enough valid words to visualize")
    
    # Step 5: Interactive word exploration
    print("\nStep 5: Interactive word exploration")
    print("Enter words to find similar words (or 'quit' to exit):")
    
    while True:
        user_input = input("\nEnter a word: ").strip().lower()
        
        if user_input == 'quit':
            break
        
        if user_input in vectorizer.word_to_idx:
            similar = vectorizer.find_similar_words(user_input, top_k=10)
            print(f"\nWords similar to '{user_input}':")
            for item in similar:
                print(f"  {item['word']}: {item['similarity']:.3f}")
        else:
            print(f"Word '{user_input}' not found in vocabulary")
            print("Available words include:", list(vectorizer.word_to_idx.keys())[:20], "...")

if __name__ == "__main__":
    main() 