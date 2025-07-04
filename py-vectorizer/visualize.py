import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import chain

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# -----------------------------
# Step 1: Load corpus JSON and build vocab
# -----------------------------
with open('corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

tokenized_corpus = [sentence.split() for sentence in corpus]
vocab = sorted(set(chain.from_iterable(tokenized_corpus)))
word_to_idx = {word: i for i, word in enumerate(vocab)}

print(f"Loaded corpus with {len(corpus)} sentences and vocab size {len(vocab)}")

# -----------------------------
# Step 2: Load embeddings from CSV
# -----------------------------
w1 = np.load('word_vectors.npy')

print(f"Loaded embeddings matrix with shape {w1.shape}")

# -----------------------------
# Step 3: Cosine similarity matrix
# -----------------------------
words_to_visualize = [
    # People
    "man", "woman", "boy", "girl", "king", "queen", "uncle", "aunt",
    
    # Animals
    "dog", "cat", "lion", "tiger", "elephant", "wolf", "fox", "bear",
    
    # Food
    "apple", "banana", "orange", "grape", "fruit", "carrot", "vegetable", "bread",
    
    # Vehicles
    "car", "truck", "bus", "bike", "train", "airplane", "vehicle", "boat",
    
    # Places
    "city", "village", "town", "country", "state", "continent", "planet", "earth",
    
    # Emotions / States
    "happy", "sad", "angry", "excited", "tired", "bored", "nervous", "calm",
    
    # Tech
    "computer", "laptop", "keyboard", "mouse", "phone", "tablet", "internet", "software",
    
    # School / Learning
    "book", "notebook", "pencil", "teacher", "student", "class", "school", "university",
    
    # Abstract concepts
    "freedom", "justice", "truth", "honesty", "love", "hate", "peace", "war"
]
vectors = []
missing_words = []

for w in words_to_visualize:
    if w in word_to_idx:
        vectors.append(w1[word_to_idx[w]])
    else:
        missing_words.append(w)

if missing_words:
    print(f"Warning: these words not found in vocab and will be skipped: {missing_words}")

# Compute pairwise cosine similarities
print("\nCosine Similarities:")
for i in range(len(vectors)):
    for j in range(i, len(vectors)):
        sim = cosine_similarity(vectors[i], vectors[j])
        print(f"{words_to_visualize[i]} <-> {words_to_visualize[j]}: {sim:.4f}")

# -----------------------------
# Step 4: Visualization function
# -----------------------------
def visualize_embeddings(words):
    filtered_words = [w for w in words if w in word_to_idx]
    vecs = np.array([w1[word_to_idx[w]] for w in filtered_words])
    reduced = PCA(n_components=2).fit_transform(vecs)

    plt.figure(figsize=(8, 6))
    for i, word in enumerate(filtered_words):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, word, fontsize=12)
    plt.title("Word Embeddings Visualization (PCA Projection)")
    plt.grid(True)
    plt.show()

visualize_embeddings(words_to_visualize)