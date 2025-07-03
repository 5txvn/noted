import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import chain

# ----------------------------
# STEP 1: Sample Corpus
# ---------------------------
with open('./js/wikipedia_corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)
print(len(corpus))

# ----------------------------
# STEP 2: Preprocessing (lowercase, tokenize)
# ----------------------------
def preprocess(sentences):
    return [sentence.lower().split() for sentence in sentences]

tokenized_corpus = preprocess(corpus)

# ----------------------------
# STEP 3: Vocabulary Building
# ----------------------------
vocab = sorted(set(chain.from_iterable(tokenized_corpus)))
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}
vocab_size = len(vocab)

# ----------------------------
# STEP 4: Generate Skip-Gram Pairs
# ----------------------------
def generate_skipgram_pairs(sentences, window_size=2):
    pairs = []
    for sentence in sentences:
        for i, center in enumerate(sentence):
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if i != j:
                    pairs.append((center, sentence[j]))
    return pairs

training_pairs = generate_skipgram_pairs(tokenized_corpus)
print(len(training_pairs))

# ----------------------------
# STEP 5: Initialize Embeddings
# ----------------------------
embedding_dim = 50  # you can raise this to 50+ for real training
W1 = np.random.randn(vocab_size, embedding_dim) * 0.01
W2 = np.random.randn(embedding_dim, vocab_size) * 0.01

# ----------------------------
# STEP 6: Softmax Function
# ----------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ----------------------------
# STEP 7: Training Skip-Gram Model
# ----------------------------
def train_skipgram(pairs, epochs=5, lr=0.05):
    global W1, W2
    for epoch in range(epochs):
        total_loss = 0
        i = 0
        for center, context in pairs:
            i += 1
            if i % 1000 == 0: print(f"Iteration: {i}")
            center_idx = word_to_idx[center]
            context_idx = word_to_idx[context]

            x = np.zeros(vocab_size)
            x[center_idx] = 1  # one-hot

            # Forward pass
            h = W1.T @ x              # hidden layer (embedding of center word)
            u = W2.T @ h              # output scores
            y_pred = softmax(u)       # softmax probabilities

            # True label
            y_true = np.zeros(vocab_size)
            y_true[context_idx] = 1

            # Loss (optional to store)
            loss = -np.log(y_pred[context_idx] + 1e-9)
            total_loss += loss

            # Backpropagation
            error = y_pred - y_true   # shape: (vocab_size,)
            dW2 = np.outer(h, error)  # (embedding_dim, vocab_size)
            dW1 = np.outer(x, W2 @ error)  # (vocab_size, embedding_dim)

            # Gradient descent update
            W1 -= lr * dW1
            W2 -= lr * dW2

        # Print loss
        i = 0
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

train_skipgram(training_pairs)

# ----------------------------
# STEP 8: Get Word Vector
# ----------------------------
def get_vector(word):
    return W1[word_to_idx[word]]

np.save("/content/w1.npy", W1)

# ----------------------------
# STEP 9: Visualize Vectors with PCA
# ----------------------------
def visualize_embeddings(words_to_plot):
    vectors = np.array([get_vector(word) for word in words_to_plot])
    reduced = PCA(n_components=2).fit_transform(vectors)

    plt.figure(figsize=(8, 6))
    for i, word in enumerate(words_to_plot):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, word, fontsize=12)
    plt.title("Word Embeddings Visualized (PCA Projection)")
    plt.grid(True)
    plt.show()

# ----------------------------
# STEP 10: Run Visualization
# ----------------------------
words = ["artificial", "intelligence", "machine", "language"]
visualize_embeddings(words)
