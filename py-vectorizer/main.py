# import json
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from collections import defaultdict
# from tqdm import tqdm

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # -----------------------------
# # Load and process corpus
# # -----------------------------
# with open("corpus.json", "r", encoding="utf-8") as f:
#     corpus_json = json.load(f)

# # Flatten corpus into list of sentences of tokens
# corpus = [line.split() for line in corpus_json]
# print(f"Loaded {len(corpus)} sentences")

# # Build vocabulary
# word_freq = defaultdict(int)
# for sentence in corpus:
#     for word in sentence:
#         word_freq[word] += 1

# vocab = sorted(word_freq.keys())
# word2idx = {word: idx for idx, word in enumerate(vocab)}
# idx2word = {idx: word for word, idx in word2idx.items()}
# vocab_size = len(vocab)
# print(f"Vocabulary size: {vocab_size}")

# # -----------------------------
# # Generate skip-gram pairs
# # -----------------------------
# window_size = 2
# pairs = []
# for sentence in corpus:
#     for i, center_word in enumerate(sentence):
#         center_idx = word2idx[center_word]
#         for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
#             if i != j:
#                 context_idx = word2idx[sentence[j]]
#                 pairs.append((center_idx, context_idx))

# print(f"Generated {len(pairs)} training pairs")

# # -----------------------------
# # Word2Vec Skip-gram Model
# # -----------------------------
# class SkipGramModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim):
#         super(SkipGramModel, self).__init__()
#         self.embeddings = nn.Embedding(vocab_size, embedding_dim)
#         self.output = nn.Linear(embedding_dim, vocab_size)

#     def forward(self, center_words):
#         embeds = self.embeddings(center_words)
#         out = self.output(embeds)
#         return out

# # -----------------------------
# # Training setup
# # -----------------------------
# embedding_dim = 300
# model = SkipGramModel(vocab_size, embedding_dim).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.025)

# epochs = 10
# batch_size = 512

# def batchify(data, batch_size):
#     for i in range(0, len(data), batch_size):
#         yield data[i:i + batch_size]

# # -----------------------------
# # Training loop
# # -----------------------------
# #print length of batches
# counter = 0;
# print(f"Length of batches: {len(list(batchify(pairs, batch_size)))}")
# for epoch in range(epochs):
#     total_loss = 0
#     np.random.shuffle(pairs)
#     for batch in tqdm(batchify(pairs, batch_size), desc=f"Epoch {epoch+1}"):
#         counter += 1;
#         if counter % 10000 == 0:
#             #save to numpy file
#             embeddings = model.embeddings.weight.data.cpu().numpy()
#             np.save("w1.npy", embeddings)
#             torch.save(model.state_dict(), f"model_weights.pth")
#             torch.save(optimizer.state_dict(), f"optimizer_state.pth")
#         center_batch = torch.tensor([c for c, _ in batch], dtype=torch.long).to(device)
#         context_batch = torch.tensor([ctx for _, ctx in batch], dtype=torch.long).to(device)

#         optimizer.zero_grad()
#         output = model(center_batch)  # shape: (batch_size, vocab_size)
#         loss = criterion(output, context_batch)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#     print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
#     np.save("w1.npy", embeddings)
#     torch.save(model.state_dict(), f"model_weights.pth")
#     torch.save(optimizer.state_dict(), f"optimizer_state.pth")

# # -----------------------------
# # Save learned embeddings
# # -----------------------------
# embeddings = model.embeddings.weight.data.cpu().numpy()
# #save weights to numpy
# np.save("w1.npy", embeddings)
# print("Saved embeddings to w1.npy")
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import math

# -----------------------------
# Parameters
# -----------------------------
embedding_dim = 300
window_size = 5
epochs = 5
batch_size = 512
num_neg_samples = 5
subsample_thresh = 1e-5

# -----------------------------
# Load and process corpus
# -----------------------------
with open("corpus.json", "r", encoding="utf-8") as f:
    corpus_json = json.load(f)

corpus = [line.split() for line in corpus_json]
print(f"Loaded {len(corpus)} sentences")

# -----------------------------
# Build vocabulary
# -----------------------------
word_freq = defaultdict(int)
total_tokens = 0
for sentence in corpus:
    for word in sentence:
        word_freq[word] += 1
        total_tokens += 1

vocab = sorted(word_freq.keys())
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)
word_counts = np.array([word_freq[w] for w in vocab], dtype=np.float32)

# Compute word frequencies and subsampling probabilities
word_freqs = word_counts / total_tokens
subsampling_probs = {word: 1 - math.sqrt(subsample_thresh / freq) for word, freq in zip(vocab, word_freqs)}

print(f"Vocabulary size: {vocab_size}")

# -----------------------------
# Create training pairs with negative samples
# -----------------------------
def generate_training_data():
    pairs = []
    for sentence in corpus:
        sentence = [w for w in sentence if random.random() > subsampling_probs.get(w, 0)]
        for i, center_word in enumerate(sentence):
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if i != j:
                    pairs.append((word2idx[center_word], word2idx[sentence[j]]))
    return pairs

unigram_dist = word_counts**0.75
unigram_dist /= unigram_dist.sum()  # Normalize
unigram_dist = np.clip(unigram_dist, 0, 1)  # Ensure valid range
unigram_dist /= unigram_dist.sum()  # Re-normalize after clipping

def get_negative_samples(batch_size, num_negatives):
    return np.random.choice(len(vocab), size=(batch_size, num_negatives), p=unigram_dist)

# -----------------------------
# Model
# -----------------------------
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_words, context_words, negative_words):
        center_embeds = self.input_embeddings(center_words)             # (B, D)
        context_embeds = self.output_embeddings(context_words)          # (B, D)
        neg_embeds = self.output_embeddings(negative_words)             # (B, N, D)

        # Positive score
        pos_score = torch.sum(center_embeds * context_embeds, dim=1)    # (B,)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-9)

        # Negative score
        neg_score = torch.bmm(neg_embeds.neg(), center_embeds.unsqueeze(2)).squeeze()  # (B, N)
        neg_loss = torch.sum(torch.log(torch.sigmoid(neg_score) + 1e-9), dim=1)         # (B,)

        return -torch.mean(pos_loss + neg_loss)

# -----------------------------
# Training
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkipGramNegSampling(vocab_size, embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.003)

print("Generating training pairs...")
pairs = generate_training_data()
print(f"Generated {len(pairs)} training pairs")

def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

counter = 0;
for epoch in range(epochs):
    random.shuffle(pairs)
    total_loss = 0
    for batch in tqdm(batchify(pairs, batch_size), desc=f"Epoch {epoch+1}"):
        counter += 1;
        if counter % 10000 == 0:
            embeddings = model.input_embeddings.weight.data.cpu().numpy()
            np.save("word_vectors.npy", embeddings)
        center_batch = torch.tensor([c for c, _ in batch], dtype=torch.long).to(device)
        context_batch = torch.tensor([ctx for _, ctx in batch], dtype=torch.long).to(device)
        negative_batch = torch.tensor(get_negative_samples(len(batch), num_neg_samples), dtype=torch.long).to(device)

        optimizer.zero_grad()
        loss = model(center_batch, context_batch, negative_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# -----------------------------
# Save embeddings
# -----------------------------
embeddings = model.input_embeddings.weight.data.cpu().numpy()
np.save("word_vectors.npy", embeddings)
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f)

print("Saved word vectors and vocab.")
