# Finding the optimal range for the learning rate

import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import copy
import math

# Set parameters for permutation size and number of samples
max_n = 10         # Max size of permutation
min_k = 1          # Minimum number of swaps per sequence
max_k = 20         # Maximum number of swaps per sequence
num_samples = 3000 # Total training samples to generate

# Data generation helpers

# Get all possible (i, j) swap pairs where i < j
def all_possible_swaps(n):
    return [(i, j) for i in range(1, n) for j in range(i + 1, n + 1)]

# Generate a random sequence of k swaps
def generate_random_swap_sequence(n, k):
    swaps = all_possible_swaps(n)
    return [random.choice(swaps) for _ in range(k)]

# Apply the sequence of swaps to the identity permutation
def apply_swaps(n, swap_seq):
    perm = list(range(1, n + 1))
    for i, j in reversed(swap_seq):  # Apply swaps from right to left
        perm[i - 1], perm[j - 1] = perm[j - 1], perm[i - 1]
    return perm

# Check if a permutation is a complete cycle
def is_complete_cycle(perm):
    visited = [False] * len(perm)
    i = 0
    for _ in range(len(perm)):
        if visited[i]:
            return False
        visited[i] = True
        i = perm[i] - 1
    return all(visited) and i == 0

# Collate function for training batches

# Pad swap sequences and prepare batch tensors
def collate_batch(batch):
    sequences, labels = zip(*batch)

    # Convert each swap sequence to a FloatTensor
    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]

    # Pad sequences to match the longest in the batch
    padded_seqs = pad_sequence(sequences, batch_first=True)

    lengths = torch.tensor([len(seq) for seq in sequences])  # Store original lengths
    labels = torch.tensor(labels, dtype=torch.float32)       # Convert labels to tensor

    return padded_seqs, lengths, labels

# Create training data

# Create balanced dataset with equal positive and negative samples
target_per_class = 1500

pos_samples = []
neg_samples = []

# Generate random (swap sequence, label) pairs until balanced
while len(pos_samples) < target_per_class or len(neg_samples) < target_per_class:
    n = random.randint(2, max_n)  # Random n between 2 and max_n
    k = random.randint(min_k, max_k)  # Random number of swaps
    swap_seq = generate_random_swap_sequence(n, k)
    perm = apply_swaps(n, swap_seq)
    label = int(is_complete_cycle(perm))  # 1 if complete cycle, 0 otherwise
    encoded_seq = swap_seq  # Store raw (i, j) swap pairs

    if label == 1 and len(pos_samples) < target_per_class:
        pos_samples.append((encoded_seq, label))
    elif label == 0 and len(neg_samples) < target_per_class:
        neg_samples.append((encoded_seq, label))

# Combine and shuffle dataset
training_data = pos_samples + neg_samples
random.shuffle(training_data)

# Learning rate finder function

# Find a good learning rate using exponential increase
def run_lr_finder(model, train_loader, loss_fn, optimizer_class, start_lr=1e-5, end_lr=10, beta=0.98):
    model_copy = copy.deepcopy(model)
    model_copy.train()

    num = len(train_loader) - 1
    mult = (end_lr / start_lr) ** (1 / num)
    lr = start_lr
    optimizer = optimizer_class(model_copy.parameters(), lr=lr)

    avg_loss = 0.0
    best_loss = float('inf')
    batch_num = 0
    log_lrs, losses = [], []

    for inputs, lengths, labels in train_loader:
        batch_num += 1
        optimizer.param_groups[0]['lr'] = lr

        optimizer.zero_grad()
        y_pred = model_copy(inputs, lengths)
        loss = loss_fn(y_pred, labels)

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)

        # Stop if loss explodes
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break

        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        loss.backward()
        optimizer.step()
        lr *= mult

    # Plot learning rate vs loss curve
    import matplotlib.pyplot as plt
    plt.plot(log_lrs, losses)
    plt.xlabel("Log10(Learning Rate)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.grid(True)
    plt.show()

    return log_lrs, losses

# Split training and testing sets

# Split into 80% train, 20% test
split_ratio = 0.8
split_index = int(split_ratio * len(training_data))
train_set = training_data[:split_index]
test_set = training_data[split_index:]

# Count number of positive and negative samples in test set
test_labels = [label for _, label in test_set]
num_pos = sum(test_labels)
num_neg = len(test_labels) - num_pos

# Define batch size and epochs
BATCH_SIZE = 8
EPOCHS = 4

# Create DataLoaders for training and testing
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
print(f"Test Set — Positives: {num_pos}, Negatives: {num_neg}")

# Define GRU model with embedding

# Define GRU-based binary classifier that takes padded swap sequences
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRUWithSwapPairs(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.fc_in = nn.Linear(2, embedding_dim)  # Map (i, j) pair to embedding
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # For binary classification output

    def forward(self, x, lengths):
        x = self.fc_in(x)  # Embed input swap pairs
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)  # Only final hidden state is used
        return self.sigmoid(self.fc_out(h_n[-1])).squeeze(1)

# Model setup

# Set architecture parameters
embedding_dim = 16
hidden_size = 64
NUM_LAYERS = 2

# Initialize model
start_time = time.time()
model = GRUWithSwapPairs(embedding_dim, hidden_size, output_size=1, num_layers = NUM_LAYERS)

# Use binary cross-entropy loss
loss_fn = nn.BCELoss()

# Run learning rate finder using Adam optimizer
#run_lr_finder(model, train_loader, loss_fn, torch.optim.SGD, start_lr=1e-5, end_lr=1)
run_lr_finder(model, train_loader, loss_fn, torch.optim.Adam, start_lr=1e-5, end_lr=1)