# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from scipy.optimize import minimize_scalar
from numpy.polynomial import Polynomial

# Parameters for the model and data generation
max_n = 10
min_k = 1
max_k = 20
target_per_class = 1500
BATCH_SIZE = 8
EPOCHS = 4
embedding_dim = 16
hidden_size = 64
NUM_LAYERS = 2

# Swap helpers
def all_possible_swaps(n):
    return [(i, j) for i in range(1, n) for j in range(i + 1, n + 1)]

def generate_random_swap_sequence(n, k):
    swaps = all_possible_swaps(n)
    return [random.choice(swaps) for _ in range(k)]

def apply_swaps(n, swap_seq):
    perm = list(range(1, n + 1))
    for i, j in reversed(swap_seq):
        perm[i - 1], perm[j - 1] = perm[j - 1], perm[i - 1]
    return perm

def is_complete_cycle(perm):
    visited = [False] * len(perm)
    i = 0
    for _ in range(len(perm)):
        if visited[i]:
            return False
        visited[i] = True
        i = perm[i] - 1
    return all(visited) and i == 0

# Prepare each batch with padding and tensor conversion
def collate_batch(batch):
    sequences, labels = zip(*batch)
    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    padded_seqs = pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor([len(seq) for seq in sequences])
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded_seqs, lengths, labels

# Generate a balanced dataset of complete and non-complete cycles
pos_samples = []
neg_samples = []

while len(pos_samples) < target_per_class or len(neg_samples) < target_per_class:
    n = random.randint(2, max_n)  # Random n each time
    k = random.randint(min_k, max_k)
    swap_seq = generate_random_swap_sequence(n, k)
    perm = apply_swaps(n, swap_seq)
    label = int(is_complete_cycle(perm))

    if label == 1 and len(pos_samples) < target_per_class:
        pos_samples.append((swap_seq, label))
    elif label == 0 and len(neg_samples) < target_per_class:
        neg_samples.append((swap_seq, label))

# Shuffle and split the data
training_data = pos_samples + neg_samples
random.shuffle(training_data)
split_idx = int(0.8 * len(training_data))
train_set = training_data[:split_idx]
test_set = training_data[split_idx:]
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

# GRU model to classify swap sequences
class GRUWithSwapPairs(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.fc_in = nn.Linear(2, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        x = self.fc_in(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.sigmoid(self.fc_out(h_n[-1])).squeeze(1)

# Train and evaluate model for a given learning rate
def train_and_eval_model(lr):
    model = GRUWithSwapPairs(embedding_dim, hidden_size, output_size=1, num_layers=NUM_LAYERS)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(EPOCHS):
        model.train()
        for padded_seqs, lengths, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(padded_seqs, lengths)
            loss = loss_fn(y_pred, labels)
            loss.backward()
            optimizer.step()

    # Evaluate model accuracy on test set
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for seq, label in test_set:
            x_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            length_tensor = torch.tensor([len(seq)])
            y_pred = model(x_tensor, length_tensor)
            prediction = int(y_pred.item() >= 0.5)
            correct += int(prediction == label)
            total += 1

    return correct / total

# Test results from different learning rates
learning_rates = np.array([
    0.00001, 0.00002, 0.00005, 0.00007, 0.0001,
    0.0002, 0.0005, 0.001, 0.002, 0.005,
    0.01, 0.009, 0.007, 0.003, 0.004
])
test_accuracies = np.array([
    0.64, 0.65, 0.64, 0.66, 0.68,
    0.65, 0.66, 0.67, 0.67, 0.67,
    0.65, 0.65, 0.65, 0.67, 0.68
])

# Fit a polynomial to the data to estimate optimal learning rate
p = Polynomial.fit(learning_rates, test_accuracies, deg=3)
x_fit = np.linspace(learning_rates.min(), learning_rates.max(), 200)
y_fit = p(x_fit)

# Maximize the polynomial to find the best learning rate
neg_p = lambda x: -p(x)
result = minimize_scalar(neg_p, bounds=(learning_rates.min(), learning_rates.max()), method='bounded')
optimal_lr = result.x
optimal_acc = -result.fun

# Visualize results and optimal learning rate
plt.figure(figsize=(8, 5))
plt.plot(learning_rates, test_accuracies, 'o', label='Actual Results')
plt.plot(x_fit, y_fit, '-', label='Polynomial Fit')
plt.axvline(optimal_lr, color='green', linestyle='--', label=f'Optimal LR ≈ {optimal_lr:.4f}')
plt.title("Test Accuracy vs Learning Rate")
plt.xlabel("Learning Rate")
plt.ylabel("Test Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the best learning rate and its estimated accuracy
print(f"Optimal learning rate ≈ {optimal_lr:.5f}, with estimated accuracy ≈ {optimal_acc:.4f}")
