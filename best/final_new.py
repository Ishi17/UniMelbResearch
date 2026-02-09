import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader

# Functions for creating and applying swaps

def all_possible_swaps(n):
    # Just gets all valid (i, j) swaps for a permutation of size n
    return [(i, j) for i in range(1, n) for j in range(i + 1, n + 1)]

def generate_random_swap_sequence(n, k):
    # Randomly pick k swaps from the list of all possible swaps
    swaps = all_possible_swaps(n)
    return [random.choice(swaps) for _ in range(k)]

def apply_swaps(n, swap_seq):
    # Start with [1, 2, ..., n] and apply swaps in reverse order
    perm = list(range(1, n + 1))
    for i, j in reversed(swap_seq):
        perm[i - 1], perm[j - 1] = perm[j - 1], perm[i - 1]
    return perm

def is_complete_cycle(perm):
    # Check if this permutation loops through every number exactly once
    visited = [False] * len(perm)
    i = 0
    for _ in range(len(perm)):
        if visited[i]:
            return False
        visited[i] = True
        i = perm[i] - 1
    return all(visited) and i == 0

def collate_batch(batch):
    # Pads all the sequences in a batch so they're the same length
    sequences, labels = zip(*batch)
    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    padded_seqs = pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor([len(seq) for seq in sequences])
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded_seqs, lengths, labels

# Now we generate the dataset with random n and k

max_n = 10
min_k = 1
max_k = 20
target_per_class = 1500

pos_samples, neg_samples = [], []

# Keep generating until we have 1500 positive and 1500 negative samples
while len(pos_samples) < target_per_class or len(neg_samples) < target_per_class:
    n = random.randint(2, max_n)
    k = random.randint(min_k, max_k)
    swap_seq = generate_random_swap_sequence(n, k)
    perm = apply_swaps(n, swap_seq)
    label = int(is_complete_cycle(perm))

    if label == 1 and len(pos_samples) < target_per_class:
        pos_samples.append((swap_seq, label))
    elif label == 0 and len(neg_samples) < target_per_class:
        neg_samples.append((swap_seq, label))

# Combine and shuffle everything before training
training_data = pos_samples + neg_samples
random.shuffle(training_data)

# Split into training and test sets
split_ratio = 0.8
split_index = int(split_ratio * len(training_data))
train_set = training_data[:split_index]
test_set = training_data[split_index:]

test_labels = [label for _, label in test_set]
num_pos = sum(test_labels)
num_neg = len(test_labels) - num_pos

BATCH_SIZE = 8
EPOCHS = 4

# Create DataLoaders with padding handled automatically
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

print(f"Test Set — Positives: {num_pos}, Negatives: {num_neg}")

# Define the GRU model that takes swap pairs as input

class GRUWithSwapPairs(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.fc_in = nn.Linear(2, embedding_dim)  # Convert (i, j) into embedding space
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        x = self.fc_in(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.sigmoid(self.fc_out(h_n[-1])).squeeze(1)

# Set model hyperparameters and initialise everything

embedding_dim = 16
hidden_size = 64
NUM_LAYERS = 2

model = GRUWithSwapPairs(embedding_dim, hidden_size, output_size=1, num_layers=NUM_LAYERS)
loss_fn = nn.BCELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0028)
optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)

# Train the model

losses = []
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for padded_seqs, lengths, labels in train_loader:
        optimizer.zero_grad()
        y_pred = model(padded_seqs, lengths)
        loss = loss_fn(y_pred, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_epoch_loss = epoch_loss / len(train_loader)
    losses.append(avg_epoch_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_epoch_loss:.4f}")

training_time = time.time() - start_time

# Plot the training loss

plt.plot(losses)
plt.title("Loss over Epochs (Adam)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show(block=False)

# Evaluate the model on any dataset (train/test)

def evaluate(model, dataset, loss_fn):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for seq, label in dataset:
            x_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            length_tensor = torch.tensor([len(seq)])
            y_tensor = torch.tensor([label], dtype=torch.float32)
            y_pred = model(x_tensor, length_tensor)
            prediction = int(y_pred.item() >= 0.5)
            all_preds.append(prediction)
            all_labels.append(int(label))
            correct += int(prediction == label)
            total_loss += loss_fn(y_pred.view(-1), y_tensor).item()
    return correct / len(dataset), total_loss / len(dataset), all_preds, all_labels

# Get accuracy, loss and predictions

train_accuracy, avg_train_loss, _, _ = evaluate(model, train_set, loss_fn)
test_accuracy, avg_test_loss, all_preds, all_labels = evaluate(model, test_set, loss_fn)

precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
conf_matrix = confusion_matrix(all_labels, all_preds)

# Print out the final results

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Sample predictions:", all_preds[:20])
print("Sample labels:     ", all_labels[:20])
from collections import Counter
print("Train set label counts:", Counter(label for _, label in train_set))
print("Test set label counts:", Counter(label for _, label in test_set))

print("\nMODEL EVALUATION SUMMARY")
print(f"Train Accuracy:       {train_accuracy:.4f}")
print(f"Test Accuracy:        {test_accuracy:.4f}")
print(f"F1 Score:             {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
