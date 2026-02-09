import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

# Parameters
max_n = 10
min_k = 1
max_k = 50
num_samples = 500     # Reduced to avoid memory issues
batch_size = 8        # Smaller batch size for safe batching

# === 1. DATA GENERATION HELPERS ===

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

# === 2. CREATE TRAINING DATA ===

n = max_n
swap_list = all_possible_swaps(n)
vocab_size = len(swap_list)
swap_to_idx = {swap: idx for idx, swap in enumerate(swap_list)}

training_data = []
for _ in range(num_samples):
    k = random.randint(min_k, max_k)
    swap_seq = generate_random_swap_sequence(n, k)
    perm = apply_swaps(n, swap_seq)
    label = int(is_complete_cycle(perm))
    encoded_seq = [swap_to_idx[s] for s in swap_seq]
    training_data.append((encoded_seq, label))

# === 3. DEFINE DATASET AND DATALOADER ===

class PermutationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_seqs = pad_sequence(sequences, batch_first=False, padding_value=0)
    labels = torch.tensor(labels).float()
    return padded_seqs, lengths, labels

dataset = PermutationDataset(training_data)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# === 4. DEFINE GRU MODEL ===

class GRUEmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), enforce_sorted=False)
        _, h_n = self.gru(packed)
        h_final = h_n.squeeze(0)
        return self.sigmoid(self.fc(h_final))

# === 5. TRAINING SETUP ===

embedding_dim = 16
hidden_size = 64
model = GRUEmbeddingClassifier(vocab_size, embedding_dim, hidden_size, 1)

loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# === 6. TRAINING LOOP ===

losses = []

for padded_x, lengths, y_batch in loader:
    y_pred = model(padded_x, lengths)
    loss = loss_fn(y_pred.view(-1), y_batch)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    losses.append(loss.item())

# === 7. PLOT LOSS ===

plt.plot(losses)
plt.title("Loss per Batch (Mini-Batched, 1 Epoch, Variable k)")
plt.xlabel("Batch Index")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
