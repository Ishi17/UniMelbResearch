#trying the new input format - from final.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

#add bias = true
#calculate input and previous state
#chunk just makes it loop over the code for the number of hidden units provided
#batch size = how many inputs of training data is provided to the machine at once so that it trains 
#epochs = how many times the machine runs all of the samples (can provide it shuffled multiple times)
#decide training and testing split
#we use Sigmoid for the final classification of cycle
#backprop calculates the gradient and optimisers decide how to update the weights based on that
#use Adam instead of SGD - optimisers - method of updating weights
#explain Adam's optimisation
#one sample per batch = stokhastic gradient descent mode
#batch size is how many samples are run before updating
#change it to have 10 per batch, see whats better
#explain learning rate
#training and testing split
#Write about the binary classification imbalanced case - use pos_weight - giving more weight to some classification mistakes
#because of this, apply sigmoid at the end for the binary classification, but not throughout because its part of BCELOgits - happens internally
#BCEWithLogitsLoss = combines Sigmoid + BCELoss in one step.
#If you pass it a sigmoid output, you're applying sigmoid twice, which breaks the math.
#logit loss makes it overfit too much, much more error prone
#Instead oversample the uncommon case by duplicating the samples - limitations to this
#add padding to inputs so the batching can make them the same length - caused a lot of problems with the MLP trying to learn from it 
#have a larger numeber of epochs

# Parameters
max_n = 10         # Max size of permutation
min_k = 1 #Possible number of swaps per squence
max_k = 20
num_samples = 3000  # Total training samples

# Data generation helpers

def all_possible_swaps(n):
    """Return list of all (i, j) pairs with i < j for a permutation of size n"""
    return [(i, j) for i in range(1, n) for j in range(i + 1, n + 1)]

def generate_random_swap_sequence(n, k):
    """Generate a sequence of k random swaps from the full swap list"""
    swaps = all_possible_swaps(n)
    return [random.choice(swaps) for _ in range(k)]

def apply_swaps(n, swap_seq):
    """Apply swaps right-to-left to identity permutation"""
    perm = list(range(1, n + 1))
    for i, j in reversed(swap_seq):  # Right to left
        perm[i - 1], perm[j - 1] = perm[j - 1], perm[i - 1]
    return perm

def is_complete_cycle(perm):
    """Check if a permutation is a complete cycle"""
    visited = [False] * len(perm)
    i = 0
    for _ in range(len(perm)):
        if visited[i]:
            return False
        visited[i] = True
        i = perm[i] - 1
    return all(visited) and i == 0

#for training
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    sequences, labels = zip(*batch)

    # Convert each swap sequence to a FloatTensor of shape [seq_len, 2]
    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]

    # Pad sequences so they have the same length
    padded_seqs = pad_sequence(sequences, batch_first=True)  # shape: [batch, max_seq_len, 2]

    lengths = torch.tensor([len(seq) for seq in sequences])  # original lengths
    labels = torch.tensor(labels, dtype=torch.float32)       # shape: [batch]

    return padded_seqs, lengths, labels

# Create training data
target_per_class = 1500

n = max_n
swap_list = all_possible_swaps(n)
vocab_size = len(swap_list)  # total unique swaps (i.e. input vocab)

# Map swap (i, j) to unique index
swap_to_idx = {swap: idx for idx, swap in enumerate(swap_list)}

training_data = []

swap_list = all_possible_swaps(n)
swap_to_idx = {swap: idx for idx, swap in enumerate(swap_list)}

pos_samples = []
neg_samples = []

while len(pos_samples) < target_per_class or len(neg_samples) < target_per_class:
    k = random.randint(min_k, max_k)
    swap_seq = generate_random_swap_sequence(n, k)
    perm = apply_swaps(n, swap_seq)
    label = int(is_complete_cycle(perm))
    #encoded_seq = [swap_to_idx[s] for s in swap_seq]
    encoded_seq = swap_seq
    if label == 1 and len(pos_samples) < target_per_class:
        pos_samples.append((encoded_seq, label))
    elif label == 0 and len(neg_samples) < target_per_class:
        neg_samples.append((encoded_seq, label))

#print(pos_samples)
training_data = pos_samples + neg_samples
random.shuffle(training_data)

#undersampling majority case
# positive_examples = [sample for sample in training_data if sample[1] == 1]
# negative_examples = [sample for sample in training_data if sample[1] == 0]
# num_pos = len(positive_examples)
# num_neg = len(negative_examples)



# if num_pos < num_neg:
#     oversampled_positives = random.choices(positive_examples, k = int(0.8*num_neg))
#     training_data = oversampled_positives + negative_examples
# elif num_neg < num_pos:
#     oversampled_negatives = random.choices(negative_examples, k = int(0.8*num_pos))
#     training_data = positive_examples + oversampled_negatives
# else:
#     training_data = positive_examples + negative_examples


# Split training and test sets
split_ratio = 0.8
split_index = int(split_ratio * len(training_data))
train_set = training_data[:split_index]
test_set = training_data[split_index:]

test_labels = [label for _, label in test_set]
num_pos = sum(test_labels)
num_neg = len(test_labels) - num_pos

BATCH_SIZE = 8
EPOCHS = 4

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
print(f"Test Set — Positives: {num_pos}, Negatives: {num_neg}")

# Define GRU model with embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRUWithSwapPairs(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.fc_in = nn.Linear(2, embedding_dim)  # 🔁 embedding swap (i,j) pair
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        x = self.fc_in(x)  # [batch, seq_len, 2] → [batch, seq_len, embed_dim]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.sigmoid(self.fc_out(h_n[-1])).squeeze(1)


# Model setup

embedding_dim = 16
hidden_size = 64
NUM_LAYERS = 7
start_time = time.time()
model = GRUWithSwapPairs(embedding_dim, hidden_size, output_size=1, num_layers = NUM_LAYERS)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Training loop (SGD, 1 epoch)

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

end_time = time.time()
training_time = end_time - start_time
# Optional: visualize or save

# Plot training loss over time
plt.plot(losses)
plt.title("Loss per Training Sample (Adams, 1 Epoch, Variable k)")
plt.xlabel("Sample Index")
plt.ylabel("Loss")
plt.grid(True)
plt.show(block=False)
input("Press Enter to close...")
    
# Evaluate on test set
correct = 0
total = 0
train_accuracy = correct_train = total_train = 0
train_loss_sum = 0

with torch.no_grad():
    model.train()
    for seq, label in train_set:
        x_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, 2]
        y_tensor = torch.tensor([label], dtype=torch.float32)
        length_tensor = torch.tensor([len(seq)])  # shape: [1]
        y_pred = model(x_tensor, length_tensor)
        prediction = (y_pred.item() >= 0.5)
        correct_train += int(prediction == label)
        total_train += 1
        train_loss_sum += loss_fn(y_pred.view(-1), y_tensor).item()

test_loss_sum = 0
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():  # No gradient tracking needed
    for seq, label in test_set:
        x_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # shape: [1, seq_len, 2]
        length_tensor = torch.tensor([len(seq)])  # shape: [1]
        y_tensor = torch.tensor([label], dtype=torch.float32)
        y_pred = model(x_tensor, length_tensor)
        prediction = int(y_pred.item() >= 0.5)  # threshold at 0.5
        all_preds.append(prediction)
        all_labels.append(int(label))
        correct += int(prediction == label)
        total += 1
        test_loss_sum += loss_fn(y_pred.view(-1), y_tensor).item()

#Print Metrics
train_accuracy = correct_train / total_train
avg_train_loss = train_loss_sum / total_train
test_accuracy = correct / total
avg_test_loss = test_loss_sum / total
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
conf_matrix = confusion_matrix(all_labels, all_preds)


# Count model parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Sample predictions:", all_preds[:20])
print("Sample labels:     ", all_labels[:20])
from collections import Counter
print("Train set label counts:", Counter(label for _, label in train_set))
print("Test set label counts:", Counter(label for _, label in test_set))

# Print summary
print("\n=== MODEL EVALUATION SUMMARY ===")
print(f"GRU Layers:           {NUM_LAYERS}")
print(f"Train Accuracy:       {train_accuracy:.4f}")
print(f"Train Avg Loss:       {avg_train_loss:.4f}")
print(f"Test Accuracy:        {test_accuracy:.4f}")
print(f"Test Avg Loss:        {avg_test_loss:.4f}")
print(f"Train Samples:        {total_train}")
print(f"Test Samples:         {total}")
print(f"Model Parameters:     {num_params}")
print(f"\nTraining time: {training_time:.2f} seconds")
print(f"Precision:            {precision:.4f}")
print(f"Recall:               {recall:.4f}")
print(f"F1 Score:             {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
