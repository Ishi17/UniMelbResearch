import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

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
#chain hidden size to 64
#add batch sizes
#batches need to colate the sequences so that they are all of the same length, so we pad it because of that

# Parameters
max_n = 10         # Max size of permutation
min_k = 1 #Possible number of swaps per squence
max_k = 20
num_samples = 1000  # Total training samples

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

# Create training data
def custom_collate(batch):
    """
    batch: list of (sequence, label)
    """
    sequences, labels = zip(*batch)
    sequence_tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    lengths = torch.tensor([len(seq) for seq in sequence_tensors])
    padded_sequences = pad_sequence(sequence_tensors, batch_first=True)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    return padded_sequences, lengths, label_tensor
target_per_class = 500

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
    encoded_seq = [swap_to_idx[s] for s in swap_seq]

    if label == 1 and len(pos_samples) < target_per_class:
        pos_samples.append((encoded_seq, label))
    elif label == 0 and len(neg_samples) < target_per_class:
        neg_samples.append((encoded_seq, label))

print(pos_samples)
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
print(f"Test Set — Positives: {num_pos}, Negatives: {num_neg}")

# Define GRU model with embedding
class GRUEmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, bias=True, num_layers = num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    # def forward(self, x):
    #     embedded = self.embedding(x)        # shape: [batch, seq_len, embed_dim]
    #     out, h_n = self.gru(embedded)       # h_n: [num_layers, batch, hidden_size]
    #     h_final = h_n[-1]                   # use the hidden state from the last GRU layer
    #     return self.sigmoid(self.fc(h_final))
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)  # h_n: [num_layers, batch, hidden]
        h_final = h_n[-1]          # last layer
        return self.sigmoid(self.fc(h_final))


# Model setup

embedding_dim = 16
hidden_size = 64
NUM_LAYERS = 2
model = GRUEmbeddingClassifier(vocab_size, embedding_dim, hidden_size, output_size=1, num_layers = NUM_LAYERS)
BATCH_SIZE = 16

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop (SGD, 1 epoch)

losses = []
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)

# for batch in train_loader:
#     seqs, labels = zip(*batch)

#     # Convert to tensors
#     seqs = [torch.tensor(seq, dtype=torch.long) for seq in seqs]
#     lengths = torch.tensor([len(seq) for seq in seqs])

#     # Pad and sort
#     padded = pad_sequence(seqs, batch_first=True)
#     labels_tensor = torch.tensor(labels, dtype=torch.float32)

#     # Forward pass
#     outputs = model(padded, lengths)
#     loss = loss_fn(outputs.view(-1), labels_tensor)

#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     losses.append(loss.item())

start_time = time.time()
NUM_EPOCHS = 5

for epoch in range(NUM_EPOCHS):
    model.train()  # Make sure model is in training mode
    epoch_loss = 0

    for batch in train_loader:
        padded_seqs, lengths, labels = batch              # From custom_collate
        lengths = lengths.cpu()                           # Ensure lengths are on CPU for packing

        # Forward pass
        embedded = model.embedding(padded_seqs)          
        print("padded_seqs shape:", padded_seqs.shape)
        print("lengths:", lengths)
# [batch_size, seq_len, embed_dim]
        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, h_n = model.gru(packed_input)      # h_n: [num_layers, batch_size, hidden_size]
        h_final = h_n[-1]                                  # Use last GRU layer's hidden state

        logits = model.fc(h_final)                         # [batch_size, 1]
        probs = model.sigmoid(logits).squeeze()            # [batch_size] after squeeze

        # Compute loss
        loss = loss_fn(probs, labels)
        epoch_loss += loss.item()

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

            #losses.append(loss.item())
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Avg Loss: {avg_loss:.4f}")
        losses.append(avg_loss)
# for seq, label in train_set:
#     seq_len = len(seq)
#     x_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0)  # shape: [1, seq_len]
#     y_tensor = torch.tensor([label], dtype=torch.float32)
#     y_pred = model(x_tensor)
#     loss = loss_fn(y_pred.view(-1), y_tensor)

#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()

#     losses.append(loss.item())
    

end_time = time.time()
training_time = end_time - start_time
# Optional: visualize or save

# Plot training loss over time
plt.plot(losses)
plt.title("Loss per Training Sample (Adams, 1 , Variable k)")
plt.xlabel("Sample Index")
plt.ylabel("Loss")
plt.grid(True)
plt.show(block=False)
input("Press Enter to close...")

# Print some sample results
for i in range(5):
    swap_ids, label = training_data[i]
    swap_seq = [swap_list[idx] for idx in swap_ids[:3]]
    print(f"Swaps (first 3): {swap_seq}, Label: {label}, Loss: {round(losses[i], 4)}")
    
# Evaluate on test set
correct = 0
total = 0
train_accuracy = correct_train = total_train = 0
train_loss_sum = 0

# with torch.no_grad():
#     for seq, label in train_set:
#         x_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
#         y_tensor = torch.tensor([label], dtype=torch.float32)
#         y_pred = model(x_tensor)
#         prediction = (y_pred.item() >= 0.5)
#         correct_train += int(prediction == label)
#         total_train += 1
#         train_loss_sum += loss_fn(y_pred.view(-1), y_tensor).item()

with torch.no_grad():
    for seq, label in train_set:
        x_tensor = torch.tensor(seq, dtype=torch.long)
        length_tensor = torch.tensor([len(x_tensor)])

        padded = nn.utils.rnn.pad_sequence([x_tensor], batch_first=True)  # shape: [1, seq_len]
        y_tensor = torch.tensor([label], dtype=torch.float32)

        y_pred = model(padded, length_tensor)  # pass both padded input and length
        prediction = int(y_pred.item() >= 0.5)
test_loss_sum = 0
all_preds = []
all_labels = []
model.eval()

with torch.no_grad():  # No gradient tracking needed
    for seq, label in test_set:
        x_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0)  # shape: [1, seq_len]
        lengths_tensor = torch.tensor([len(seq)])  # Add this line
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
