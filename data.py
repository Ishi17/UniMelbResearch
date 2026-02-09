import torch
import torch.nn as nn
import torch.optim as optim
import random

# Parameters
max_n = 10         # Max size of permutation
min_k = 1 #Possible number of swaps per squence
max_k = 50
num_samples = 1000  # Total training samples

# === 1. DATA GENERATION HELPERS ===

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

# === 2. CREATE TRAINING DATA ===

n = max_n
swap_list = all_possible_swaps(n)
vocab_size = len(swap_list)  # total unique swaps (i.e. input vocab)

# Map swap (i, j) to unique index
swap_to_idx = {swap: idx for idx, swap in enumerate(swap_list)}

training_data = []
for _ in range(num_samples):
    k = random.randint(min_k, max_k)
    swap_seq = generate_random_swap_sequence(n, k)
    perm = apply_swaps(n, swap_seq)
    label = int(is_complete_cycle(perm))  # 1 = complete cycle, 0 = not
    encoded_seq = [swap_to_idx[s] for s in swap_seq]
    training_data.append((encoded_seq, label))
print(training_data[1])