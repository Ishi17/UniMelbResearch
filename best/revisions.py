import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score
)

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader


#1. REPRODUCIBILITY

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# 2. SWAP / PERMUTATION FUNCTIONS

def all_possible_swaps(n):
    return [(i, j) for i in range(1, n) for j in range(i + 1, n + 1)]


def generate_random_swap_sequence(n, k):
    swaps = all_possible_swaps(n)
    return [random.choice(swaps) for _ in range(k)]


def apply_swaps(n, swap_seq):
    perm = list(range(1, n + 1))

    # Apply swaps right-to-left
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


# 3. DATA GENERATION

def generate_one_sample(max_n=10, min_k=1, max_k=20):
    n = random.randint(2, max_n)
    k = random.randint(min_k, max_k)

    # Written swap sequence: S = [s1, s2, ..., sk]
    swap_seq = generate_random_swap_sequence(n, k)

    # Labels are generated using standard right-to-left composition.
    # Therefore, sk is applied first.
    perm = apply_swaps(n, swap_seq)
    label = int(is_complete_cycle(perm))

    # The GRU input is reversed so it reads swaps in the same order
    # used by the labelling algorithm.
    model_input_seq = list(reversed(swap_seq))

    return model_input_seq, label, n, k

def generate_natural_dataset(num_samples, max_n=10, min_k=1, max_k=20):
    """
    Generates data using the natural distribution implied by random n and k.
    This is not forced to be 50/50.
    """
    data = []
    metadata = []

    for _ in range(num_samples):
        swap_seq, label, n, k = generate_one_sample(max_n, min_k, max_k)
        data.append((swap_seq, label))
        metadata.append({"n": n, "k": k, "label": label})

    counts = Counter(label for _, label in data)

    print("\nNATURAL DATASET CLASS DISTRIBUTION")
    print("----------------------------------")
    print(f"Total samples: {num_samples}")
    print(f"Negative / not complete cycle: {counts[0]} ({counts[0] / num_samples:.4f})")
    print(f"Positive / complete cycle:     {counts[1]} ({counts[1] / num_samples:.4f})")

    return data, pd.DataFrame(metadata)


def generate_balanced_dataset(target_per_class, max_n=10, min_k=1, max_k=20):
    """
    Generates an artificial 50/50 balanced dataset.
    This is useful as a controlled setting, but it does not represent the natural distribution.
    """
    pos_samples = []
    neg_samples = []
    attempts = 0

    while len(pos_samples) < target_per_class or len(neg_samples) < target_per_class:
        swap_seq, label, n, k = generate_one_sample(max_n, min_k, max_k)
        attempts += 1

        if label == 1 and len(pos_samples) < target_per_class:
            pos_samples.append((swap_seq, label))

        elif label == 0 and len(neg_samples) < target_per_class:
            neg_samples.append((swap_seq, label))

    data = pos_samples + neg_samples
    random.shuffle(data)

    counts = Counter(label for _, label in data)

    print("\nBALANCED DATASET CLASS DISTRIBUTION")
    print("-----------------------------------")
    print(f"Total samples: {len(data)}")
    print(f"Negative / not complete cycle: {counts[0]} ({counts[0] / len(data):.4f})")
    print(f"Positive / complete cycle:     {counts[1]} ({counts[1] / len(data):.4f})")
    print(f"Generation attempts needed:    {attempts}")

    return data


# 4. DATA SPLITTING AND BATCHING

def split_dataset(data, train_ratio=0.70, val_ratio=0.15):
    """
    Splits dataset into train, validation and test sets.
    Validation set is needed for early stopping and threshold tuning.
    """
    data = data.copy()
    random.shuffle(data)

    train_end = int(train_ratio * len(data))
    val_end = int((train_ratio + val_ratio) * len(data))

    train_set = data[:train_end]
    val_set = data[train_end:val_end]
    test_set = data[val_end:]

    print("\nDATA SPLIT")
    print("----------")
    print("Train:", Counter(label for _, label in train_set))
    print("Val:  ", Counter(label for _, label in val_set))
    print("Test: ", Counter(label for _, label in test_set))

    return train_set, val_set, test_set


def collate_batch(batch):
    sequences, labels = zip(*batch)

    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    padded_seqs = pad_sequence(sequences, batch_first=True)

    lengths = torch.tensor([len(seq) for seq in sequences])
    labels = torch.tensor(labels, dtype=torch.float32)

    return padded_seqs, lengths, labels


# 5. MODEL

class GRUWithSwapPairs(nn.Module):
    def __init__(
        self,
        embedding_dim=16,
        hidden_size=64,
        output_size=1,
        num_layers=2,
        dropout=0.0
    ):
        super().__init__()

        self.fc_in = nn.Linear(2, embedding_dim)

        # PyTorch GRU dropout only works when num_layers > 1
        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout
        )

        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        x = self.fc_in(x)

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, h_n = self.gru(packed)

        # Use final hidden state from the last GRU layer
        logits = self.fc_out(h_n[-1]).squeeze(1)

        # Important: no sigmoid here.
        # BCEWithLogitsLoss applies sigmoid internally during training.
        return logits


# 6. LOSS FUNCTIONS

def get_pos_weight(train_set):
    """
    Used for naturally imbalanced data.
    If positives are rare, pos_weight > 1 makes positive errors more costly.
    """
    labels = [label for _, label in train_set]
    counts = Counter(labels)

    num_neg = counts[0]
    num_pos = counts[1]

    pos_weight_value = num_neg / max(num_pos, 1)

    print(f"\nPositive class weight: {pos_weight_value:.4f}")

    return torch.tensor([pos_weight_value], dtype=torch.float32)


# 7. EVALUATION

def evaluate_loader(model, loader, loss_fn, threshold=0.5):
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    total_loss = 0

    with torch.no_grad():
        for padded_seqs, lengths, labels in loader:
            logits = model(padded_seqs, lengths)
            loss = loss_fn(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).int()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.int().cpu().numpy())

            total_loss += loss.item()

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "probs": np.array(all_probs),
        "preds": np.array(all_preds),
        "labels": np.array(all_labels)
    }


def find_best_threshold(model, val_loader, loss_fn):
    """
    Finds the threshold that gives the best validation F1 score.
    This is especially important for imbalanced data.
    """
    val_metrics = evaluate_loader(model, val_loader, loss_fn, threshold=0.5)

    probs = val_metrics["probs"]
    labels = val_metrics["labels"]

    best_threshold = 0.5
    best_f1 = -1

    for threshold in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= threshold).astype(int)
        current_f1 = f1_score(labels, preds, zero_division=0)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold, best_f1


# 8. TRAINING WITH EARLY STOPPING

def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    max_epochs=50,
    patience=5
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1": [],
        "val_f1": []
    }

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()

        for padded_seqs, lengths, labels in train_loader:
            optimizer.zero_grad()

            logits = model(padded_seqs, lengths)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

        train_metrics = evaluate_loader(model, train_loader, loss_fn, threshold=0.5)
        val_metrics = evaluate_loader(model, val_loader, loss_fn, threshold=0.5)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# 9. LEARNING CURVE PLOTS

def plot_learning_curves(history, title="Learning Curves"):
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"{title}: Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{title}: Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(history["train_f1"], label="Train F1")
    plt.plot(history["val_f1"], label="Validation F1")
    plt.title(f"{title}: F1 vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.grid(True)
    plt.show()


# 10. SINGLE EXPERIMENT RUN

def run_single_experiment(
    data,
    num_layers=2,
    hidden_size=64,
    dropout=0.2,
    learning_rate=0.0028,
    batch_size=32,
    max_epochs=50,
    patience=5,
    use_weighted_loss=False,
    seed=1,
    plot_curves=False
):
    set_seed(seed)

    train_set, val_set, test_set = split_dataset(data)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    model = GRUWithSwapPairs(
        embedding_dim=16,
        hidden_size=hidden_size,
        output_size=1,
        num_layers=num_layers,
        dropout=dropout
    )

    if use_weighted_loss:
        pos_weight = get_pos_weight(train_set)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()

    model, history = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        max_epochs=max_epochs,
        patience=patience
    )

    training_time = time.time() - start_time

    best_threshold, best_val_f1 = find_best_threshold(model, val_loader, loss_fn)

    test_metrics = evaluate_loader(
        model,
        test_loader,
        loss_fn,
        threshold=best_threshold
    )

    conf_matrix = confusion_matrix(
        test_metrics["labels"],
        test_metrics["preds"]
    )

    num_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    if plot_curves:
        plot_learning_curves(
            history,
            title=f"{num_layers} layers, hidden={hidden_size}, dropout={dropout}"
        )

    result = {
        "seed": seed,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "epochs_run": len(history["train_loss"]),
        "best_threshold": best_threshold,
        "best_val_f1": best_val_f1,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "num_params": num_params,
        "training_time_seconds": training_time,
        "confusion_matrix": conf_matrix,
        "history": history
    }

    print("\nFINAL TEST RESULTS")
    print("------------------")
    print(f"Layers:             {num_layers}")
    print(f"Hidden size:        {hidden_size}")
    print(f"Dropout:            {dropout}")
    print(f"Learning rate:      {learning_rate}")
    print(f"Epochs run:         {result['epochs_run']}")
    print(f"Best threshold:     {best_threshold:.2f}")
    print(f"Test Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"Test Precision:     {test_metrics['precision']:.4f}")
    print(f"Test Recall:        {test_metrics['recall']:.4f}")
    print(f"Test F1:            {test_metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    return result


# 11. REPEATED LAYER SWEEP

def run_layer_sweep(
    data,
    layer_options=None,
    trials=10,
    hidden_size=64,
    dropout=0.2,
    learning_rate=0.0028,
    batch_size=32,
    max_epochs=50,
    patience=5,
    use_weighted_loss=False
):
    if layer_options is None:
        layer_options = list(range(1, 11))

    results = []

    for num_layers in layer_options:
        for trial in range(trials):
            seed = 1000 + (num_layers * 100) + trial

            print("\n" + "=" * 70)
            print(f"Running layer sweep | Layers={num_layers} | Trial={trial + 1}/{trials}")
            print("=" * 70)

            result = run_single_experiment(
                data=data,
                num_layers=num_layers,
                hidden_size=hidden_size,
                dropout=dropout,
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                use_weighted_loss=use_weighted_loss,
                seed=seed,
                plot_curves=False
            )

            result["trial"] = trial + 1
            results.append(result)

    return results


# 12. RESULTS SUMMARY

def results_to_dataframe(results):
    rows = []

    for r in results:
        rows.append({
            "trial": r["trial"],
            "seed": r["seed"],
            "layers": r["num_layers"],
            "hidden_size": r["hidden_size"],
            "dropout": r["dropout"],
            "learning_rate": r["learning_rate"],
            "batch_size": r["batch_size"],
            "epochs_run": r["epochs_run"],
            "best_threshold": r["best_threshold"],
            "test_accuracy": r["test_accuracy"],
            "test_precision": r["test_precision"],
            "test_recall": r["test_recall"],
            "test_f1": r["test_f1"],
            "num_params": r["num_params"],
            "training_time_seconds": r["training_time_seconds"]
        })

    return pd.DataFrame(rows)


def summarize_results(df):
    summary = df.groupby("layers").agg(
        mean_accuracy=("test_accuracy", "mean"),
        sd_accuracy=("test_accuracy", "std"),
        mean_f1=("test_f1", "mean"),
        sd_f1=("test_f1", "std"),
        mean_precision=("test_precision", "mean"),
        sd_precision=("test_precision", "std"),
        mean_recall=("test_recall", "mean"),
        sd_recall=("test_recall", "std"),
        mean_epochs_run=("epochs_run", "mean"),
        mean_training_time=("training_time_seconds", "mean")
    ).reset_index()

    return summary


def flag_outliers_iqr(df, group_col="layers", metric="test_accuracy"):
    """
    Flags outliers using the IQR rule.
    This should be used for interpretation, not automatic deletion.
    """
    df = df.copy()
    df["is_outlier"] = False

    for group_value in df[group_col].unique():
        values = df[df[group_col] == group_value][metric]

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (
            (df[group_col] == group_value) &
            ((df[metric] < lower) | (df[metric] > upper))
        )

        df.loc[mask, "is_outlier"] = True

    return df


def plot_layer_summary(summary_df, metric="mean_accuracy", error_col="sd_accuracy"):
    plt.figure()
    plt.errorbar(
        summary_df["layers"],
        summary_df[metric],
        yerr=summary_df[error_col],
        marker="o",
        capsize=4
    )
    plt.title(f"Number of GRU Layers vs {metric}")
    plt.xlabel("Number of GRU Layers")
    plt.ylabel(metric)
    plt.grid(True)
    plt.show()

# 13. MAIN RUN SETTINGS

if __name__ == "__main__":

    # -----------------------------
    # Choose experiment type
    # -----------------------------
    # Options:
    # "natural_single"
    # "balanced_single"
    # "natural_layer_sweep"
    # "balanced_layer_sweep"

    EXPERIMENT_TYPE = "natural_layer_sweep"

    # Sampling parameters
    MAX_N = 10
    MIN_K = 1
    MAX_K = 20

    # For revision feedback, use at least 20,000 to 50,000 samples.
    NUM_NATURAL_SAMPLES = 30000

    # Balanced data with 15,000 per class gives 30,000 total.
    TARGET_PER_CLASS = 15000

    # Model / training defaults
    DEFAULT_LAYERS = 2
    DEFAULT_HIDDEN_SIZE = 64
    DEFAULT_DROPOUT = 0.2
    DEFAULT_LEARNING_RATE = 0.0028
    DEFAULT_BATCH_SIZE = 32
    MAX_EPOCHS = 50
    PATIENCE = 5

    # Use fewer trials if your computer is slow.
    # For the final paper, use 5 to 10.
    TRIALS = 10

    set_seed(42)

    # Generate dataset

    if "natural" in EXPERIMENT_TYPE:
        dataset, metadata_df = generate_natural_dataset(
            num_samples=NUM_NATURAL_SAMPLES,
            max_n=MAX_N,
            min_k=MIN_K,
            max_k=MAX_K
        )

        metadata_df.to_csv("natural_dataset_metadata.csv", index=False)

        USE_WEIGHTED_LOSS = True

    elif "balanced" in EXPERIMENT_TYPE:
        dataset = generate_balanced_dataset(
            target_per_class=TARGET_PER_CLASS,
            max_n=MAX_N,
            min_k=MIN_K,
            max_k=MAX_K
        )

        USE_WEIGHTED_LOSS = False

    else:
        raise ValueError("Invalid EXPERIMENT_TYPE selected.")


    # Run selected experiment

    if EXPERIMENT_TYPE in ["natural_single", "balanced_single"]:

        result = run_single_experiment(
            data=dataset,
            num_layers=DEFAULT_LAYERS,
            hidden_size=DEFAULT_HIDDEN_SIZE,
            dropout=DEFAULT_DROPOUT,
            learning_rate=DEFAULT_LEARNING_RATE,
            batch_size=DEFAULT_BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            use_weighted_loss=USE_WEIGHTED_LOSS,
            seed=42,
            plot_curves=True
        )

    elif EXPERIMENT_TYPE in ["natural_layer_sweep", "balanced_layer_sweep"]:

        layer_results = run_layer_sweep(
            data=dataset,
            layer_options=list(range(1, 11)),
            trials=TRIALS,
            hidden_size=DEFAULT_HIDDEN_SIZE,
            dropout=DEFAULT_DROPOUT,
            learning_rate=DEFAULT_LEARNING_RATE,
            batch_size=DEFAULT_BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            use_weighted_loss=USE_WEIGHTED_LOSS
        )

        raw_df = results_to_dataframe(layer_results)
        summary_df = summarize_results(raw_df)
        flagged_df = flag_outliers_iqr(raw_df, metric="test_accuracy")

        print("\nRAW RESULTS")
        print(raw_df)

        print("\nSUMMARY RESULTS: MEAN ± STANDARD DEVIATION")
        print(summary_df)

        print("\nFLAGGED OUTLIERS")
        print(flagged_df[flagged_df["is_outlier"]])

        raw_df.to_csv(f"{EXPERIMENT_TYPE}_raw_results.csv", index=False)
        summary_df.to_csv(f"{EXPERIMENT_TYPE}_summary_results.csv", index=False)
        flagged_df.to_csv(f"{EXPERIMENT_TYPE}_flagged_outliers.csv", index=False)

        plot_layer_summary(
            summary_df,
            metric="mean_accuracy",
            error_col="sd_accuracy"
        )

        plot_layer_summary(
            summary_df,
            metric="mean_f1",
            error_col="sd_f1"
        )
    else:
        raise ValueError("Invalid EXPERIMENT_TYPE selected.")