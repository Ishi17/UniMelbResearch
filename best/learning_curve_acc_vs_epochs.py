# ============================================================
# Selected Learning Curves for GRU Layer Configurations
# ============================================================
# Purpose:
# Run only a few selected GRU configurations and save
# accuracy/loss vs epoch graphs.
#
# This does NOT rerun the full 1-10 layer sweep.
# It only reruns selected representative layer counts.
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import functions from your main experiment file.
# Rename your main file to gru_cycle_model.py first.
from revisions import (
    set_seed,
    generate_natural_dataset,
    run_single_experiment
)


# ============================================================
# 1. SETTINGS
# ============================================================

MAX_N = 10
MIN_K = 1
MAX_K = 20
NUM_NATURAL_SAMPLES = 30000

HIDDEN_SIZE = 64
DROPOUT = 0.2
LEARNING_RATE = 0.0028
BATCH_SIZE = 32
MAX_EPOCHS = 50
PATIENCE = 5
USE_WEIGHTED_LOSS = True

# Representative layers to plot
SELECTED_LAYERS = [3, 6, 10]

# Use one reproducible trial per layer
BASE_SEED = 9000

OUTPUT_DIR = "selected_learning_curves"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. PLOT FUNCTION
# ============================================================

def plot_learning_curves_from_history(history, title, save_file):
    epochs = range(1, len(history["train_accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy curve
    axes[0].plot(epochs, history["train_accuracy"], marker="o", label="Training accuracy")
    axes[0].plot(epochs, history["val_accuracy"], marker="o", label="Validation accuracy")
    axes[0].set_title("Accuracy vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True)
    axes[0].legend()

    # Loss curve
    axes[1].plot(epochs, history["train_loss"], marker="o", label="Training loss")
    axes[1].plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    axes[1].set_title("Loss vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()

    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.show()


def save_history_csv(history, save_file):
    epochs = range(1, len(history["train_accuracy"]) + 1)

    history_df = pd.DataFrame({
        "epoch": list(epochs),
        "train_accuracy": history["train_accuracy"],
        "val_accuracy": history["val_accuracy"],
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "train_f1": history["train_f1"],
        "val_f1": history["val_f1"]
    })

    history_df.to_csv(save_file, index=False)


# ============================================================
# 3. GENERATE SAME NATURAL DATASET
# ============================================================

set_seed(42)

dataset, metadata_df = generate_natural_dataset(
    num_samples=NUM_NATURAL_SAMPLES,
    max_n=MAX_N,
    min_k=MIN_K,
    max_k=MAX_K
)

metadata_df.to_csv(
    os.path.join(OUTPUT_DIR, "selected_learning_curve_metadata.csv"),
    index=False
)


# ============================================================
# 4. RUN SELECTED CONFIGURATIONS ONLY
# ============================================================

summary_rows = []

for layer in SELECTED_LAYERS:
    seed = BASE_SEED + layer

    print("\n" + "=" * 70)
    print(f"Running selected learning curve model | Layers={layer}")
    print("=" * 70)

    result = run_single_experiment(
        data=dataset,
        num_layers=layer,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        use_weighted_loss=USE_WEIGHTED_LOSS,
        seed=seed,
        plot_curves=False
    )

    history = result["history"]

    csv_file = os.path.join(
        OUTPUT_DIR,
        f"layers_{layer}_history.csv"
    )

    graph_file = os.path.join(
        OUTPUT_DIR,
        f"layers_{layer}_learning_curve.png"
    )

    save_history_csv(history, csv_file)

    plot_learning_curves_from_history(
        history=history,
        title=f"{layer}-Layer GRU Learning Curve",
        save_file=graph_file
    )

    summary_rows.append({
        "layers": layer,
        "seed": seed,
        "epochs_run": result["epochs_run"],
        "best_threshold": result["best_threshold"],
        "test_accuracy": result["test_accuracy"],
        "test_precision": result["test_precision"],
        "test_recall": result["test_recall"],
        "test_f1": result["test_f1"],
        "history_csv": csv_file,
        "graph_file": graph_file
    })


# ============================================================
# 5. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary_rows)

summary_file = os.path.join(
    OUTPUT_DIR,
    "selected_learning_curve_summary.csv"
)

summary_df.to_csv(summary_file, index=False)

print("\nSELECTED LEARNING CURVE SUMMARY")
print("===============================")
print(summary_df)

print(f"\nSaved summary to: {summary_file}")
print(f"Saved graphs in: {OUTPUT_DIR}")