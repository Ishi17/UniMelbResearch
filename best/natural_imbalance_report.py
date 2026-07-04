# ============================================================
# Natural Class Imbalance Report
# ============================================================
# Purpose:
# This script calculates the natural class imbalance from the
# already-generated natural_dataset_metadata.csv file.
#
# It does NOT train the model.
# It does NOT rerun the layer sweep.
# It only reads the saved metadata file and reports the label distribution.
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = "natural_dataset_metadata.csv"
OUTPUT_SUMMARY_FILE = "natural_class_imbalance_summary.csv"
OUTPUT_GRAPH_FILE = "natural_class_imbalance_graph.png"


# ============================================================
# 2. LOAD METADATA
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE}.\n"
        "Make sure this script is in the same folder as natural_dataset_metadata.csv.\n"
        "You do not need to rerun training, but the metadata CSV must already exist."
    )

metadata_df = pd.read_csv(INPUT_FILE)

if "label" not in metadata_df.columns:
    raise ValueError(
        "The metadata file must contain a column called 'label'.\n"
        "Expected labels: 0 = not complete cycle, 1 = complete cycle."
    )


# ============================================================
# 3. CALCULATE CLASS IMBALANCE
# ============================================================

total_samples = len(metadata_df)

class_counts = metadata_df["label"].value_counts().sort_index()

negative_count = int(class_counts.get(0, 0))
positive_count = int(class_counts.get(1, 0))

negative_percentage = (negative_count / total_samples) * 100
positive_percentage = (positive_count / total_samples) * 100

imbalance_ratio = negative_count / max(positive_count, 1)


# ============================================================
# 4. CREATE SUMMARY TABLE
# ============================================================

summary_df = pd.DataFrame({
    "Class": ["Not complete cycle", "Complete cycle"],
    "Label": [0, 1],
    "Count": [negative_count, positive_count],
    "Percentage": [negative_percentage, positive_percentage]
})

summary_df.to_csv(OUTPUT_SUMMARY_FILE, index=False)


# ============================================================
# 5. PRINT RESULTS
# ============================================================

print("\nNATURAL CLASS IMBALANCE REPORT")
print("==============================")
print(f"Total samples: {total_samples}")
print()
print(f"Class 0 / Not complete cycle: {negative_count} samples")
print(f"Percentage: {negative_percentage:.2f}%")
print()
print(f"Class 1 / Complete cycle: {positive_count} samples")
print(f"Percentage: {positive_percentage:.2f}%")
print()
print(f"Imbalance ratio, Class 0 : Class 1 = {imbalance_ratio:.2f} : 1")
print()
print(f"Saved summary table to: {OUTPUT_SUMMARY_FILE}")


# ============================================================
# 6. CREATE BAR GRAPH
# ============================================================

plt.figure(figsize=(7, 5))

plt.bar(
    summary_df["Class"],
    summary_df["Count"]
)

plt.title("Natural Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.grid(axis="y", alpha=0.3)

# Add count and percentage labels above bars
for index, row in summary_df.iterrows():
    plt.text(
        index,
        row["Count"],
        f"{int(row['Count'])}\n({row['Percentage']:.2f}%)",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(OUTPUT_GRAPH_FILE, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved graph to: {OUTPUT_GRAPH_FILE}")


# ============================================================
# 7. PAPER-READY SENTENCE
# ============================================================

print("\nPAPER-READY SENTENCE")
print("====================")
print(
    f"Under the natural sampling procedure, the dataset contained "
    f"{negative_count} non-complete-cycle examples ({negative_percentage:.2f}%) "
    f"and {positive_count} complete-cycle examples ({positive_percentage:.2f}%), "
    f"giving an imbalance ratio of approximately {imbalance_ratio:.2f}:1 "
    f"between Class 0 and Class 1."
)