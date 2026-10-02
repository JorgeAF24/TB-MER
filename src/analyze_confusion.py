import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ================================
# LOAD CONFUSION MATRICES
# ================================
def load_confusion_matrices(results_dir):
    cms_by_seed = {}
    labels = None

    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(results_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)

        seed = data["seed"]
        if seed in cms_by_seed:
            raise ValueError(f"Duplicate seed {seed} in {results_dir}")

        cms_by_seed[seed] = np.array(data["confusion_matrix"], dtype=float)

        if labels is None:
            labels = data["labels"]
        elif labels != data["labels"]:
            raise ValueError(f"Inconsistent label ordering in {results_dir}")

    if not cms_by_seed:
        raise ValueError(f"No JSON files found in {results_dir}")

    return cms_by_seed, labels


# ================================
# MICRO-AVERAGED CONFUSION MATRIX
# ================================
def compute_average_cm(cms):
    total_raw_cm = cms.sum(axis=0)

    row_sums = total_raw_cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1

    normalized_avg_cm = total_raw_cm / row_sums
    return normalized_avg_cm

def plot_flow_arrows(flows, labels, save_path, threshold=1.0):
    """
    Visualize dominant flows as arrows between classes.

    threshold: minimum % change to display (cleaner figure)
    """
    n = len(labels)

    # Place nodes in a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = {
        i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)
    }

    plt.figure(figsize=(9, 9))  # Slightly increased size to match expanded viewport

    # 1. Draw nodes with smart, symmetrical label positioning
    for i, (x, y) in positions.items():
        # Draw the colored anchor circle (zorder keeps it above the lines)
        plt.scatter(x, y, s=300, zorder=3)
        
        # Calculate dynamic alignment padding based on the node's angle
        # This pushes text outward radially from the center of the circle
        offset_x = x * 1.20
        offset_y = y * 1.20
        
        # Dynamically set alignment anchors based on which side of the circle the node sits
        if abs(x) < 0.1:  # Top or Bottom nodes (e.g., fear, sadness)
            ha = 'center'
            va = 'bottom' if y > 0 else 'top'
        elif x > 0:       # Right side nodes (e.g., disgust, anger, surprise)
            ha = 'left'
            va = 'center'
        else:             # Left side nodes (e.g., joy, neutral)
            ha = 'right'
            va = 'center'

        # Minor manual nudge to prevent labels from touching the dots directly
        if va == 'bottom': offset_y += 0.03
        if va == 'top':    offset_y -= 0.03

        plt.text(
            offset_x, offset_y, 
            labels[i], 
            ha=ha, 
            va=va, 
            fontsize=10, 
            weight="bold"
        )

    # 2. Draw arrows using professional annotation anchors
    for val, i, j in flows:
        val_pct = val * 100

        if abs(val_pct) < threshold:
            continue

        x1, y1 = positions[i]  # Source node
        x2, y2 = positions[j]  # Target node

        color = "red" if val > 0 else "blue"
        
        # Scale the line width dynamically based on the percentage size
        width = max(0.5, abs(val_pct) * 0.5) 

        # This draws a beautiful arrow perfectly anchored between the two positions
        plt.annotate(
            "", 
            xy=(x2, y2),      # Pointing at the target node
            xytext=(x1, y1),  # Starting from the source node
            arrowprops=dict(
                arrowstyle="->", 
                color=color, 
                lw=width,
                alpha=0.6,
                shrinkA=15,   # Spacing from the source node circle (in points)
                shrinkB=15    # Spacing from the target node circle (in points)
            )
        )

        # 3. Label the arrow at the exact midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Add a clean background box behind the text so lines don't slice through it
        plt.text(
            mid_x, mid_y, 
            f"{val_pct:+.1f}%", 
            fontsize=9, 
            color=color,
            weight="bold",
            ha="center", 
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
        )

    # CRITICAL FIX FOR FEAR & SYMMETRY: Expand limits so text isn't clipped by bounding boxes
    plt.xlim(-1.4, 1.4)
    plt.ylim(-1.4, 1.4)

    plt.title("Error Flow Graph (MLB - Concat)", fontsize=12, pad=20, weight="bold")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300) # Added higher dpi for sharp text rendering in papers
    plt.close()


# ================================
# RECALL PER CLASS
# ================================
def compute_mean_recall_per_class(results_dir):
    labels = None
    recalls = []

    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(results_dir, fname)

        with open(path, "r") as f:
            data = json.load(f)

        recalls.append(np.array(data["recall_per_class"], dtype=float))

        if labels is None:
            labels = data["labels"]
        elif labels != data["labels"]:
            raise ValueError(f"Inconsistent label ordering in {results_dir}")
    
    if len(recalls) == 0:
        raise ValueError(f"No JSON files found in {results_dir}")
    
    recalls = np.stack(recalls)

    mean_recall = recalls.mean(axis=0)

    return mean_recall, labels


# ================================
# FLOW EXTRACTION
# ================================
def extract_top_flows(delta_cm, labels, top_k=8):
    flows = []

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue

            value = delta_cm[i, j]
            flows.append((value, i, j))

    flows = sorted(flows, key=lambda x: abs(x[0]), reverse=True)

    return flows[:top_k]


# ================================
# PLOT
# ================================
def plot_delta_cm(delta_cm, labels, delta_recall, save_path):

    labels_with_recall = [
        f"{label}\nΔR={delta_recall[i]:+.2f}%"
        for i, label in enumerate(labels)
    ]

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        delta_cm,
        annot=True,
        fmt=".2f",
        xticklabels=labels_with_recall,
        yticklabels=labels_with_recall,
        cmap="coolwarm",
        center=0
    )

    plt.title("Δ Confusion Matrix (Percentage Points) (MLB - Concatenation)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()


# ================================
# MAIN
# ================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlb", type=str, required=True)
    parser.add_argument("--concat", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/confusion_analyze")

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load CM
    mlb_cms_by_seed, mlb_labels = load_confusion_matrices(args.mlb)
    concat_cms_by_seed, concat_labels = load_confusion_matrices(args.concat)

    if mlb_labels != concat_labels:
        raise ValueError("MLB and Concat label ordering does not match.")
    
    mlb_seeds = set(mlb_cms_by_seed)
    concat_seeds = set(concat_cms_by_seed)

    if mlb_seeds != concat_seeds:
        raise ValueError(
            f"Seed mismatch.\n"
            f"MLB only: {sorted(mlb_seeds - concat_seeds)}\n"
            f"Concat only: {sorted(concat_seeds - mlb_seeds)}"
        )
    
    seeds = sorted(mlb_seeds)
    labels = mlb_labels

    mlb_cms = np.stack([mlb_cms_by_seed[s] for s in seeds])
    concat_cms = np.stack([concat_cms_by_seed[s] for s in seeds])
    
    # Compute averages
    mlb_avg = compute_average_cm(mlb_cms)
    concat_avg = compute_average_cm(concat_cms)

    delta_cm = mlb_avg - concat_avg

    # Convert to %
    mlb_avg_pct = mlb_avg * 100
    concat_avg_pct = concat_avg * 100
    delta_cm_pct = delta_cm * 100

    # ============================
    # RECALL ANALYSIS
    # ============================
    mlb_recall, _ = compute_mean_recall_per_class(args.mlb)
    concat_recall, _ = compute_mean_recall_per_class(args.concat)

    delta_recall = (mlb_recall - concat_recall) * 100  # Convert to percentage

    print("\n=== Per-Class Recall (Δ) ===")
    for i, label in enumerate(labels):
        print(f"{label}: {delta_recall[i]:+.3f}")

    # ============================
    # FLOW ANALYSIS
    # ============================
    flows = extract_top_flows(delta_cm, labels)

    print("\n=== Top Error Flows ===")
    for val, i, j in flows:
        print(f"{labels[i]} → {labels[j]} : {val:+.3f}")

    # ============================
    # SAVE FILES
    # ============================
    json_path = os.path.join(args.out, "confusion_analysis.json")
    plot_path = os.path.join(args.out, "confusion_delta.png")
    flow_plot_path = os.path.join(args.out, "error_flow_arrows.png")

    results = {
        "labels": labels,
        "mlb_avg_percentage": mlb_avg_pct.tolist(),
        "concat_avg_percentage": concat_avg_pct.tolist(),
        "delta_cm_percentage": delta_cm_pct.tolist(),
        "delta_recall": delta_recall.tolist(),
        "top_flows": [
            {
                "from": labels[i],
                "to": labels[j],
                "delta_percentage_points": float(val * 100)
            }
            for val, i, j in flows
        ]
    }

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # Plot
    plot_delta_cm(delta_cm_pct, labels, delta_recall, plot_path)
    plot_flow_arrows(flows, labels, flow_plot_path, threshold=1.0)

    print("\n=== Confusion Matrix Analysis ===")
    print(f"Saved JSON → {json_path}")
    print(f"Saved PNG  → {plot_path}")
    print(f"Saved PNG → {flow_plot_path}")


if __name__ == "__main__":
    main()
#Running args: py src/analyze_confusion.py --mlb results/meld/mean_pooling_4hz/mlb --concat results/meld/mean_pooling_4hz/concat  --out results/meld/mean_pooling_4hz/confusion_analysis