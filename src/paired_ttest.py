import os
import json
import argparse
import numpy as np
from scipy.stats import ttest_rel

METRICS = [
    "accuracy",
    "macro_f1",
    "weighted_f1"
]
def load_all_metrics(results_dir):
    """
    Load metric values indexed by seed from a directory of JSON files.
    """
    # Define an buffer empty dictionary to store metric values by seed
    data_by_seed = {}
    label_names = None # Initialize label_names to None to handle cases where it may not be present in the JSON files

    # fname looks at all files in the results directory and skip them if it does not end with .json
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        
        # Combines the results directory path with the filename to get the full path to the JSON file, then opens it and loads the data
        path = os.path.join(results_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)

        # Extract the seed and the specified metric value from the loaded JSON data, and store it in the values_by_seed dictionary using the seed as the key
        seed = data["seed"]
        data_by_seed[seed] = data

        # Save labels (same across files)
        if label_names is None:
            label_names = data["labels"]

    return data_by_seed, label_names

def compute_ttest(a_vals, b_vals):
    t_stat, p_value = ttest_rel(a_vals, b_vals)
    diff = a_vals - b_vals
    cohen_d = diff.mean() / (diff.std(ddof=1) + 1e-8)

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohen_d": float(cohen_d),
        "mean_a": float(a_vals.mean()),
        "mean_b": float(b_vals.mean()),
        "std_a": float(a_vals.std(ddof=1)),
        "std_b": float(b_vals.std(ddof=1)),
    }


def analyze(mlb_dir, concat_dir):
    mlb_data, mlb_labels = load_all_metrics(mlb_dir)
    concat_data, concat_labels = load_all_metrics(concat_dir)

    if mlb_labels != concat_labels:
        raise ValueError(
            f"Label mismatch:\n"
            f"MLB: {mlb_labels}\n"
            f"Concat: {concat_labels}"
        )

    labels = mlb_labels

    mlb_seeds = set(mlb_data.keys())
    concat_seeds = set(concat_data.keys())

    if mlb_seeds != concat_seeds:
        raise ValueError(
            f"Seed mismatch.\n"
            f"MLB only: {sorted(mlb_seeds - concat_seeds)}\n"
            f"Concat only: {sorted(concat_seeds - mlb_seeds)}"
        )

    common_seeds = sorted(mlb_seeds)

    if len(common_seeds) < 2:
        raise ValueError("Need at least 2 seeds")

    results = {}

    # -------------------
    # GLOBAL METRICS
    # -------------------
    for metric in METRICS:
        mlb_vals = np.array([mlb_data[s]["metrics"][metric] for s in common_seeds])
        concat_vals = np.array([concat_data[s]["metrics"][metric] for s in common_seeds])

        results[metric] = compute_ttest(mlb_vals, concat_vals)

    # -------------------
    # PER-CLASS RECALL
    # -------------------
    per_class_results = {}

    for i, label in enumerate(labels):
        mlb_vals = np.array([mlb_data[s]["recall_per_class"][i] for s in common_seeds])
        concat_vals = np.array([concat_data[s]["recall_per_class"][i] for s in common_seeds])

        per_class_results[label] = compute_ttest(mlb_vals, concat_vals)

    results["recall_per_class"] = per_class_results

    return results

def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "ttest_results.json")
    txt_path = os.path.join(output_dir, "ttest_report.txt")

    # JSON (for reproducibility)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # TXT (for reading / paper)
    with open(txt_path, "w") as f:
        f.write("=== Paired T-Test Results ===\n\n")

        for metric, res in results.items():
            if metric == "recall_per_class":
                f.write("\n--- Per-class Recall ---\n")
                for label, r in res.items():
                    f.write(f"{label}:\n")
                    f.write(f"  MLB mean: {r['mean_a']:.4f} ± {r['std_a']:.4f}\n")
                    f.write(f"  CONCAT mean: {r['mean_b']:.4f} ± {r['std_b']:.4f}\n")
                    f.write(f"  p-value: {r['p_value']:.6f}\n")
                    f.write(f"  Cohen's d: {r['cohen_d']:.4f}\n\n")
            else:
                f.write(f"{metric}:\n")
                f.write(f"  MLB mean: {res['mean_a']:.4f} ± {res['std_a']:.4f}\n")
                f.write(f"  CONCAT mean: {res['mean_b']:.4f} ± {res['std_b']:.4f}\n")
                f.write(f"  p-value: {res['p_value']:.6f}\n")
                f.write(f"  Cohen's d: {res['cohen_d']:.4f}\n\n")

    print(f"Saved JSON → {json_path}")
    print(f"Saved TXT  → {txt_path}")


def main():
    # Uses argparse library to handle code execution from terminal/command line
    # parser lines defines what arguments are allowed to change when running the script
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlb", type=str, required=True,
                        help="Directory containing per-seed MLB test-result JSON files")
    parser.add_argument("--concat", type=str, required=True,
                        help="Directory containing per-seed concatenation test-result JSON files")
    parser.add_argument("--out", default="results/meld/mean_pooling_4hz/statistical_analysis",
                        help="Directory for statistical analysis outputs")
    args = parser.parse_args()

    # Call the paired t-test function to calculate the t-statistic and p-value for the specified metric
    result = analyze(
        mlb_dir=args.mlb,
        concat_dir=args.concat,
    )
    save_results(result, args.out)


   


if __name__ == "__main__":
    main()
# Running args for MELD: py src/paired_ttest.py --mlb results/meld/mean_pooling_4hz/mlb --concat results/meld/mean_pooling_4hz/concat --out results/meld/mean_pooling_4hz/statistical_analysis
# Running args for IEMOCAP: py src/paired_ttest.py --mlb results/iemocap/mean_pooling_4hz/mlb --concat results/iemocap/mean_pooling_4hz/concat --out results/iemocap/mean_pooling_4hz/statistical_analysis