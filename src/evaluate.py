import os
import torch
import random
import argparse
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from cached_dataset import CachedFeatureDataset
from classifier import EmotionClassifier


def set_seed(seed: int):
    # Fixed random seed to force sources of randomness to be deterministic
    # Define the seed that defines the starting point for generating the same sequence across runs of random numbers
    # This ensures that every time the code is run with the same seed, it produces the same results
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model(model_path, feature_dir: str, temporal_encoder: str, fusion_type="mlb", hidden_dim=128, proj_dim=128, dropout=0.3, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluate a saved model on the MELD test split.

    Args:
        model_path (str): Path to the saved model checkpoint (.pt).
        root_dir (str): Path to MELD.Raw folder (for CSVs).
        frames_dir (str): Path to MELD.Frames folder (for .jpg frames).
        device (str): 'cuda' or 'cpu'.

    Returns:
        dict: dictionary with accuracy, precision, recall, and f1-score.
    """
    # Load dataset 
    test_dataset = CachedFeatureDataset(feature_dir + "test")
    
    for name, ds in [("test", test_dataset)]:
        text, image, label = ds[0][:3]

        print(type(ds))
        print(len(ds))
        print(f"\n{name.upper()}")
        print("text :", text.shape)
        print("image:", image.shape)
        print("label:", label)


    emotions = test_dataset.label_names  # Get the list of emotion labels
    num_classes = int(len(test_dataset.label_names)) # Get number of unique emotion labels

    # Load model 

    model = EmotionClassifier(
        hidden_dim=hidden_dim, 
        num_classes=num_classes,
        clip_dim=768,
        proj_dim=proj_dim, 
        dropout=dropout,
        fusion_type=fusion_type,
        temporal_encoder=temporal_encoder,
    ).to(device)
    print(f"Fusion type (evaluation): {fusion_type}")
    # Load the model best learned parameters (weights and biases)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Switch model to evaluation (inference) mode
    model.eval()


    # Store true and predicted labels
    y_true, y_pred = [], []
    sample_ids = []

    # Iterate through test data 
    with torch.no_grad():
        for text, image, labels, sample_id in test_dataset:
            # Move tensors to GPU
            text_features = text.unsqueeze(0).to(device) # [1, 512]
            image_features = image.unsqueeze(0).to(device) # [1, T, 512]
            
            # Create mask (all valid since no padding here)
            T = image_features.size(1)
            mask = torch.ones(1, T, dtype=torch.bool).to(device)  # [1, T]
            MAX_T = 140

            if image_features.size(1) > MAX_T:
                print(f"[DEBUG] Truncated sequence from {image_features.size(1)} → {MAX_T}")
                image_features = image_features[:, :MAX_T]
                mask = mask[:, :MAX_T]
            
            labels = labels.to(device)                     # [] 

            
            logits = model(text_features, image_features, mask)
            # logits are the raw output scores from the model for each class
            # Use torch.argmax to get the index of the class with the highest score
            # Use .item to convert single-value tensor to a standard Python number
            pred_class = torch.argmax(logits, dim=1).item()

            # Append true and predicted labels
            y_true.append(labels.item())
            y_pred.append(pred_class)
            sample_ids.append(sample_id)
    # Compute metrics 
    acc = accuracy_score(y_true, y_pred)
    # Use average="weighted" to calculate average metrics across all classes
    # weight by how many true instances there are for each label
    # zero_division=0 to handle any potential division by zero
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels = list(range(num_classes)))

    # Per-class recall
    recall_per_class = recall_score(
        y_true,
        y_pred,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0
    )
    
    # Store results in a dictionary
    results = {
        "accuracy": acc,
        #Macro
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        # Weighted
        "weighted_precision": prec_weighted,
        "weighted_recall": rec_weighted,
        "weighted_f1": f1_weighted,
        "recall_per_class": recall_per_class.tolist(),  # Convert numpy array to list for JSON serialization
        "confusion_matrix": cm.tolist(),  # Convert numpy array to list so it can be saved as JSON object
        "labels": emotions,  # Store the order of labels for reference
        "y_true": y_true,
        "y_pred": y_pred,
        "sample_ids": sample_ids
    }

    # Print results
    print("\n--- Test Evaluation ---")
    # Store each Key and Value from results dictionary
    for k, v in results.items():
        if isinstance(v, (float, int)):
            print(f"{k.capitalize()}: {v:.4f}")
    return results

def main():
    # Uses argparse library to handle code execution from terminal/command line
    # parser lines defines what arguments are allowed to change when running the script
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_json", type=str, required=True)
    parser.add_argument("--feature_root", type=str, default="data/features")
    args = parser.parse_args()

    # Load run configuration from JSON file
    with open(args.run_json, "r") as f:
        run_info = json.load(f)

    # Set random seed for reproducibility
    seed = run_info["seed"]
    set_seed(seed)
    # Extract configuration details
    model_file = run_info["model_path"]
    pipeline = run_info["pipeline"]
    hidden_dim = run_info.get("hidden_dim", 128)
    proj_dim = run_info.get("proj_dim", 128)
    dropout = run_info.get("dropout", 0.3)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    experiment = run_info.get("experiment", {})
    dataset_name = experiment.get("dataset", "unknown_dataset")
    fusion_type = experiment.get("fusion","mlb")
    temporal_encoder = experiment.get("temporal_encoder", "conformer")
    exp_name = experiment.get("name", "unknown_experiment")

    print("\n--- Evaluation Configuration ---")
    print(f"Seed: {seed}")
    print(f"Pipeline: {pipeline}")
    print(f"Device: {device_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Model path: {model_file}")
    print(f"Temporal Encoder: {temporal_encoder}")
    print("--------------------------------\n")
    
   
    # Define feature directory for evaluation
    dataset_name = dataset_name.lower()  # e.g., "meld" or "iemocap"
    if dataset_name == "meld":
        feature_dir = "data/meld_features_4Hz/meld_"
    elif dataset_name == "iemocap":
        feature_dir = "data/iemocap_features/iemocap_"

    # Evaluate the model    
    results = evaluate_model(model_file, feature_dir, temporal_encoder, fusion_type=fusion_type, hidden_dim=hidden_dim, proj_dim=proj_dim, dropout=dropout)

    # Define directory to save json object for paired t-test analysis
    base_dir = os.path.join("results", exp_name, "test")

    # Create directory for test results if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

     # Timestamp for the results file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create test metrics file
    metrics_file = os.path.join(base_dir, f"test_metrics_{exp_name}_{seed}_{timestamp}.json")
    log_file = os.path.join(base_dir, f"test_log_{exp_name}_{seed}_{timestamp}.txt")
    cm_file = os.path.join(base_dir, f"confusion_matrix_{exp_name}_{seed}_{timestamp}.png")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(os.path.dirname(cm_file), exist_ok=True)

    # Prepare test record dictionary
    test_record = {
        "dataset": dataset_name.upper(),
        "seed": seed,
        "pipeline": pipeline,
        "device": device_name,
        "model_path": model_file,
        "timestamp": timestamp,
        "experiment": experiment,
        "metrics": {
            "accuracy": results["accuracy"],
             # Macro
            "macro_precision": results["macro_precision"],
            "macro_recall": results["macro_recall"],
            "macro_f1": results["macro_f1"],
            # Weighted
            "weighted_precision": results["weighted_precision"],
            "weighted_recall": results["weighted_recall"],
            "weighted_f1": results["weighted_f1"]
        },
        "recall_per_class": results["recall_per_class"],
        "confusion_matrix": results["confusion_matrix"],
        "labels": results["labels"],
        "sample_ids": results["sample_ids"],
        "y_true": results["y_true"],
        "y_pred": results["y_pred"]
    }
    
    # Save results to a JSON file
    with open(metrics_file, "w") as f:
        # Write the Python Object results as a JSON formatted stream to the file
        json.dump(test_record, f, indent=4)

    # Write results to test metrics file
    with open(log_file, "a") as f:
        f.write("----- Test Evaluation -----\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Pipeline: {pipeline}\n")
        f.write(f"Device: {device_name}\n")
        f.write(f"Model path: {model_file}\n\n")
        f.write("Experiment Configuration:\n")
        for k, v in experiment.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        f.write(f"Results:\n")
        for metric, value in results.items():
            if isinstance(value, (float, int)):
                f.write(f"  {metric}: {value:.4f}\n")
        f.write("Per-class Recall:\n")
        for label, value in zip(results["labels"], results["recall_per_class"]):
            f.write(f"  {label}: {value:.4f}\n")
        f.write("\n")

    # Save confusion matrix as an image
    # Restore the confusion matrix list back to numpy array
    cm = np.array(results["confusion_matrix"])
    labels = results["labels"]

    plt.figure(figsize=(10, 8))
    # Use seaborn to visualize the confusion matrix as a colored grid ("heatmap")
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_name.upper()} Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(cm_file)
    plt.close()

    print(f"\nMetrics saved to: {metrics_file}")
    print(f"Log saved to: {log_file}")
    print(f"Confusion matrix image will be saved to: {cm_file}")

if __name__ == "__main__":
    main()
# Running args: python src/evaluate.py --run_json results\temporal_conforme_lite_regularization\train\metrics_seed_temporal_conforme_lite_regularization_42_2026-05-06_18-06-09.json