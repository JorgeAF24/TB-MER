import os
import json
import random
import datetime
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# DataLoader manipulates the dataset and forms batches
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from cached_dataset import CachedFeatureDataset
#from features import CLIPFeatureExtractor
from classifier import EmotionClassifier


def collate_fn(batch):
    """
    batch:
        [(text_feat, image_feat_seq, label), ...]

    image_feat_seq: Tensor [T_i, 768]

    Returns:
        text_feats  : [B, 768]
        images      : [B, T_max, 768]
        mask        : [B, T_max]
        labels      : [B]
    """
    
    text_feats, image_seqs, labels, sample_ids = zip(*batch)

    # Stack text
    text_feats = torch.stack(text_feats)  # [B, 768]
    labels = torch.stack(labels)

    # --- PAD IMAGE SEQUENCES ---
    lengths = [seq.size(0) for seq in image_seqs]
    T_max = max(lengths)

    B = len(image_seqs)
    D = image_seqs[0].size(1)

    padded = torch.zeros(B, T_max, D, dtype=image_seqs[0].dtype)
    mask = torch.zeros(B, T_max, dtype=torch.bool)
    
    for i, seq in enumerate(image_seqs):
        T_i = seq.size(0)
        padded[i, :T_i] = seq
        mask[i, :T_i] = True

    return text_feats, padded, mask, labels
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

def get_batch_preparer(device, gpu_resident):
    """
    Returns a function to prepare each batch depending on the mode.
    """
    if gpu_resident:
        # Data is already on GPU -> no transfer needed
        def prepare_batch(text, image, mask, label):
            return (
                text.to(device),
                image.to(device),
                mask.to(device),
                label.to(device),
            )
    else:
        def prepare_batch(text, image, mask, label):
            # Move data from CPU to GPU for each batch
            return (
                text.to(device, non_blocking=True),
                image.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
                label.to(device, non_blocking=True),
            )
    return prepare_batch
# -------------------------------
# Training function
# -------------------------------
def train_epoch(model, dataloader, optimizer, loss_fn, prepare_batch, debug=False):
    """
    Train the classifier for one epoch.

    Args:
        model (nn.Module): EmotionClassifier instance.
        dataloader (DataLoader): PyTorch DataLoader for the training set.
        optimizer (torch.optim.Optimizer): Optimizer (e.g., Adam).
        loss_fn (nn.Module): Loss function (e.g., CrossEntropyLoss).
        device (str): "cuda" or "cpu".

    Returns:
        avg_loss (float): Average training loss for the epoch.
        accuracy (float): Training accuracy for the epoch.
    """   
    # Setting the model to training mode
    model.train()
    # Initialize metrics
    # total_loss accumulates the loss over all batches
    # correct counts the number of correct predictions
    # total counts the total number of samples processed
    total_loss, correct, total = 0, 0, 0

    for text, image, mask, label, *_ in dataloader:
        # Prepare batch based on gpu_resident flag
        text_features, image_features, mask, batch_labels = prepare_batch(text, image, mask, label)
        
        # ===== DEBUG BLOCK =====
        if debug:
            print("image:", image_features.shape)   # [B, T_max, 768]
            print("mask:", mask.shape)              # [B, T_max]
            print("mask example:", mask[0])
        
        # Forward pass
        logits = model(text_features, image_features, mask)
        loss = loss_fn(logits, batch_labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        # loss returns the average loss for the batch
        # .item() converts the PyTorch tensor containing a single value to a standard Python number
        # Multiply by len(batch_labels) to get the total loss for the batch
        batch_size = batch_labels.size(0)
        total_loss += loss.item() * batch_size
        # Use argmax to get the predicted class index for each sample in the batch
        # The index corresponds to the class with the highest predicted score
        preds = logits.argmax(dim=1)
        correct += (preds == batch_labels).sum().item()
        total += batch_size

    # Compute average loss and accuracy
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, loss_fn, prepare_batch, debug=False):
    # Setting the model to evaluation mode
    model.eval()
    # Initialize metrics
    # total_loss accumulates the loss over all batches
    # correct counts the number of correct predictions
    # total counts the total number of samples processed
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for text, image, mask, label, *_ in dataloader:
            # Prepare batch based on gpu_resident flag
            text_features, image_features, mask, batch_labels = prepare_batch(text, image, mask, label)

            # ===== DEBUG BLOCK =====
            if debug:
                print("image:", image_features.shape)   # [B, T_max, 768]
                print("mask:", mask.shape)              # [B, T_max]
                print("mask example:", mask[0])
        
            # Forward pass
            logits = model(text_features, image_features, mask)
            loss = loss_fn(logits, batch_labels)

            # Metrics
            # loss returns the average loss for the batch
            # .item() converts the PyTorch tensor containing a single value to a standard Python number
            # Multiply by len(batch_labels) to get the total loss for the batch
            batch_size = batch_labels.size(0)
            total_loss += loss.item() * batch_size
            # Use argmax to get the predicted class index for each sample in the batch
            # The index corresponds to the class with the highest predicted score
            preds = logits.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_size
    # Compute average loss and accuracy
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


if __name__ == "__main__":
    # -------------------------------
    # Setup
    # -------------------------------
    # Uses argparse library to handle code execution from terminal/command line
    # parser lines defines what arguments are allowed to change when running the script
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu_resident", action="store_true")
    parser.add_argument("--fusion", type=str, default="mlb", choices=["mlb", "concat"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dataset", type=str, default="meld", choices=["meld", "iemocap"])
    parser.add_argument("--feature_root", type=str, default="data/features")
    parser.add_argument("--temporal_encoder", type=str, default="conformer", choices=["conformer", "none"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)

    args = parser.parse_args()
    
    # Set random seed for reproducibility
    seed = args.seed
    set_seed(seed)
    # Define configuration details
    pipeline = "Cached_Features"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    # Describe fusion method clearly
    if args.fusion == "mlb":
        fusion_desc = "MLB"
    elif args.fusion == "concat":
        fusion_desc = "concat + linear projection"
    else:
        raise ValueError(f"Unknown fusion type: {args.fusion}")
    # Experiment Configuration
    experiment = {
            "name": f"temporal_{args.temporal_encoder}_masked_4Hz_{args.dataset}_{args.fusion}_seed{seed}", # Encode the main design choice being tested
            "dataset": args.dataset.upper(), # Dataset used for training and evaluation
            "backbone": "CLIP ViT-L/14", # Feature extractor used to encode text and images into embeddings
            "fusion": args.fusion, # Method used to combine text and image features
            "temporal_handling": "full_conformer", # Method used to handle the temporal dimension T of image features 
            "temporal_encoder": args.temporal_encoder, # Specific architecture used for temporal encoding (if applicable) 
            "input_format": f"[B,T,768] -> projection -> {fusion_desc} -> {args.temporal_encoder} -> pooling -> [B,128]", # Transformation applied to feature tensor before classifier
        }

    exp_name = experiment["name"]

    # Get the appropriate batch preparation function based on gpu_resident flag
    prepare_batch = get_batch_preparer(device, args.gpu_resident)

    #Create a directory to save results
    base_dir = os.path.join("results", exp_name, "train")
    os.makedirs(base_dir, exist_ok=True)

    # Timestamp for each run
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # File Paths
    results_file = os.path.join(base_dir, f"log_seed_{exp_name}_{seed}_{timestamp}.txt")
    metrics_file = os.path.join(base_dir, f"metrics_seed_{exp_name}_{seed}_{timestamp}.json")
    model_file = os.path.join(base_dir, f"model_seed_{exp_name}_{seed}_{timestamp}.pt")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    
    if args.dataset == "meld":
        feature_root = "data/meld_features_4Hz/meld_"
    elif args.dataset == "iemocap":
        feature_root = "data/iemocap_features/iemocap_"
    
    # Datasets with cached features
    train_dataset = CachedFeatureDataset(feature_root + "train")
    dev_dataset = CachedFeatureDataset(feature_root + "dev")

    for name, ds in [
        ("train", train_dataset),
        ("dev", dev_dataset),
    ]:
        text, image, label = ds[0][:3]

        print(type(ds))
        print(len(ds))
        print(f"\n{name.upper()}")
        print("text :", text.shape)
        print("image:", image.shape)
        print("label:", label)


    # Move datasets to GPU if gpu_resident is True
    if args.gpu_resident:
        print("Moving entire dataset to GPU...")
        train_dataset = train_dataset.to(device)
        dev_dataset = dev_dataset.to(device)

    # Get number of unique emotion labels
    num_classes = int(len(train_dataset.label_names))

    if args.gpu_resident:
        num_workers = 0
        pin_memory = False
    else: 
        num_workers = args.num_workers
        pin_memory = True

    # Use dataloader to create batches and shuffle the training data
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    # Use dataloader for the validation set without shuffling
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)



    # Initialize the classifier
    classifier = EmotionClassifier(
        hidden_dim=args.hidden_dim, 
        num_classes=num_classes,
        clip_dim=768,
        proj_dim=args.proj_dim, 
        dropout=args.dropout,
        fusion_type=args.fusion,
        temporal_encoder=args.temporal_encoder,
        debug=args.debug
    ).to(device)

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.arange(num_classes),
                                         y=train_dataset.labels.cpu().numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Modify loss function to include class weights
    # Loss + optimizer
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    #loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(classifier.parameters(), 
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay, # L2 regularization to prevent overfitting
                           ) 

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        min_lr=1e-6
    )

    # To store metrics for each epoch
    metrics = {
        "timestamp": timestamp,
        "seed": seed,
        "model_path": model_file,
        "pipeline": pipeline,
        "device": device_name,
        "gpu_resident": args.gpu_resident,
        "num_workers": num_workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "dropout": args.dropout,
        "hidden_dim":args.hidden_dim,
        "proj_dim":args.proj_dim,
        "experiment": experiment,   
        "results": []
    }

    
    # --- Early Stopping Setup ---
    patience = 10                # How many epochs to wait for improvement
    best_val_loss = float('inf') # Track the lowest loss seen so far
    counter = 0                  # Counts epochs without improvement
    # Global start time for training
    global_start_time = time.perf_counter()
    # -------------------------------
    # Training loop
    # -------------------------------
    # Open the results file in write mode
    with open(results_file, "w") as f:
        print("Current working directory:", os.getcwd())
        f.write(f"-----Training run started at {timestamp}-----\n")
        exp = metrics["experiment"]
        f.write("Experiment Configuration:\n")
        for k, v in exp.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        f.write(f"Learning Rate: {args.learning_rate}, Weight Decay: {args.weight_decay}, Batch Size: {args.batch_size}, Num Epochs: {args.num_epochs}, Pipeline: {pipeline}, Device: {device_name}, Seed: {seed}\n\n")
        f.write(f"GPU Resident Mode: {args.gpu_resident}\n\n")

        for epoch in range(args.num_epochs):
            # Record start time for this epoch
            # Forces all CUDA operations to complete before starting the timer
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            epoch_start_time = time.perf_counter()

            train_loss, train_acc = train_epoch(classifier, train_loader, optimizer, loss_fn, prepare_batch, debug=args.debug)
            val_loss, val_acc = evaluate(classifier, dev_loader, loss_fn, prepare_batch, debug=args.debug)
            gap = val_loss - train_loss

            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} finished. Current Learning Rate: {current_lr:.2e}")
            f.write(f"Epoch {epoch+1} finished. Current Learning Rate: {current_lr:.2e}\n\n")

            # Record end time for this epoch
            # Forces all CUDA operations to complete before starting the timer
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            epoch_end_time = time.perf_counter() - epoch_start_time

            print(f"Epoch {epoch+1}/{args.num_epochs}")
            print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
            print(f"  Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")
            print(f"  Gen gap (val - train loss): {gap:.4f}")
            print(f"  Epoch time: {epoch_end_time:.2f} seconds\n")

            # Log metrics
            f.write(f"Epoch {epoch+1}/{args.num_epochs}\n")
            f.write(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}\n")
            f.write(f"  Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}\n")
            f.write(f"  Gen gap (val - train loss): {gap:.4f}\n\n")
            f.write(f"  Epoch time: {epoch_end_time:.2f} seconds\n\n")

            # Save metrics for this epoch
            metrics["results"].append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc, 
                "learning_rate": current_lr,
                "generalization_gap": gap,
                "epoch_time": epoch_end_time
            })

            # --- Early Stopping Logic ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # Move your model saving code here so you only save the "Best" version
                torch.save(classifier.state_dict(), model_file)
                print(f"  New best Val Loss: {best_val_loss:.4f}. Model saved.")
                f.write(f"  New best Val Loss: {best_val_loss:.4f}. Model saved.\n\n")
            else:
                counter += 1
                print(f"  No improvement in Val Loss. Early stopping counter: {counter}/{patience}")
                f.write(f"  No improvement in Val Loss. Early stopping counter: {counter}/{patience}\n\n")

                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. Training stopped.")
                    break # This exits the epoch loop early
            # ----------------------------
        # Record total training time
        # Forces all CUDA operations to complete before starting the timer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_training_time = time.perf_counter() - global_start_time
        f.write("-----Training completed-----\n")
        f.write(f"Total training time: {total_training_time:.2f} seconds\n")

    epochs = [r["epoch"] for r in metrics["results"]]

    train_losses = [r["train_loss"] for r in metrics["results"]]
    val_losses   = [r["val_loss"] for r in metrics["results"]]

    train_accs = [r["train_accuracy"] for r in metrics["results"]]
    val_accs   = [r["val_accuracy"] for r in metrics["results"]]

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()

    loss_plot_path = os.path.join(base_dir, f"loss_curve_seed{seed}_{timestamp}.png")
    plt.savefig(loss_plot_path)
    plt.close()


    # Accuracy curve
    plt.figure()
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid()

    acc_plot_path = os.path.join(base_dir, f"acc_curve_seed{seed}_{timestamp}.png")
    plt.savefig(acc_plot_path)
    plt.close()
    
    # Save all metrics to a JSON file
    # Open the results file in write mode
    with open(metrics_file, "w") as jf:
        json.dump(metrics, jf, indent=4)

    print(f"\n Training run completed. Logs saved to: {results_file}")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Best model saved to: {model_file}")
# Running args: python src/train.py --seed 42 --gpu_resident --num_workers 0 --fusion mlb --dataset meld --temporal_encoder conformer
# python src/train.py --seed 42 --gpu_resident --num_workers 0 --fusion concat --dataset meld --temporal_encoder conformer
# Debug ON: python src/train.py --seed 42 --gpu_resident --num_workers 0 --fusion mlb --debug --dataset iemocap --temporal_encoder conformer
# Debug ON: python src/train.py --seed 42 --gpu_resident --num_workers 0 --fusion concat --debug
