"""
Offline CLIP Feature Extraction Script
====================================

Optimization Technique:
- CLIP feature caching (offline feature extraction)

Purpose:
- Extract frozen CLIP text and image embeddings ONCE per dataset split
- Save them as tensors to disk
- Remove CLIP, tokenization, and image decoding from the training loop

This script MUST be run before training.
"""

from typing import List, Tuple
import os
import argparse
from PIL import Image
# LabelEncoder converts categorical labels (strings) to numeric values (integers)
from sklearn.preprocessing import LabelEncoder
import torch
import clip
from tqdm import tqdm

from meld_dataset import MELDDataset
from iemocap_dataset import IEMOCAPDataset


class CLIPFeatureExtractor:
    """
    Frozen CLIP feature extractor.

    CLIP model:
        - ViT-L/14
        - Frozen weights
        - Output embedding dim = 768

    This class is intentionally minimal:
    - No gradients
    - No training logic
    - Only deterministic feature extraction
    """

    def __init__(self, device: str) -> None:
        self.device = device

        # Load the CLIP model (ViT-L/14) and its corresponding image preprocessing pipeline.
        # clip.load returns a tuple containing (model, preprocess function)
        # Use tuple unpacking to assign these to self.model and self.preprocess
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)
        self.model.eval()

        # Explicitly freeze parameters (clarity + safety)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    # The arrow tells what data type this function returns
    # This function takes a list of strings and returns a torch.Tensor
    def encode_text_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of text strings.

        Args:
            texts: list[str] of length B

        Returns:
            text_features: torch.Tensor [B, 768] (on CPU)
        """
        truncated_list = []
        for t in texts:
            # Tokenize one by one with truncation
            try:
                # Check if tokenization works without truncation
                tokens = clip.tokenize(t) #Output is tensor of shape (1, N) where N is number of tokens
                truncated_list.append(t)
            # If tokenization fails due to text being too long
            # Text is truncated word-byword until it fits within the token limit 
            except RuntimeError:
                # Handle text longer than context limit (77 tokens)
                print(f"[Warning] Text too long, truncating: {t[:100]}...")
                # Truncate the text to fit safely under the limit
                words = t.split()
                truncated_t = None
                for n in range(len(words), 0, -1):
                    candidate = " ".join(words[:n])
                    try:
                        tokens = clip.tokenize(candidate)
                        truncated_t = candidate
                        break  # Successfully tokenized, exit loop
                    except RuntimeError:
                        continue

                if truncated_t is None:
                    # If all else fails, just take the first 200 words
                    truncated_t = " ".join(words[:200])  # limit ~70 words ≈ 77 tokens
                truncated_list.append(truncated_t)

        # Tokenize (CPU → GPU)
        tokens = clip.tokenize(truncated_list).to(self.device)

        # CLIP text encoder (GPU)
        # Output is tensor of shape (B, 768)
        features = self.model.encode_text(tokens)

        # L2 normalize (as in CLIP usage)
        # features = features / features.norm(dim=-1, keepdim=True)

        # Move to CPU for caching
        return features.cpu().float()

    @torch.no_grad()
     # The arrow tells what data type this function returns
     # This function takes a list of NumPy arrays or None and returns a torch.Tensor
    def encode_image_batch(self, images: List[List]) -> List[torch.Tensor]:
        """
        Encode a batch of images sequences.

        Args:
            images: list of sequences, length B
                    each element = list of frames (length T)

        Returns:
            image_features: List[torch.Tensor] [B, T, 768] (on CPU)
            where len(images) = B (batch size) and len(images[i]) = T (number of frames per sample)
        """
        #Define a buffer to store the batch features as they are loaded. 
        # Buffer: temporary storage area
        batch_features = []

        # Process each sequence in the batch (one sequence = one sample = one video clip)
        # seq = [frame1, frame2, ..., frameT]
        for seq in images:
            #Define a buffer to store the processed images as they are loaded. 
            # Buffer: temporary storage area
            processed_images = []
            # Process each frame in the sequence
            # img = [T, 3, 224, 224] (after processing)
            for img in seq:
                if img is None:
                    # Placeholder for missing frames
                    processed_images.append(torch.zeros(3, 224, 224))
                else:
                    # Convert NumPy array → PIL → CLIP preprocess

                    # NOTE:
                    # We already resized to 224x224 in dataset.py.
                    # CLIP preprocess will normalize + convert to tensor.
                    # .fromarray converts NumPy array to PIL Image Object
                    img_pil = Image.fromarray(img)
                    img_tensor = self.preprocess(img_pil)
                    processed_images.append(img_tensor)

            if len(processed_images) == 0:
                print(f"[WARNING] Empty sequence → inserting dummy frame")
                processed_images.append(torch.zeros(3, 224, 224))
                # skip completely broken sample
            
            # Stack into batch tensor: [T, 3, 224, 224]
            image_tensor = torch.stack(processed_images).to(self.device)

            # CLIP image encoder (GPU)
            # Output is tensor of shape (T, 768)
            features = self.model.encode_image(image_tensor)

            # Move to CPU for caching and append to batch features
            batch_features.append(features.cpu().float())

        # L2 normalize
        # features = features / features.norm(dim=-1, keepdim=True)
        
        # Stack batch -> [B, T, 768]
        return batch_features 

# The arrow tells what data type this function returns
# This function takes several arguments and returns None
def extract_split_features(
    split: str,
    raw_root: str,
    frames_root: str,
    output_dir: str,
    batch_size: int,
    device: str,
    dataset_name: str
) -> None:
    """
    Extract and save CLIP features for a single dataset split.

    Saved tensors:
        - text.pt   : [N, 512]
        - image.pt  : [N, 512]
        - labels.pt : [N]

    Args:
        split: "train", "dev", or "test"
    """

    print(f"\n[INFO] Extracting CLIP features for split: {split}")

    # Load the specified dataset split
    if dataset_name == "meld":
        dataset = MELDDataset(root_dir=raw_root, frames_dir=frames_root, split=split)
    elif dataset_name == "iemocap":
        dataset = IEMOCAPDataset(metadata_path=raw_root, frames_dir=frames_root, split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Get the list of unique emotion labels
    label_names = dataset.emotions
    extractor = CLIPFeatureExtractor(device=device)

    # Buffers to hold all features and labels
    # Buffer: temporary storage area
    all_text_feats = []
    all_image_feats = []
    all_labels = []
    all_sample_ids = []

    # Batch buffers
    # These hold samples until we have a full batch to process
    texts_batch = []
    images_batch = []

    for idx in tqdm(range(len(dataset))):
        # Get the raw text, label, and image for this sample index
        text, label, image = dataset[idx]

        # Extract metadata 
        meta = dataset.data.iloc[idx]

        if dataset_name == "meld":
            sample_id = {
                "index": int(idx),  # dataset index (for alignment debugging)
                "dialogue_id": int(meta["Dialogue_ID"]),
                "utterance_id": int(meta["Utterance_ID"])
            }
        else:
            sample_id = {
                "index": int(idx),
                "dialogue_id": meta["dialogue_id"],
                "utterance_id": meta["utterance_id"]
            }


        # Append to batch buffers
        texts_batch.append(text)
        images_batch.append(image)
        all_labels.append(label)
        all_sample_ids.append(sample_id)

        # When batch is full OR last sample
        if len(texts_batch) == batch_size or idx == len(dataset) - 1:
            # Encode text + image in batches
            text_feats = extractor.encode_text_batch(texts_batch)
            image_feats = extractor.encode_image_batch(images_batch)

            assert len(image_feats) == text_feats.size(0), \
                "[ERROR] Mismatch between image and text samples inside batch"
            
            # Append to all-features buffers
            all_text_feats.append(text_feats)
            all_image_feats.extend(image_feats)

            # Reset batch buffers
            texts_batch = []
            images_batch = []

    # Concatenate all image and text features as rows (dim=0) into a tensor matrix 
    # Where each row corresponds to an image or text for the whole dataset split
    # And each column corresponds to a feature dimension
    text_features = torch.cat(all_text_feats, dim=0)    # [N, 768]
    image_features = all_image_feats # List of tensors

    # =========================
    # SANITY CHECKS (ADD HERE)
    # =========================

    print(f"[DEBUG] number of samples = {len(image_features)}")
    print(f"[DEBUG] example sequence shape = {image_features[0].shape}")

    assert len(image_features) == text_features.size(0), \
        "[ERROR] Mismatch between image and text samples"

    assert len(all_sample_ids) == text_features.size(0), \
    "[ERROR] Mismatch between sample IDs and features"

    # Check each sequence
    for seq in image_features:
        assert seq.dim() == 2, "[ERROR] Each sequence must be [T_i, D]"
        assert seq.size(1) == 768, "[ERROR] Feature dim must be 768"
        assert not torch.isnan(seq).any(), "[ERROR] NaNs in sequence"

    print("[DEBUG] Sanity checks passed ✔")
    
    # Encode labels as integers (consistent with training)
    le = LabelEncoder()
    labels_encoded = torch.tensor(le.fit_transform(all_labels),dtype=torch.long)

    # Save to disk
    # Use os.path.join to create the output directory path
    # Use a f-string to pass the split variable into the file name
    split_out = os.path.join(output_dir, f"{dataset_name}_{split}")
    # Create the output directory if it doesn't exist
    os.makedirs(split_out, exist_ok=True)

    # Using torch.save to save CLIP tensors embeddings living in memory (RAM) 
    # and turn it into a permanent file on your disk
    # torch.save(object, filepath)
    # It adds the .pt extension automatically
    torch.save(text_features, os.path.join(split_out, "text.pt"))
    torch.save(image_features, os.path.join(split_out, "image.pt"))
    torch.save(labels_encoded, os.path.join(split_out, "labels.pt"))
    torch.save(label_names, os.path.join(split_out, "label_names.pt"))
    torch.save(all_sample_ids, os.path.join(split_out, "sample_ids.pt"))
    
    print(f"[INFO] Saved features to: {split_out}")
    print(f"       text  : {text_features.shape}")
    print(f"       image : list of {len(image_features)} sequences")
    print(f"       labels: {labels_encoded.shape}")
    print(f"       label_names: {label_names}")


def main():
    # Uses argparse library to handle code execution from terminal/command line
    # parser lines defines what arguments are allowed to change when running the script
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True,
                        choices=["train", "dev", "test"])
    parser.add_argument("--raw_root", type=str, default="data/MELD.Raw")
    parser.add_argument("--frames_root", type=str, default="data/MELD.Frames_4Hz")
    # Specifies where to save the extracted features
    parser.add_argument("--out_dir", type=str, default="data/meld_features_4Hz")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset", type=str, required=True, choices=["meld", "iemocap"])
    
    # args gather all the arguments together
    args = parser.parse_args()

    # Call the function to extract features for the specified split
    extract_split_features(
        split=args.split,
        raw_root=args.raw_root,
        frames_root=args.frames_root,
        output_dir=args.out_dir,
        batch_size=args.batch_size,
        device=args.device,
        dataset_name=args.dataset
    )


if __name__ == "__main__":
    main()
# Run Program: python src/feature_extractor.py --split train --dataset meld
# python src/feature_extractor.py --split train --dataset iemocap --frames_root data/IEMOCAP.Frames --raw_root data/IEMOCAP/metadata --out_dir data/iemocap_features