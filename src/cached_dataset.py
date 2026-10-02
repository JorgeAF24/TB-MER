"""
Cached MELD Dataset
===================

Optimization Technique:
- Cleaner Dataset (tensor-only loading)

Purpose:
- Load precomputed CLIP text and image features from disk
- Serve them directly to the training loop
- Eliminate all raw data processing during training
"""

import os
import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    """
    Generic cached feature dataset.

    Supports:
        MELD
        IEMOCAP

    Loads:
        text.pt
        image.pt
        labels.pt
        label_names.pt
        sample_ids.pt
    """

    def __init__(self, feature_dir: str):
        """
        Args:
            feature_dir (str):
                Path to cached feature directory, e.g.
                'data/features/meld_train'
        """
        self.feature_dir = feature_dir
        self.dataset_name = os.path.basename(feature_dir).split("_")[0].upper()  # e.g., "MELD" or "IEMOCAP"
        # Paths to cached tensor files
        text_path = os.path.join(feature_dir, "text.pt")
        image_path = os.path.join(feature_dir, "image.pt")
        label_path = os.path.join(feature_dir, "labels.pt")
        label_names = os.path.join(feature_dir, "label_names.pt")
        sample_ids_path = os.path.join(feature_dir, "sample_ids.pt")

        # Use assertions as a "safety check" mechanism
        # To verify that all required files exist
        assert os.path.exists(text_path), f"Missing file: {text_path}"
        assert os.path.exists(image_path), f"Missing file: {image_path}"
        assert os.path.exists(label_path), f"Missing file: {label_path}"
        assert os.path.exists(label_names), f"Missing file: {label_names}"
        if os.path.exists(sample_ids_path):
            assert os.path.exists(sample_ids_path), f"Missing file: {sample_ids_path}"

        # Load all tensors into memory once
        self.text_features = torch.load(text_path)   # [N, 512]
        self.image_features = torch.load(image_path) # [N, 512]
        self.labels = torch.load(label_path)         # [N]
        self.label_names = torch.load(label_names)  # [N]
        if os.path.exists(sample_ids_path):
            self.sample_ids = torch.load(sample_ids_path) # [N]
        else:
            self.sample_ids = None

        # Basic consistency checks
        # Use assertions as a "safety check" mechanism
        # Assert that the number of samples in each tensor is the same
        assert len(self.image_features) == self.text_features.size(0), \
            "[ERROR] mismatch between text and image samples"

        assert self.text_features.size(0) == self.labels.size(0), \
            "[ERROR] mismatch between text and labels"
        
        if self.sample_ids is not None:
            assert self.text_features.size(0) == len(self.sample_ids)
            assert len(self.sample_ids) == len(self.labels)

    def to(self, device):
        """
        Move entire dataset tensors to a device (e.g., GPU).
        Used for GPU-resident training mode.
        """
        self.text_features = self.text_features.to(device)
        self.image_features = [seq.to(device) for seq in self.image_features]
        self.labels = self.labels.to(device)
        self.device = device
        return self

    # Return the number of samples in the dataset
    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int):
        """
        Return one cached sample.

        Returns:
            text_feat  : torch.Tensor [512]
            image_feat : torch.Tensor [512]
            label      : torch.Tensor []
            sample_id  : torch.Tensor []
        """
        if self.sample_ids is not None:
            return (
                self.text_features[idx],
                self.image_features[idx],
                self.labels[idx],
                self.sample_ids[idx],
            )
        else:
            return (
                self.text_features[idx],
                self.image_features[idx],
                self.labels[idx],
            )
if __name__ == "__main__":

    dataset = CachedFeatureDataset(
        "data/meld_features_4Hz/meld_train"
    )

    print(type(dataset))
    print(len(dataset))

    sample = dataset[0]

    print(sample[0].shape)
    print(sample[1].shape)
    print(sample[2])

    print(dataset.label_names)
    print(dataset.dataset_name)