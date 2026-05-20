"""
Chest X-Ray dataset loader for pneumonia classification.

Wraps the Kermany et al. (2018) dataset structure:
    data_dir/
        train/ NORMAL/ ... PNEUMONIA/ ...
        val/   NORMAL/ ... PNEUMONIA/ ...
        test/  NORMAL/ ... PNEUMONIA/ ...
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple


CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def get_default_transforms(
    image_size: int = 224,
    is_training: bool = True,
    horizontal_flip: bool = False,
) -> transforms.Compose:
    """
    Build standard preprocessing transforms.

    Args:
        image_size: Target square image size
        is_training: Apply training augmentations if True
        horizontal_flip: Whether to enable horizontal flip
            (False for chest X-rays — heart is left-sided)

    Returns:
        torchvision Compose pipeline
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if is_training:
        ops = [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
        if horizontal_flip:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops.extend([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        ops = [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]

    return transforms.Compose(ops)


class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for the Kermany et al. chest X-ray dataset.

    Args:
        data_dir: Root directory containing train/val/test splits
        split: One of 'train', 'val', 'test'
        transform: Optional torchvision transforms
        image_extensions: File extensions to scan for
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        image_extensions: Tuple[str, ...] = (".jpeg", ".jpg", ".png"),
    ):
        valid_splits = {"train", "val", "test"}
        if split not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}")

        self.data_dir = Path(data_dir) / split
        self.split = split
        self.transform = transform
        self.image_extensions = image_extensions

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.data_dir}"
            )

        self.samples: List[Tuple[Path, int]] = []
        self._scan_files()

    def _scan_files(self):
        """Scan class subdirectories for image files."""
        for class_name in CLASS_NAMES:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue

            class_idx = CLASS_TO_IDX[class_name]
            for ext in self.image_extensions:
                for filepath in class_dir.glob(f"*{ext}"):
                    self.samples.append((filepath, class_idx))

        if not self.samples:
            raise RuntimeError(
                f"No images found in {self.data_dir}. "
                f"Expected subdirectories: {CLASS_NAMES}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem
