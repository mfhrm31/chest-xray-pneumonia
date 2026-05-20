"""
ResNet50 classifier for chest X-ray pneumonia detection.

Wraps torchvision's ResNet50 with a custom classification head
and supports a two-stage training schedule (frozen backbone first,
then full fine-tuning).
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ResNet50Classifier(nn.Module):
    """
    ResNet50-based classifier with custom head for medical imaging.

    Replaces the original 1000-class ImageNet head with a smaller
    classifier suited to binary or small-multiclass medical tasks.
    Supports freezing the backbone for an initial training warmup
    where only the new head is trained.

    Args:
        num_classes: Number of output classes (default: 2 for binary)
        pretrained: Load ImageNet pretrained weights (default: True)
        dropout: Dropout rate in the classification head (default: 0.3)
        freeze_backbone: Freeze ResNet50 backbone parameters
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all parameters except the new classification head."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (batch_size, 3, H, W)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)

    def count_parameters(self) -> dict:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }


def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
    device: Optional[str] = None,
) -> ResNet50Classifier:
    """
    Factory function to build a ResNet50 classifier.

    Args:
        num_classes: Number of output classes
        pretrained: Load ImageNet weights
        dropout: Head dropout rate
        freeze_backbone: Initial backbone freeze
        device: 'cuda' or 'cpu' (auto-detected if None)

    Returns:
        Configured ResNet50Classifier on the chosen device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNet50Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
    return model.to(device)


if __name__ == "__main__":
    # Sanity check
    model = ResNet50Classifier(num_classes=2, freeze_backbone=True)
    counts = model.count_parameters()
    print("ResNet50 Classifier — Parameter Counts")
    print(f"  Total:     {counts['total']:>12,}")
    print(f"  Trainable: {counts['trainable']:>12,}")
    print(f"  Frozen:    {counts['frozen']:>12,}")

    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    model.unfreeze_backbone()
    counts_unfrozen = model.count_parameters()
    print(f"\nAfter unfreezing:")
    print(f"  Trainable: {counts_unfrozen['trainable']:>12,}")
