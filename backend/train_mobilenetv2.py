#!/usr/bin/env python3
"""Train a MobileNetV2 classifier for VisionSort AI quality labels.

Expected dataset structure:

training_data/
  train/
    good/
    blurry/
    dark/
    overexposed/
    duplicates/
  val/
    good/
    blurry/
    dark/
    overexposed/
    duplicates/
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

CLASS_NAMES: List[str] = ["good", "blurry", "dark", "overexposed", "duplicates"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "training_data"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "model" / "photo_model.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for VisionSort AI.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output .pth path.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--freeze-features",
        action="store_true",
        help="Freeze MobileNetV2 feature extractor and train classifier head only.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remap_dataset_to_expected_classes(dataset: datasets.ImageFolder) -> None:
    expected = set(CLASS_NAMES)
    observed = set(dataset.class_to_idx.keys())
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)

    if missing or unexpected:
        raise ValueError(
            "Dataset class folder mismatch. "
            f"Missing={missing or 'none'}, Unexpected={unexpected or 'none'}. "
            f"Expected exactly: {CLASS_NAMES}"
        )

    desired_mapping = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    source_mapping = dataset.class_to_idx
    idx_map = {source_mapping[name]: desired_mapping[name] for name in CLASS_NAMES}

    dataset.samples = [(path, idx_map[label]) for path, label in dataset.samples]
    dataset.targets = [idx_map[label] for label in dataset.targets]
    dataset.imgs = list(dataset.samples)
    dataset.class_to_idx = desired_mapping
    dataset.classes = list(CLASS_NAMES)


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    image_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Missing train/val folders in {data_root}. Expected {train_dir} and {val_dir}."
        )

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    remap_dataset_to_expected_classes(train_dataset)
    remap_dataset_to_expected_classes(val_dataset)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Train/val dataset cannot be empty.")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def build_model(num_classes: int) -> nn.Module:
    try:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        print("Loaded pretrained MobileNetV2 base weights.")
    except Exception as exc:
        print(f"Could not load pretrained MobileNetV2 weights ({exc}). Using random init.")
        model = mobilenet_v2(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == labels).sum().item())
        total_samples += int(batch_size)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    best_val_acc: float,
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "arch": "mobilenet_v2",
        "image_size": args.image_size,
        "best_val_accuracy": best_val_acc,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint, output_path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Classes: {CLASS_NAMES}")

    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    if args.freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
        print("Feature extractor frozen. Training classifier head only.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state: Dict[str, torch.Tensor] | None = None
    best_val_acc = 0.0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    save_checkpoint(args.output, model, best_val_acc, args)
    print(f"Saved best model to: {args.output}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({"history": history, "best_val_acc": best_val_acc}, indent=2))
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
