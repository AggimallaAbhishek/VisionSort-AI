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
    parser.add_argument("--epochs", type=int, default=24, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--lr-head", type=float, default=3e-4, help="Learning rate while head-only training.")
    parser.add_argument("--lr-finetune", type=float, default=7e-5, help="Learning rate after unfreezing backbone.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Cross-entropy label smoothing.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout for classifier head.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience.")

    parser.add_argument(
        "--freeze-features",
        action="store_true",
        help="Start with frozen MobileNetV2 features.",
    )
    parser.add_argument(
        "--unfreeze-epoch",
        type=int,
        default=5,
        help="Epoch after which frozen features are unfrozen.",
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


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.04)], p=0.75
            ),
            transforms.RandomRotation(degrees=8),
            transforms.RandomGrayscale(p=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, val_transform


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    image_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, List[int]]:
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Missing train/val folders in {data_root}. Expected {train_dir} and {val_dir}."
        )

    train_transform, val_transform = build_transforms(image_size=image_size)
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

    class_counts = [int(sum(1 for t in train_dataset.targets if t == idx)) for idx in range(len(CLASS_NAMES))]
    return train_loader, val_loader, class_counts


def compute_class_weights(class_counts: List[int]) -> torch.Tensor:
    counts = np.asarray(class_counts, dtype=np.float32)
    if np.any(counts <= 0):
        raise ValueError(f"Class count cannot be zero. Counts={class_counts}")

    total = float(counts.sum())
    weights = total / (len(class_counts) * counts)
    weights /= float(weights.mean())
    return torch.tensor(weights, dtype=torch.float32)


def build_model(num_classes: int, dropout: float) -> nn.Module:
    try:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        print("Loaded pretrained MobileNetV2 base weights.")
    except Exception as exc:
        print(f"Could not load pretrained MobileNetV2 weights ({exc}). Using random init.")
        model = mobilenet_v2(weights=None)

    if isinstance(model.classifier[0], nn.Dropout):
        model.classifier[0].p = dropout

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def set_feature_freeze(model: nn.Module, freeze: bool) -> None:
    for param in model.features.parameters():
        param.requires_grad = not freeze


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found.")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip: float | None = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    use_amp = bool(scaler is not None and device.type == "cuda")

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip and grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == labels).sum().item())
        total_samples += int(batch_size)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_per_class(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = [0 for _ in CLASS_NAMES]
    total = [0 for _ in CLASS_NAMES]
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        for idx in range(len(CLASS_NAMES)):
            mask = labels == idx
            c = int(mask.sum().item())
            if c == 0:
                continue
            total[idx] += c
            correct[idx] += int((preds[mask] == labels[mask]).sum().item())

    return {
        name: (correct[idx] / total[idx] if total[idx] else 0.0)
        for idx, name in enumerate(CLASS_NAMES)
    }


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    best_val_acc: float,
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    checkpoint = {
        "state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "arch": "mobilenet_v2",
        "image_size": args.image_size,
        "best_val_accuracy": best_val_acc,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "args": safe_args,
    }
    torch.save(checkpoint, output_path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.patience <= 0:
        raise ValueError("--patience must be > 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Classes: {CLASS_NAMES}")

    train_loader, val_loader, class_counts = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Train class counts: {dict(zip(CLASS_NAMES, class_counts))}")

    model = build_model(num_classes=len(CLASS_NAMES), dropout=args.dropout).to(device)
    features_frozen = False
    if args.freeze_features:
        set_feature_freeze(model, True)
        features_frozen = True
        print(f"Feature extractor frozen until epoch {args.unfreeze_epoch}.")

    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(class_counts).to(device),
        label_smoothing=max(0.0, min(float(args.label_smoothing), 0.2)),
    )
    optimizer = build_optimizer(
        model,
        lr=args.lr_head if features_frozen else args.lr_finetune,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_state: Dict[str, torch.Tensor] | None = None
    best_val_acc = 0.0
    history: List[Dict[str, float]] = []
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        if features_frozen and epoch > args.unfreeze_epoch:
            set_feature_freeze(model, False)
            features_frozen = False
            optimizer = build_optimizer(model, lr=args.lr_finetune, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
            )
            print("Unfroze backbone features and switched to fine-tuning LR.")

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=args.grad_clip,
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        val_per_class = evaluate_per_class(model, val_loader, device)
        scheduler.step(val_acc)
        current_lr = float(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"lr={current_lr:.7f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        print("  val_per_class:", {k: round(v, 3) for k, v in val_per_class.items()})

        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                **{f"val_acc_{k}": v for k, v in val_per_class.items()},
            }
        )

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    save_checkpoint(args.output, model, best_val_acc, args)
    print(f"Saved best model to: {args.output}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "history": history,
                "best_val_acc": best_val_acc,
                "class_names": CLASS_NAMES,
            },
            indent=2,
        )
    )
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
