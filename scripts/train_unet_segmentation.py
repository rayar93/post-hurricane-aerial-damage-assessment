#!/usr/bin/env python3

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


NUM_CLASSES = 5
IGNORE_INDEX = 255

ID_TO_LABEL = {
    0: "background",
    1: "no damage",
    2: "minor damage",
    3: "major damage",
    4: "destroyed",
}


def parse_bool(value):
    return str(value).lower() in {"true", "1", "yes"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SegmentationDataset(Dataset):
    def __init__(self, manifest_csv, split):
        self.manifest_csv = Path(manifest_csv)
        self.split = split

        with self.manifest_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.rows = [row for row in rows if row["split"] == split]

        if split in {"val", "test"}:
            self.rows = [row for row in self.rows if not parse_bool(row["is_augmented"])]

        if not self.rows:
            raise ValueError(f"No rows found for split={split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        image = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")

        image_array = np.asarray(image).astype(np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))

        mask_array = np.asarray(mask).astype(np.int64)

        image_tensor = torch.from_numpy(image_array)
        mask_tensor = torch.from_numpy(mask_array)

        return image_tensor, mask_tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, base_channels=32):
        super().__init__()

        self.inc = DoubleConv(in_channels, base_channels)

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels, base_channels * 2),
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 2, base_channels * 4),
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 4, base_channels * 8),
        )

        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 8, base_channels * 16),
        )

        self.up1 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up4 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_channels * 2, base_channels)

        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)

        return self.outc(x)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        valid = targets != self.ignore_index

        if valid.sum() == 0:
            return ce.mean() * 0.0

        ce = ce[valid]
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce

        return loss.mean()


def compute_class_weights(manifest_csv, device):
    with Path(manifest_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    train_rows = [row for row in rows if row["split"] == "train"]

    counts = np.zeros(NUM_CLASSES, dtype=np.float64)

    for row in train_rows:
        mask = Image.open(row["mask_path"]).convert("L")
        arr = np.asarray(mask).astype(np.int64)
        valid = arr != IGNORE_INDEX
        bincount = np.bincount(arr[valid].reshape(-1), minlength=NUM_CLASSES)
        counts += bincount[:NUM_CLASSES]

    freq = counts / max(counts.sum(), 1.0)

    weights = 1.0 / np.log(1.02 + freq)
    weights = weights / weights.mean()

    weights = np.clip(weights, 0.25, 5.0)

    print("Pixel counts by class:", counts.tolist())
    print("Auto class weights:", weights.tolist())

    return torch.tensor(weights, dtype=torch.float32, device=device)


def confusion_matrix_from_batch(preds, targets, num_classes=NUM_CLASSES):
    valid = targets != IGNORE_INDEX
    preds = preds[valid]
    targets = targets[valid]

    if targets.numel() == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    indices = num_classes * targets.cpu().numpy().astype(np.int64) + preds.cpu().numpy().astype(np.int64)
    matrix = np.bincount(indices, minlength=num_classes ** 2)
    return matrix.reshape(num_classes, num_classes)


def metrics_from_confusion(confusion):
    total = confusion.sum()
    correct = np.diag(confusion).sum()

    pixel_accuracy = correct / total if total > 0 else 0.0

    ious = {}

    for class_id in range(NUM_CLASSES):
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp

        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else float("nan")
        ious[class_id] = iou

    foreground_ious = [
        ious[class_id]
        for class_id in [1, 2, 3, 4]
        if not np.isnan(ious[class_id])
    ]

    mean_iou_foreground = float(np.mean(foreground_ious)) if foreground_ious else 0.0

    all_ious = [value for value in ious.values() if not np.isnan(value)]
    mean_iou_all = float(np.mean(all_ious)) if all_ious else 0.0

    return {
        "pixel_accuracy": float(pixel_accuracy),
        "mean_iou_all": mean_iou_all,
        "mean_iou_foreground": mean_iou_foreground,
        "iou_background": float(ious[0]) if not np.isnan(ious[0]) else 0.0,
        "iou_no_damage": float(ious[1]) if not np.isnan(ious[1]) else 0.0,
        "iou_minor_damage": float(ious[2]) if not np.isnan(ious[2]) else 0.0,
        "iou_major_damage": float(ious[3]) if not np.isnan(ious[3]) else 0.0,
        "iou_destroyed": float(ious[4]) if not np.isnan(ious[4]) else 0.0,
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * images.size(0)
        confusion += confusion_matrix_from_batch(preds, masks)

    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def save_metrics_csv(metrics_rows, output_csv):
    if not metrics_rows:
        return

    fieldnames = list(metrics_rows[0].keys())

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)


def main():
    parser = argparse.ArgumentParser(description="Train a baseline U-Net segmentation model.")

    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--loss", choices=["cross_entropy", "weighted_cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    manifest_csv = args.dataset_root / "metadata" / "manifest.csv"

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    train_dataset = SegmentationDataset(manifest_csv, split="train")
    val_dataset = SegmentationDataset(manifest_csv, split="val")
    test_dataset = SegmentationDataset(manifest_csv, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_channels=args.base_channels,
    ).to(device)

    class_weights = None

    if args.loss in {"weighted_cross_entropy", "focal"}:
        class_weights = compute_class_weights(manifest_csv, device=device)

    if args.loss in {"cross_entropy", "weighted_cross_entropy"}:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=IGNORE_INDEX,
        )
    elif args.loss == "focal":
        criterion = FocalLoss(
            weight=class_weights,
            gamma=2.0,
            ignore_index=IGNORE_INDEX,
        )
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    best_val_miou = -1.0
    metrics_rows = []

    config = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "base_channels": args.base_channels,
        "loss": args.loss,
        "seed": args.seed,
        "device": str(device),
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "id_to_label": ID_TO_LABEL,
    }

    with (args.output_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_mean_iou_all": val_metrics["mean_iou_all"],
            "val_mean_iou_foreground": val_metrics["mean_iou_foreground"],
            "val_iou_background": val_metrics["iou_background"],
            "val_iou_no_damage": val_metrics["iou_no_damage"],
            "val_iou_minor_damage": val_metrics["iou_minor_damage"],
            "val_iou_major_damage": val_metrics["iou_major_damage"],
            "val_iou_destroyed": val_metrics["iou_destroyed"],
        }

        metrics_rows.append(row)
        save_metrics_csv(metrics_rows, args.output_dir / "metrics.csv")

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_mIoU_fg={val_metrics['mean_iou_foreground']:.4f} | "
            f"val_acc={val_metrics['pixel_accuracy']:.4f}"
        )

        if val_metrics["mean_iou_foreground"] > best_val_miou:
            best_val_miou = val_metrics["mean_iou_foreground"]

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_mean_iou_foreground": best_val_miou,
                "config": config,
            }

            torch.save(checkpoint, args.output_dir / "best_model.pt")

    print()
    print("Loading best model for final test evaluation...")

    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print()
    print("Final test metrics")
    print("------------------")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    print()
    print(f"Saved best model to: {args.output_dir / 'best_model.pt'}")
    print(f"Saved training metrics to: {args.output_dir / 'metrics.csv'}")
    print(f"Saved test metrics to: {args.output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
