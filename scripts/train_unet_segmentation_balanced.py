#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_unet_segmentation import (
    UNet,
    SegmentationDataset,
    FocalLoss,
    compute_class_weights,
    train_one_epoch,
    evaluate,
    save_metrics_csv,
    set_seed,
    NUM_CLASSES,
    IGNORE_INDEX,
    ID_TO_LABEL,
)


def compute_sample_weights(dataset):
    labels = [row["label"] for row in dataset.rows]
    counts = Counter(labels)

    print("Training image-label counts:", dict(counts))

    weights = []

    for row in dataset.rows:
        label = row["label"]
        weights.append(1.0 / counts[label])

    return torch.DoubleTensor(weights)


def main():
    parser = argparse.ArgumentParser(
        description="Train U-Net segmentation model with class-balanced image sampling."
    )

    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "weighted_cross_entropy", "focal"],
        default="weighted_cross_entropy",
    )
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

    sample_weights = compute_sample_weights(train_dataset)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
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
        criterion = torch.nn.CrossEntropyLoss(
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
        "balanced_sampler": True,
        "sampler": "WeightedRandomSampler by image-level label",
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
