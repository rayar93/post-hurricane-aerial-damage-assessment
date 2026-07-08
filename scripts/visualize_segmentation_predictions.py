#!/usr/bin/env python3

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import torch

from train_unet_segmentation import UNet, NUM_CLASSES, ID_TO_LABEL, parse_bool


COLOR_MAP = {
    0: (0, 0, 0),          # background
    1: (0, 180, 0),        # no damage
    2: (255, 210, 0),      # minor damage
    3: (255, 120, 0),      # major damage
    4: (220, 0, 0),        # destroyed
    255: (100, 100, 100),  # ignore
}


def mask_to_color(mask):
    mask_arr = np.asarray(mask).astype(np.uint8)
    color = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 3), dtype=np.uint8)

    for class_id, rgb in COLOR_MAP.items():
        color[mask_arr == class_id] = rgb

    return Image.fromarray(color)


def overlay_mask(image, mask, alpha=0.35):
    image = image.convert("RGB")
    color_mask = mask_to_color(mask).convert("RGB")

    mask_arr = np.asarray(mask)
    foreground = (mask_arr > 0) & (mask_arr != 255)

    image_arr = np.asarray(image).copy()
    color_arr = np.asarray(color_mask).copy()

    blended = image_arr.copy()
    blended[foreground] = (
        (1 - alpha) * image_arr[foreground] + alpha * color_arr[foreground]
    ).astype(np.uint8)

    return Image.fromarray(blended)


def load_rows(manifest_csv, split, max_samples, seed):
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows = [
        row for row in rows
        if row["split"] == split and not parse_bool(row["is_augmented"])
    ]

    rng = random.Random(seed)
    rng.shuffle(rows)

    return rows[:max_samples]


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    base_channels = int(config.get("base_channels", 16))

    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_channels=base_channels,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config


@torch.no_grad()
def predict_mask(model, image, device):
    image_array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))

    image_tensor = torch.from_numpy(image_array).unsqueeze(0).to(device)

    logits = model(image_tensor)
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    return Image.fromarray(pred)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize segmentation model predictions as image / ground truth / prediction contact sheet."
    )

    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    manifest_csv = args.dataset_root / "metadata" / "manifest.csv"

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model, config = load_model(args.checkpoint, device)
    rows = load_rows(
        manifest_csv=manifest_csv,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    if not rows:
        raise ValueError(f"No rows found for split={args.split}")

    thumb = 160
    label_h = 54
    cols = 3
    sample_rows = len(rows)

    sheet_w = cols * thumb
    sheet_h = sample_rows * (thumb + label_h)

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for i, row in enumerate(rows):
        image = Image.open(row["image_path"]).convert("RGB")
        gt_mask = Image.open(row["mask_path"]).convert("L")
        pred_mask = predict_mask(model, image, device)

        image_thumb = image.resize((thumb, thumb))
        gt_overlay = overlay_mask(image, gt_mask).resize((thumb, thumb))
        pred_overlay = overlay_mask(image, pred_mask).resize((thumb, thumb))

        y = i * (thumb + label_h)

        sheet.paste(image_thumb, (0, y))
        sheet.paste(gt_overlay, (thumb, y))
        sheet.paste(pred_overlay, (2 * thumb, y))

        label = row["label"]
        building = row["building_id"][:8]

        draw.text((4, y + thumb + 4), "image", fill="black")
        draw.text((thumb + 4, y + thumb + 4), "ground truth", fill="black")
        draw.text((2 * thumb + 4, y + thumb + 4), "prediction", fill="black")

        draw.text((4, y + thumb + 22), f"{label} | {building}", fill="black")
        draw.text((thumb + 4, y + thumb + 22), f"GT: {label}", fill="black")

        pred_values = sorted(set(np.asarray(pred_mask).reshape(-1).tolist()))
        pred_labels = [ID_TO_LABEL.get(v, str(v)) for v in pred_values]
        draw.text((2 * thumb + 4, y + thumb + 22), f"pred: {', '.join(pred_labels[:2])}", fill="black")

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output_png)

    print(f"Saved prediction contact sheet to: {args.output_png}")


if __name__ == "__main__":
    main()
