#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ID_TO_LABEL = {
    0: "background",
    1: "no damage",
    2: "minor damage",
    3: "major damage",
    4: "destroyed",
    255: "ignore",
}


def parse_bool(value):
    return str(value).lower() in {"true", "1", "yes"}


def main():
    parser = argparse.ArgumentParser(
        description="Analyze class balance in a generated segmentation dataset."
    )

    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)

    args = parser.parse_args()

    manifest = args.dataset_root / "metadata" / "manifest.csv"

    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pixel_counts = defaultdict(Counter)
    image_label_counts = defaultdict(Counter)

    for row in rows:
        split = row["split"]

        # For balance analysis, keep augmentation included for train,
        # but val/test are already unaugmented.
        mask_path = Path(row["mask_path"])
        mask = Image.open(mask_path).convert("L")
        arr = np.asarray(mask)

        values, counts = np.unique(arr, return_counts=True)

        for value, count in zip(values, counts):
            value = int(value)
            count = int(count)
            pixel_counts[split][value] += count

        image_label_counts[split][row["label"]] += 1

    summary = {}

    for split, counter in pixel_counts.items():
        total_pixels = sum(counter.values())

        split_summary = {
            "total_pixels": total_pixels,
            "pixel_counts": {},
            "pixel_fractions": {},
            "image_label_counts": dict(image_label_counts[split]),
        }

        for class_id, count in sorted(counter.items()):
            label = ID_TO_LABEL.get(class_id, str(class_id))
            split_summary["pixel_counts"][label] = count
            split_summary["pixel_fractions"][label] = count / total_pixels if total_pixels else 0.0

        summary[split] = split_summary

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Segmentation class balance")
    print("--------------------------")

    for split, split_summary in summary.items():
        print()
        print(split)
        print("Image label counts:", split_summary["image_label_counts"])
        print("Pixel fractions:")
        for label, fraction in split_summary["pixel_fractions"].items():
            print(f"  {label}: {fraction:.6f}")

    print()
    print(f"Saved class balance summary to: {args.output_json}")


if __name__ == "__main__":
    main()
