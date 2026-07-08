#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


VALID_MASK_VALUES = {0, 1, 2, 3, 4, 255}


def parse_bool(value):
    return str(value).lower() in {"true", "1", "yes"}


def main():
    parser = argparse.ArgumentParser(
        description="Validate a generated segmentation dataset for leakage, file integrity, masks, and split quality."
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Root directory of generated segmentation dataset.",
    )

    parser.add_argument(
        "--expected-image-size",
        type=int,
        default=256,
        help="Expected square image/mask size.",
    )

    args = parser.parse_args()

    manifest_csv = args.dataset_root / "metadata" / "manifest.csv"

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    with manifest_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    errors = []
    warnings = []

    split_counts = Counter()
    split_counts_original_only = Counter()
    label_counts = defaultdict(Counter)
    label_counts_original_only = defaultdict(Counter)
    building_ids_by_split = defaultdict(set)
    augmentation_counts = Counter()
    mask_values_seen = set()

    for i, row in enumerate(rows):
        split = row.get("split", "")
        label = row.get("label", "")
        building_id = row.get("building_id", "")
        is_augmented = parse_bool(row.get("is_augmented", "False"))

        image_path = Path(row.get("image_path", ""))
        mask_path = Path(row.get("mask_path", ""))

        split_counts[split] += 1
        label_counts[split][label] += 1

        if not is_augmented:
            split_counts_original_only[split] += 1
            label_counts_original_only[split][label] += 1
            building_ids_by_split[split].add(building_id)

        if is_augmented:
            augmentation_counts[split] += 1

        if is_augmented and split != "train":
            errors.append(f"Augmented sample outside train split at row {i}: split={split}")

        if not image_path.exists():
            errors.append(f"Missing image at row {i}: {image_path}")
            continue

        if not mask_path.exists():
            errors.append(f"Missing mask at row {i}: {mask_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except Exception as exc:
            errors.append(f"Failed reading image/mask at row {i}: {exc}")
            continue

        if image.size != (args.expected_image_size, args.expected_image_size):
            errors.append(f"Unexpected image size at row {i}: {image.size}")

        if mask.size != (args.expected_image_size, args.expected_image_size):
            errors.append(f"Unexpected mask size at row {i}: {mask.size}")

        mask_arr = np.asarray(mask)
        values = set(np.unique(mask_arr).tolist())
        mask_values_seen.update(values)

        invalid_values = values - VALID_MASK_VALUES
        if invalid_values:
            errors.append(f"Invalid mask values at row {i}: {sorted(invalid_values)}")

        if not np.any((mask_arr > 0) & (mask_arr != 255)):
            warnings.append(f"Mask has no foreground at row {i}: {mask_path}")

    # Leakage check across original samples only.
    train_ids = building_ids_by_split.get("train", set())
    val_ids = building_ids_by_split.get("val", set())
    test_ids = building_ids_by_split.get("test", set())

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    if train_val_overlap:
        errors.append(f"Building ID leakage train-val: {len(train_val_overlap)} overlapping IDs")

    if train_test_overlap:
        errors.append(f"Building ID leakage train-test: {len(train_test_overlap)} overlapping IDs")

    if val_test_overlap:
        errors.append(f"Building ID leakage val-test: {len(val_test_overlap)} overlapping IDs")

    summary = {
        "dataset_root": str(args.dataset_root),
        "manifest_csv": str(manifest_csv),
        "total_manifest_rows": len(rows),
        "split_counts_with_augmentation": dict(split_counts),
        "split_counts_original_only": dict(split_counts_original_only),
        "label_counts_with_augmentation": {
            split: dict(counter) for split, counter in label_counts.items()
        },
        "label_counts_original_only": {
            split: dict(counter) for split, counter in label_counts_original_only.items()
        },
        "augmentation_counts_by_split": dict(augmentation_counts),
        "mask_values_seen": sorted(mask_values_seen),
        "building_id_counts_by_split_original_only": {
            split: len(ids) for split, ids in building_ids_by_split.items()
        },
        "leakage": {
            "train_val_overlap": len(train_val_overlap),
            "train_test_overlap": len(train_test_overlap),
            "val_test_overlap": len(val_test_overlap),
        },
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "errors": errors[:100],
        "warnings": warnings[:100],
    }

    output_json = args.dataset_root / "metadata" / "validation_summary.json"

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Segmentation dataset validation summary")
    print("---------------------------------------")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Total manifest rows: {len(rows)}")
    print(f"Split counts with augmentation: {dict(split_counts)}")
    print(f"Split counts original only: {dict(split_counts_original_only)}")
    print(f"Label counts original only: {summary['label_counts_original_only']}")
    print(f"Augmentation counts by split: {dict(augmentation_counts)}")
    print(f"Mask values seen: {sorted(mask_values_seen)}")
    print(f"Leakage: {summary['leakage']}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print(f"Saved validation summary to: {output_json}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
