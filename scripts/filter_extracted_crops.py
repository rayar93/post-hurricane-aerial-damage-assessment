#!/usr/bin/env python3

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


USEFUL_LABELS = {"no damage", "minor damage", "major damage", "destroyed"}


def compute_black_fraction(image_path, threshold=10):
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image)

    # Pixel is considered black/no-data if all RGB channels are very dark.
    black_pixels = np.all(arr <= threshold, axis=2)
    return float(black_pixels.mean())


def parse_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def parse_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def main():
    parser = argparse.ArgumentParser(
        description="Filter extracted CRASAR building crops using label, size, quality score, and black/no-data fraction."
    )

    parser.add_argument(
        "--input-csv",
        required=True,
        type=Path,
        help="Input crop metrics CSV, usually produced by select_best_building_crops.py.",
    )

    parser.add_argument(
        "--output-all-csv",
        required=True,
        type=Path,
        help="Output CSV with added quality flags.",
    )

    parser.add_argument(
        "--output-filtered-csv",
        required=True,
        type=Path,
        help="Output CSV containing only crops that pass the filter.",
    )

    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.15,
        help="Minimum quality_score required.",
    )

    parser.add_argument(
        "--max-black-fraction",
        type=float,
        default=0.35,
        help="Maximum allowed fraction of nearly black pixels.",
    )

    parser.add_argument(
        "--min-crop-width",
        type=int,
        default=60,
        help="Minimum crop width.",
    )

    parser.add_argument(
        "--min-crop-height",
        type=int,
        default=50,
        help="Minimum crop height.",
    )

    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    with args.input_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    enriched = []
    filtered = []

    for row in rows:
        crop_path = Path(row["crop_path"])

        if not crop_path.exists():
            row["filter_pass"] = False
            row["filter_reason"] = "missing_crop_path"
            row["black_fraction"] = ""
            enriched.append(row)
            continue

        black_fraction = compute_black_fraction(crop_path)

        label = row.get("label", "")
        quality_score = parse_float(row.get("quality_score", 0))
        crop_width = parse_int(row.get("crop_width", row.get("width", 0)))
        crop_height = parse_int(row.get("crop_height", row.get("height", 0)))

        reasons = []

        if label not in USEFUL_LABELS:
            reasons.append("excluded_label")

        if quality_score < args.min_quality_score:
            reasons.append("low_quality_score")

        if black_fraction > args.max_black_fraction:
            reasons.append("too_much_black_background")

        if crop_width < args.min_crop_width:
            reasons.append("crop_too_narrow")

        if crop_height < args.min_crop_height:
            reasons.append("crop_too_short")

        row = dict(row)
        row["black_fraction"] = black_fraction
        row["filter_pass"] = len(reasons) == 0
        row["filter_reason"] = ";".join(reasons)

        enriched.append(row)

        if row["filter_pass"]:
            filtered.append(row)

    args.output_all_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_filtered_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted(set().union(*(r.keys() for r in enriched)))

    with args.output_all_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)

    with args.output_filtered_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)

    label_counts_all = Counter(r.get("label", "") for r in enriched)
    label_counts_filtered = Counter(r.get("label", "") for r in filtered)
    reason_counts = Counter()

    for row in enriched:
        reason = row.get("filter_reason", "")
        if not reason:
            reason_counts["passed"] += 1
        else:
            for part in reason.split(";"):
                reason_counts[part] += 1

    print("Crop filtering summary")
    print("----------------------")
    print(f"Input crops: {len(enriched)}")
    print(f"Filtered crops kept: {len(filtered)}")
    print(f"Kept rate: {len(filtered) / len(enriched):.4f}" if enriched else "Kept rate: 0")
    print(f"All label counts: {dict(label_counts_all)}")
    print(f"Filtered label counts: {dict(label_counts_filtered)}")
    print(f"Filter reason counts: {dict(reason_counts)}")
    print(f"Saved all crops with flags to: {args.output_all_csv}")
    print(f"Saved filtered crops to: {args.output_filtered_csv}")


if __name__ == "__main__":
    main()
