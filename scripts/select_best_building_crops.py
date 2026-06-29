#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def dct_matrix(n):
    matrix = np.empty((n, n), dtype=np.float64)
    factor = np.pi / (2 * n)

    for k in range(n):
        alpha = np.sqrt(1 / n) if k == 0 else np.sqrt(2 / n)
        for i in range(n):
            matrix[k, i] = alpha * np.cos((2 * i + 1) * k * factor)

    return matrix


def compute_phash(image_path, hash_size=8):
    img_size = hash_size * 4

    with Image.open(image_path) as image:
        image = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float64)

    transform = dct_matrix(img_size)
    dct = transform @ pixels @ transform.T
    low_freq = dct[:hash_size, :hash_size]

    values = low_freq.flatten()
    median = np.median(values[1:])
    bits = values > median

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)

    hex_length = (hash_size * hash_size + 3) // 4
    return f"{value:0{hex_length}x}"


def hamming_distance(hash_1, hash_2):
    return bin(int(hash_1, 16) ^ int(hash_2, 16)).count("1")


def image_quality_metrics(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
    contrast = float(image.std())
    brightness = float(image.mean())
    area = int(width * height)

    brightness_balance = 1.0 - abs(brightness - 127.5) / 127.5
    brightness_balance = max(0.0, min(1.0, brightness_balance))

    return {
        "width": width,
        "height": height,
        "area": area,
        "sharpness": sharpness,
        "contrast": contrast,
        "brightness": brightness,
        "brightness_balance": brightness_balance,
    }


def normalize(values):
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    if minimum == maximum:
        return [1.0 for _ in values]

    return [(x - minimum) / (maximum - minimum) for x in values]


def read_metadata(metadata_csv):
    rows = []

    with metadata_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"building_id", "crop_path"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for row in reader:
            crop_path = Path(row["crop_path"])

            if crop_path.exists():
                rows.append(row)
            else:
                print(f"Warning: crop path not found: {crop_path}")

    return rows


def compute_within_building_redundancy(candidates, duplicate_threshold):
    if len(candidates) < 2:
        return {
            "within_building_pairs": 0,
            "within_building_duplicate_pairs": 0,
            "within_building_duplicate_rate": 0.0,
            "mean_pairwise_hamming_distance": "",
            "min_pairwise_hamming_distance": "",
        }

    total_pairs = 0
    duplicate_pairs = 0
    distances = []

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            total_pairs += 1
            distance = hamming_distance(candidates[i]["phash"], candidates[j]["phash"])
            distances.append(distance)

            if distance <= duplicate_threshold:
                duplicate_pairs += 1

    return {
        "within_building_pairs": total_pairs,
        "within_building_duplicate_pairs": duplicate_pairs,
        "within_building_duplicate_rate": duplicate_pairs / total_pairs if total_pairs else 0.0,
        "mean_pairwise_hamming_distance": float(np.mean(distances)) if distances else "",
        "min_pairwise_hamming_distance": min(distances) if distances else "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Select representative building crops using pHash and image-quality metrics."
    )

    parser.add_argument(
        "--metadata-csv",
        required=True,
        type=Path,
        help="CSV with at least building_id and crop_path columns.",
    )

    parser.add_argument(
        "--output-best-csv",
        required=True,
        type=Path,
        help="Output CSV with selected best crop per building.",
    )

    parser.add_argument(
        "--output-all-csv",
        required=True,
        type=Path,
        help="Output CSV with metrics for all crop candidates.",
    )

    parser.add_argument(
        "--duplicate-threshold",
        type=int,
        default=6,
        help="pHash Hamming-distance threshold for near-duplicate crops.",
    )

    args = parser.parse_args()

    rows = read_metadata(args.metadata_csv)

    if not rows:
        raise ValueError("No valid crop rows found.")

    enriched = []

    print(f"Processing {len(rows)} crop candidates...")

    for row in rows:
        crop_path = Path(row["crop_path"])
        metrics = image_quality_metrics(crop_path)

        enriched_row = dict(row)
        enriched_row.update(metrics)
        enriched_row["phash"] = compute_phash(crop_path)

        enriched.append(enriched_row)

    sharpness_norm = normalize([float(r["sharpness"]) for r in enriched])
    contrast_norm = normalize([float(r["contrast"]) for r in enriched])
    area_norm = normalize([float(r["area"]) for r in enriched])

    for i, row in enumerate(enriched):
        row["sharpness_norm"] = sharpness_norm[i]
        row["contrast_norm"] = contrast_norm[i]
        row["area_norm"] = area_norm[i]

        row["quality_score"] = (
            0.45 * row["sharpness_norm"]
            + 0.25 * row["contrast_norm"]
            + 0.20 * row["area_norm"]
            + 0.10 * float(row["brightness_balance"])
        )

    by_building = defaultdict(list)

    for row in enriched:
        by_building[row["building_id"]].append(row)

    best_rows = []

    for building_id, candidates in by_building.items():
        best = max(candidates, key=lambda r: float(r["quality_score"]))
        best = dict(best)

        redundancy = compute_within_building_redundancy(
            candidates,
            duplicate_threshold=args.duplicate_threshold,
        )

        best["num_candidate_crops"] = len(candidates)
        best["selection_reason"] = "highest_quality_score"
        best.update(redundancy)

        best_rows.append(best)

    args.output_best_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_all_csv.parent.mkdir(parents=True, exist_ok=True)

    all_fields = sorted(set().union(*(r.keys() for r in enriched)))
    best_fields = sorted(set().union(*(r.keys() for r in best_rows)))

    with args.output_all_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(enriched)

    with args.output_best_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=best_fields)
        writer.writeheader()
        writer.writerows(best_rows)

    print(f"Crop candidates processed: {len(enriched)}")
    print(f"Buildings found: {len(by_building)}")
    print(f"Best crops selected: {len(best_rows)}")
    print(f"Saved all metrics to: {args.output_all_csv}")
    print(f"Saved best crops to: {args.output_best_csv}")


if __name__ == "__main__":
    main()
