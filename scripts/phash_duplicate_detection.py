#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def dct_matrix(n: int) -> np.ndarray:
    matrix = np.empty((n, n), dtype=np.float64)
    factor = np.pi / (2 * n)

    for k in range(n):
        alpha = np.sqrt(1 / n) if k == 0 else np.sqrt(2 / n)
        for i in range(n):
            matrix[k, i] = alpha * np.cos((2 * i + 1) * k * factor)

    return matrix


def compute_phash(image_path: Path, hash_size: int = 8) -> str:
    img_size = hash_size * 4

    with Image.open(image_path) as image:
        image = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float64)

    transform = dct_matrix(img_size)
    dct = transform @ pixels @ transform.T
    low_freq = dct[:hash_size, :hash_size]

    values = low_freq.flatten()
    median = np.median(values[1:])  # exclude DC coefficient
    bits = values > median

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)

    hex_length = (hash_size * hash_size + 3) // 4
    return f"{value:0{hex_length}x}"


def hamming_distance(hash_1: str, hash_2: str) -> int:
    return bin(int(hash_1, 16) ^ int(hash_2, 16)).count("1")


def find_images(input_dir: Path):
    images = [
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--threshold", default=6, type=int)
    args = parser.parse_args()

    images = find_images(args.input_dir)

    if len(images) < 2:
        raise ValueError("Need at least two images to compare.")

    print(f"Found {len(images)} images.")

    hashes = {}
    for image_path in images:
        hashes[image_path] = compute_phash(image_path)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_1",
                "image_2",
                "hash_1",
                "hash_2",
                "hamming_distance",
                "duplicate_candidate",
                "threshold",
            ],
        )
        writer.writeheader()

        duplicate_count = 0
        total_count = 0

        for i in range(len(images) - 1):
            image_1 = images[i]
            image_2 = images[i + 1]

            hash_1 = hashes[image_1]
            hash_2 = hashes[image_2]
            distance = hamming_distance(hash_1, hash_2)
            duplicate_candidate = distance <= args.threshold

            if duplicate_candidate:
                duplicate_count += 1

            total_count += 1

            writer.writerow({
                "image_1": str(image_1),
                "image_2": str(image_2),
                "hash_1": hash_1,
                "hash_2": hash_2,
                "hamming_distance": distance,
                "duplicate_candidate": duplicate_candidate,
                "threshold": args.threshold,
            })

    print(f"Compared {total_count} sequential image pairs.")
    print(f"Duplicate candidates: {duplicate_count}")
    print(f"Report saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
