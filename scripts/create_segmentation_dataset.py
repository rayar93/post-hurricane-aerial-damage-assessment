#!/usr/bin/env python3

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageEnhance, ImageOps


LABEL_TO_ID = {
    "no damage": 1,
    "minor damage": 2,
    "major damage": 3,
    "destroyed": 4,
}

ID_TO_LABEL = {
    0: "background",
    1: "no damage",
    2: "minor damage",
    3: "major damage",
    4: "destroyed",
    255: "ignore",
}


def safe_filename(value):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))


def normalize_to_uint8(array):
    array = array.astype("float32")
    valid = np.isfinite(array)

    if not valid.any():
        return np.zeros(array.shape, dtype=np.uint8)

    p2, p98 = np.percentile(array[valid], [2, 98])

    if p98 <= p2:
        return np.zeros(array.shape, dtype=np.uint8)

    array = np.clip((array - p2) / (p98 - p2), 0, 1)
    return (array * 255).astype(np.uint8)


def read_geotiff_rgb(geotiff_path):
    with rasterio.open(geotiff_path) as src:
        bands = []

        if src.count >= 3:
            for band_index in [1, 2, 3]:
                band = src.read(band_index)
                if band.dtype != np.uint8:
                    band = normalize_to_uint8(band)
                bands.append(band)
        else:
            band = src.read(1)
            if band.dtype != np.uint8:
                band = normalize_to_uint8(band)
            bands = [band, band, band]

        rgb = np.dstack(bands)

        metadata = {
            "crs": str(src.crs),
            "transform": str(src.transform),
            "bounds_left": src.bounds.left,
            "bounds_bottom": src.bounds.bottom,
            "bounds_right": src.bounds.right,
            "bounds_top": src.bounds.top,
            "resolution_x": src.res[0],
            "resolution_y": src.res[1],
            "width": src.width,
            "height": src.height,
            "count_bands": src.count,
        }

    return rgb, metadata


def polygon_points_from_pixels(pixels):
    points = []

    if not isinstance(pixels, list):
        return points

    for point in pixels:
        if not isinstance(point, dict):
            continue

        if "x" not in point or "y" not in point:
            continue

        try:
            x = int(round(float(point["x"])))
            y = int(round(float(point["y"])))
            points.append((x, y))
        except Exception:
            continue

    return points


def polygon_bbox(points, image_width, image_height, margin):
    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    min_x = max(min(xs) - margin, 0)
    max_x = min(max(xs) + margin, image_width - 1)
    min_y = max(min(ys) - margin, 0)
    max_y = min(max(ys) + margin, image_height - 1)

    if max_x <= min_x or max_y <= min_y:
        return None

    return {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "width": max_x - min_x + 1,
        "height": max_y - min_y + 1,
    }


def compute_black_fraction(image):
    arr = np.asarray(image.convert("RGB"))
    black = np.all(arr <= 10, axis=2)
    return float(black.mean())


def enhance_contrast(image):
    return ImageOps.autocontrast(image)


def denoise_image(image):
    arr = np.asarray(image.convert("RGB"))
    denoised = cv2.medianBlur(arr, 3)
    return Image.fromarray(denoised)


def resize_pair(image, mask, image_size):
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
    return image, mask


def build_records(annotation_json, image_width, image_height, margin, min_crop_width, min_crop_height):
    with annotation_json.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    if not isinstance(annotations, list):
        raise ValueError(f"Expected annotation JSON list, got {type(annotations)}")

    records = []

    for index, record in enumerate(annotations):
        label = record.get("label", "")

        if label not in LABEL_TO_ID:
            continue

        points = polygon_points_from_pixels(record.get("pixels", []))
        bbox = polygon_bbox(points, image_width, image_height, margin)

        if bbox is None:
            continue

        if bbox["width"] < min_crop_width or bbox["height"] < min_crop_height:
            continue

        building_id = record.get("building_id", "")
        view_id = record.get("view_id", "")
        record_id = record.get("id", "")

        records.append({
            "record_index": index,
            "id": record_id,
            "building_id": building_id,
            "view_id": view_id,
            "label": label,
            "class_id": LABEL_TO_ID[label],
            "source": record.get("source", ""),
            "filename": record.get("filename", ""),
            "jds_version": record.get("jds_version", ""),
            "payload_version": record.get("payload_version", ""),
            "points": points,
            "min_x": bbox["min_x"],
            "min_y": bbox["min_y"],
            "max_x": bbox["max_x"],
            "max_y": bbox["max_y"],
            "crop_width": bbox["width"],
            "crop_height": bbox["height"],
        })

    return records


def stratified_group_split(records, train_ratio, val_ratio, test_ratio, seed):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(seed)

    groups = defaultdict(list)

    for record in records:
        group_key = record["building_id"] or record["id"] or str(record["record_index"])
        groups[group_key].append(record)

    group_items = []

    for group_key, group_records in groups.items():
        labels = [r["label"] for r in group_records]
        label = Counter(labels).most_common(1)[0][0]

        group_items.append({
            "group_key": group_key,
            "label": label,
            "records": group_records,
        })

    by_label = defaultdict(list)

    for item in group_items:
        by_label[item["label"]].append(item)

    split_by_group = {}

    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)

        if n < 3:
            for item in items:
                split_by_group[item["group_key"]] = "train"
            continue

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        if n_train < 1:
            n_train = 1
        if n_val < 1 and val_ratio > 0:
            n_val = 1

        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1

        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]

        for item in train_items:
            split_by_group[item["group_key"]] = "train"
        for item in val_items:
            split_by_group[item["group_key"]] = "val"
        for item in test_items:
            split_by_group[item["group_key"]] = "test"

    for record in records:
        group_key = record["building_id"] or record["id"] or str(record["record_index"])
        record["split"] = split_by_group[group_key]

    return records


def crop_image_and_mask(rgb_image, record):
    image_width, image_height = rgb_image.size

    full_mask = Image.new("L", (image_width, image_height), 0)
    draw = ImageDraw.Draw(full_mask)

    points = record["points"]
    class_id = int(record["class_id"])

    if len(points) >= 3:
        draw.polygon(points, fill=class_id)

    box = (
        record["min_x"],
        record["min_y"],
        record["max_x"] + 1,
        record["max_y"] + 1,
    )

    image_crop = rgb_image.crop(box)
    mask_crop = full_mask.crop(box)

    return image_crop, mask_crop


def mask_has_foreground(mask):
    arr = np.asarray(mask)
    return bool(np.any((arr > 0) & (arr != 255)))


def augment_pairs(image, mask, max_augmentations):
    augmentations = []

    augmentations.append(("hflip", ImageOps.mirror(image), ImageOps.mirror(mask)))
    augmentations.append(("vflip", ImageOps.flip(image), ImageOps.flip(mask)))
    augmentations.append(("rot90", image.rotate(90, expand=True), mask.rotate(90, expand=True)))
    augmentations.append(("rot270", image.rotate(270, expand=True), mask.rotate(270, expand=True)))

    bright = ImageEnhance.Brightness(image).enhance(1.15)
    augmentations.append(("brightness_up", bright, mask.copy()))

    contrast = ImageEnhance.Contrast(image).enhance(1.15)
    augmentations.append(("contrast_up", contrast, mask.copy()))

    return augmentations[:max_augmentations]


def save_pair(image, mask, output_root, split, sample_id):
    image_dir = output_root / split / "images"
    mask_dir = output_root / split / "masks"

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / f"{sample_id}.png"
    mask_path = mask_dir / f"{sample_id}_mask.png"

    image.save(image_path)
    mask.save(mask_path)

    return image_path, mask_path


def main():
    parser = argparse.ArgumentParser(
        description="Create a leakage-safe stratified segmentation dataset from one CRASAR GeoTIFF and annotation JSON."
    )

    parser.add_argument("--geotiff-path", required=True, type=Path)
    parser.add_argument("--annotation-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--margin", type=int, default=20)

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--min-crop-width", type=int, default=30)
    parser.add_argument("--min-crop-height", type=int, default=30)
    parser.add_argument("--max-black-fraction", type=float, default=0.45)

    parser.add_argument("--enhance-contrast", action="store_true")
    parser.add_argument("--denoise", action="store_true")

    parser.add_argument("--augment-train", action="store_true")
    parser.add_argument("--augmentations-per-image", type=int, default=2)

    args = parser.parse_args()

    if not args.geotiff_path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {args.geotiff_path}")

    if not args.annotation_json.exists():
        raise FileNotFoundError(f"Annotation JSON not found: {args.annotation_json}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = args.output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("Reading GeoTIFF...")
    rgb_array, geotiff_metadata = read_geotiff_rgb(args.geotiff_path)
    rgb_image = Image.fromarray(rgb_array).convert("RGB")

    image_width, image_height = rgb_image.size
    print(f"GeoTIFF image size: {image_width} x {image_height}")

    print("Building original records...")
    records = build_records(
        annotation_json=args.annotation_json,
        image_width=image_width,
        image_height=image_height,
        margin=args.margin,
        min_crop_width=args.min_crop_width,
        min_crop_height=args.min_crop_height,
    )

    print(f"Usable annotation records before split: {len(records)}")
    print(f"Label counts before split: {dict(Counter(r['label'] for r in records))}")

    print("Creating stratified leakage-safe split...")
    records = stratified_group_split(
        records=records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    output_rows = []
    counters = Counter()

    print("Extracting image/mask crops and applying preprocessing...")

    for record in records:
        split = record["split"]

        image_crop, mask_crop = crop_image_and_mask(rgb_image, record)

        black_fraction = compute_black_fraction(image_crop)

        if black_fraction > args.max_black_fraction:
            continue

        if args.enhance_contrast:
            image_crop = enhance_contrast(image_crop)

        if args.denoise:
            image_crop = denoise_image(image_crop)

        image_crop, mask_crop = resize_pair(image_crop, mask_crop, args.image_size)

        if not mask_has_foreground(mask_crop):
            continue

        sample_base = (
            f"{split}_"
            f"{counters[split]:06d}_"
            f"{safe_filename(record['label'])}_"
            f"{safe_filename(record['building_id'])[:10]}"
        )

        image_path, mask_path = save_pair(
            image=image_crop,
            mask=mask_crop,
            output_root=args.output_root,
            split=split,
            sample_id=sample_base,
        )

        row = dict(record)
        row.update(geotiff_metadata)
        row["image_path"] = str(image_path)
        row["mask_path"] = str(mask_path)
        row["is_augmented"] = False
        row["augmentation"] = ""
        row["black_fraction"] = black_fraction
        row["image_size"] = args.image_size
        row["original_geotiff_path"] = str(args.geotiff_path)
        row["annotation_json"] = str(args.annotation_json)

        output_rows.append(row)
        counters[split] += 1

        if args.augment_train and split == "train":
            aug_pairs = augment_pairs(
                image=image_crop,
                mask=mask_crop,
                max_augmentations=args.augmentations_per_image,
            )

            for aug_name, aug_image, aug_mask in aug_pairs:
                aug_image, aug_mask = resize_pair(aug_image, aug_mask, args.image_size)

                aug_sample_id = f"{sample_base}_aug_{aug_name}"

                aug_image_path, aug_mask_path = save_pair(
                    image=aug_image,
                    mask=aug_mask,
                    output_root=args.output_root,
                    split=split,
                    sample_id=aug_sample_id,
                )

                aug_row = dict(row)
                aug_row["image_path"] = str(aug_image_path)
                aug_row["mask_path"] = str(aug_mask_path)
                aug_row["is_augmented"] = True
                aug_row["augmentation"] = aug_name

                output_rows.append(aug_row)

    manifest_csv = metadata_dir / "manifest.csv"
    split_summary_json = metadata_dir / "split_summary.json"
    label_mapping_json = metadata_dir / "label_mapping.json"

    if output_rows:
        fieldnames = sorted(set().union(*(r.keys() for r in output_rows)))

        with manifest_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

    split_counts = Counter(r["split"] for r in output_rows if not r["is_augmented"])
    split_counts_with_aug = Counter(r["split"] for r in output_rows)
    label_counts_by_split = defaultdict(Counter)
    label_counts_by_split_with_aug = defaultdict(Counter)

    for row in output_rows:
        if not row["is_augmented"]:
            label_counts_by_split[row["split"]][row["label"]] += 1
        label_counts_by_split_with_aug[row["split"]][row["label"]] += 1

    summary = {
        "geotiff_path": str(args.geotiff_path),
        "annotation_json": str(args.annotation_json),
        "output_root": str(args.output_root),
        "image_size": args.image_size,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "original_usable_records": len(records),
        "saved_original_samples_by_split": dict(split_counts),
        "saved_samples_by_split_with_augmentation": dict(split_counts_with_aug),
        "label_counts_by_split_original_only": {
            split: dict(counter) for split, counter in label_counts_by_split.items()
        },
        "label_counts_by_split_with_augmentation": {
            split: dict(counter) for split, counter in label_counts_by_split_with_aug.items()
        },
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "geotiff_metadata": geotiff_metadata,
    }

    with split_summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with label_mapping_json.open("w", encoding="utf-8") as f:
        json.dump({
            "label_to_id": LABEL_TO_ID,
            "id_to_label": ID_TO_LABEL,
            "ignore_index": 255,
        }, f, indent=2)

    print()
    print("Segmentation dataset creation summary")
    print("-------------------------------------")
    print(f"Manifest CSV: {manifest_csv}")
    print(f"Split summary JSON: {split_summary_json}")
    print(f"Label mapping JSON: {label_mapping_json}")
    print(f"Saved original samples by split: {dict(split_counts)}")
    print(f"Saved samples by split with augmentation: {dict(split_counts_with_aug)}")
    print("Label counts by split, original only:")
    for split, counter in label_counts_by_split.items():
        print(f"  {split}: {dict(counter)}")


if __name__ == "__main__":
    main()
