#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


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
        if src.count >= 3:
            bands = []

            for band_index in [1, 2, 3]:
                band = src.read(band_index)

                if band.dtype == np.uint8:
                    bands.append(band)
                else:
                    bands.append(normalize_to_uint8(band))

            rgb = np.dstack(bands)
        else:
            band = src.read(1)
            if band.dtype != np.uint8:
                band = normalize_to_uint8(band)
            rgb = np.dstack([band, band, band])

        geotiff_metadata = {
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

    return rgb, geotiff_metadata


def polygon_bbox_from_pixels(pixels, image_width, image_height, margin):
    points = []

    for point in pixels:
        if not isinstance(point, dict):
            continue

        if "x" not in point or "y" not in point:
            continue

        try:
            x = int(round(float(point["x"])))
            y = int(round(float(point["y"])))
        except Exception:
            continue

        points.append((x, y))

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
        "num_polygon_points": len(points),
    }


def safe_filename(value):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))


def main():
    parser = argparse.ArgumentParser(
        description="Extract building-level crops from a CRASAR GeoTIFF using building_damage_assessment JSON pixels."
    )

    parser.add_argument(
        "--geotiff-path",
        required=True,
        type=Path,
        help="Path to local CRASAR GeoTIFF file.",
    )

    parser.add_argument(
        "--annotation-json",
        required=True,
        type=Path,
        help="Path to local CRASAR building_damage_assessment JSON file.",
    )

    parser.add_argument(
        "--output-crops-dir",
        required=True,
        type=Path,
        help="Directory where building crop PNG files will be saved.",
    )

    parser.add_argument(
        "--output-metadata-csv",
        required=True,
        type=Path,
        help="CSV metadata file for extracted crops.",
    )

    parser.add_argument(
        "--margin",
        type=int,
        default=20,
        help="Pixel margin around the building polygon bounding box.",
    )

    args = parser.parse_args()

    if not args.geotiff_path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {args.geotiff_path}")

    if not args.annotation_json.exists():
        raise FileNotFoundError(f"Annotation JSON not found: {args.annotation_json}")

    args.output_crops_dir.mkdir(parents=True, exist_ok=True)
    args.output_metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    print("Reading GeoTIFF...")
    rgb, geotiff_metadata = read_geotiff_rgb(args.geotiff_path)

    image_height, image_width = rgb.shape[:2]
    image = Image.fromarray(rgb)

    print(f"GeoTIFF image size: {image_width} x {image_height}")

    with args.annotation_json.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    if not isinstance(annotations, list):
        raise ValueError(f"Expected annotation JSON to be a list, got {type(annotations)}")

    rows = []

    print(f"Annotation records: {len(annotations)}")

    for index, record in enumerate(annotations):
        pixels = record.get("pixels")

        if not isinstance(pixels, list):
            print(f"Skipping record {index}: pixels field is not a list")
            continue

        bbox = polygon_bbox_from_pixels(
            pixels=pixels,
            image_width=image_width,
            image_height=image_height,
            margin=args.margin,
        )

        if bbox is None:
            print(f"Skipping record {index}: could not compute valid bounding box")
            continue

        crop = image.crop((
            bbox["min_x"],
            bbox["min_y"],
            bbox["max_x"] + 1,
            bbox["max_y"] + 1,
        ))

        building_id = record.get("building_id", f"unknown_building_{index}")
        view_id = record.get("view_id", f"unknown_view_{index}")
        label = record.get("label", "unknown")

        crop_filename = (
            f"{index:04d}_"
            f"{safe_filename(label)}_"
            f"{safe_filename(building_id)[:12]}_"
            f"{safe_filename(view_id)[:12]}.png"
        )

        crop_path = args.output_crops_dir / crop_filename
        crop.save(crop_path)

        row = {
            "crop_path": str(crop_path),
            "record_index": index,
            "building_id": building_id,
            "view_id": view_id,
            "label": label,
            "source": record.get("source", ""),
            "filename": record.get("filename", ""),
            "id": record.get("id", ""),
            "jds_version": record.get("jds_version", ""),
            "payload_version": record.get("payload_version", ""),
            "original_geotiff_path": str(args.geotiff_path),
            "annotation_json": str(args.annotation_json),
            "min_x": bbox["min_x"],
            "min_y": bbox["min_y"],
            "max_x": bbox["max_x"],
            "max_y": bbox["max_y"],
            "crop_width": bbox["width"],
            "crop_height": bbox["height"],
            "num_polygon_points": bbox["num_polygon_points"],
            "crs": geotiff_metadata["crs"],
            "transform": geotiff_metadata["transform"],
            "resolution_x": geotiff_metadata["resolution_x"],
            "resolution_y": geotiff_metadata["resolution_y"],
            "bounds_left": geotiff_metadata["bounds_left"],
            "bounds_bottom": geotiff_metadata["bounds_bottom"],
            "bounds_right": geotiff_metadata["bounds_right"],
            "bounds_top": geotiff_metadata["bounds_top"],
        }

        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())

        with args.output_metadata_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("Building crop extraction summary")
    print("--------------------------------")
    print(f"Crops saved: {len(rows)}")
    print(f"Output crops directory: {args.output_crops_dir}")
    print(f"Output metadata CSV: {args.output_metadata_csv}")


if __name__ == "__main__":
    main()
