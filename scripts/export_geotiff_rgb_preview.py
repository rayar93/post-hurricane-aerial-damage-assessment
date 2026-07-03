#!/usr/bin/env python3

import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="Export a normal RGB PNG preview from a GeoTIFF."
    )

    parser.add_argument(
        "--geotiff-path",
        required=True,
        type=Path,
        help="Path to a local GeoTIFF file.",
    )

    parser.add_argument(
        "--output-png",
        required=True,
        type=Path,
        help="Output PNG path.",
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=2000,
        help="Maximum width or height for preview output.",
    )

    args = parser.parse_args()

    if not args.geotiff_path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {args.geotiff_path}")

    with rasterio.open(args.geotiff_path) as src:
        if src.count >= 3:
            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)

            rgb = np.dstack([
                normalize_to_uint8(red),
                normalize_to_uint8(green),
                normalize_to_uint8(blue),
            ])
        else:
            band = normalize_to_uint8(src.read(1))
            rgb = np.dstack([band, band, band])

    image = Image.fromarray(rgb)

    width, height = image.size
    scale = min(args.max_size / max(width, height), 1.0)

    if scale < 1.0:
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output_png)

    print(f"Saved RGB preview to: {args.output_png}")
    print(f"Original size: {width} x {height}")
    print(f"Preview size: {image.size[0]} x {image.size[1]}")


if __name__ == "__main__":
    main()
