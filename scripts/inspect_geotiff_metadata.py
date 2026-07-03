#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import rasterio


def main():
    parser = argparse.ArgumentParser(
        description="Inspect metadata and raster structure of a GeoTIFF file."
    )

    parser.add_argument(
        "--geotiff-path",
        required=True,
        type=Path,
        help="Path to a local .geo.tif or .tif file.",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )

    args = parser.parse_args()

    if not args.geotiff_path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {args.geotiff_path}")

    with rasterio.open(args.geotiff_path) as src:
        metadata = {
            "path": str(args.geotiff_path),
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "count_bands": src.count,
            "crs": str(src.crs),
            "transform": str(src.transform),
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "resolution": src.res,
            "dtypes": list(src.dtypes),
            "indexes": list(src.indexes),
            "nodata": src.nodata,
        }

    print(json.dumps(metadata, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)

        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to: {args.output_json}")


if __name__ == "__main__":
    main()
