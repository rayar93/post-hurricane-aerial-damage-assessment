#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expected_bda_annotation_path(imagery_path):
    parts = imagery_path.split("/")

    if "imagery" not in parts:
        return None

    parts = list(parts)
    imagery_index = parts.index("imagery")
    parts[imagery_index] = "annotations"

    source = parts[imagery_index + 1]

    # Insert building_damage_assessment after source.
    parts.insert(imagery_index + 2, "building_damage_assessment")

    return "/".join(parts) + ".json"


def main():
    parser = argparse.ArgumentParser(
        description="Pair hurricane-related CRASAR imagery GeoTIFF files with building damage assessment annotation JSON files."
    )

    parser.add_argument(
        "--imagery-csv",
        type=Path,
        default=Path("outputs/crasar_hurricane_filter/crasar_hurricane_imagery_file_sizes.csv"),
        help="Hurricane imagery CSV with file sizes.",
    )

    parser.add_argument(
        "--annotation-csv",
        type=Path,
        default=Path("outputs/crasar_hurricane_filter/crasar_hurricane_annotation_files.csv"),
        help="Hurricane annotation CSV.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/crasar_hurricane_filter/crasar_hurricane_image_annotation_pairs.csv"),
        help="Output paired CSV.",
    )

    args = parser.parse_args()

    imagery_rows = read_csv(args.imagery_csv)
    annotation_rows = read_csv(args.annotation_csv)

    annotation_paths = {row["path"] for row in annotation_rows}

    paired_rows = []

    for row in imagery_rows:
        imagery_path = row["path"]
        expected_annotation = expected_bda_annotation_path(imagery_path)
        has_bda_annotation = expected_annotation in annotation_paths

        paired_rows.append({
            "imagery_path": imagery_path,
            "expected_bda_annotation_path": expected_annotation or "",
            "has_bda_annotation": has_bda_annotation,
            "source": row.get("source", ""),
            "matched_hurricanes": row.get("matched_hurricanes", ""),
            "size_bytes": row.get("size_bytes", ""),
            "size_human": row.get("size_human", ""),
        })

    paired_rows.sort(
        key=lambda r: (
            not r["has_bda_annotation"],
            int(r["size_bytes"]) if str(r["size_bytes"]).isdigit() else 10**18,
        )
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "imagery_path",
        "expected_bda_annotation_path",
        "has_bda_annotation",
        "source",
        "matched_hurricanes",
        "size_bytes",
        "size_human",
    ]

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paired_rows)

    total = len(paired_rows)
    matched = sum(1 for r in paired_rows if r["has_bda_annotation"])

    print("CRASAR hurricane image-annotation pairing")
    print("----------------------------------------")
    print(f"Imagery files: {total}")
    print(f"Files with expected BDA annotation: {matched}")
    print(f"Files without expected BDA annotation: {total - matched}")
    print(f"Saved pairs CSV to: {args.output_csv}")
    print()
    print("Smallest matched pairs:")
    shown = 0

    for row in paired_rows:
        if not row["has_bda_annotation"]:
            continue

        print(
            f"{row['size_human']:>10} | {row['source']:<9} | "
            f"{row['matched_hurricanes']:<18} | {row['imagery_path']}"
        )
        shown += 1

        if shown >= 20:
            break


if __name__ == "__main__":
    main()
