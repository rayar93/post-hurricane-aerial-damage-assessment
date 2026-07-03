#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "CRASAR/CRASAR-U-DROIDs"
REPO_TYPE = "dataset"


def format_size(num_bytes):
    if num_bytes is None:
        return ""

    size = float(num_bytes)

    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Check file sizes for hurricane-related CRASAR imagery files."
    )

    parser.add_argument(
        "--hurricane-imagery-csv",
        type=Path,
        default=Path("outputs/crasar_hurricane_filter/crasar_hurricane_imagery_files.csv"),
        help="CSV created by summarize_crasar_hurricane_files.py.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/crasar_hurricane_filter/crasar_hurricane_imagery_file_sizes.csv"),
        help="Output CSV with file sizes.",
    )

    args = parser.parse_args()

    if not args.hurricane_imagery_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.hurricane_imagery_csv}")

    with args.hurricane_imagery_csv.open(newline="", encoding="utf-8") as f:
        hurricane_rows = list(csv.DictReader(f))

    hurricane_paths = {row["path"] for row in hurricane_rows}

    print("Listing repository tree with file metadata...")
    api = HfApi()
    tree = api.list_repo_tree(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        recursive=True,
        expand=True,
    )

    size_by_path = {}

    for item in tree:
        path = getattr(item, "path", None)
        size = getattr(item, "size", None)

        if path in hurricane_paths:
            size_by_path[path] = size

    output_rows = []

    for row in hurricane_rows:
        path = row["path"]
        size_bytes = size_by_path.get(path)

        output_row = dict(row)
        output_row["size_bytes"] = size_bytes if size_bytes is not None else ""
        output_row["size_human"] = format_size(size_bytes)
        output_rows.append(output_row)

    output_rows.sort(
        key=lambda r: int(r["size_bytes"]) if str(r["size_bytes"]).isdigit() else 10**18
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(output_rows[0].keys())

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved file sizes to: {args.output_csv}")
    print()
    print("Smallest hurricane imagery files:")
    for row in output_rows[:20]:
        print(
            f"{row['size_human']:>10} | {row['source']:<9} | "
            f"{row['matched_hurricanes']:<18} | {row['path']}"
        )


if __name__ == "__main__":
    main()
