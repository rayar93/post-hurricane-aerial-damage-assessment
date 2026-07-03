#!/usr/bin/env python3

import csv
import json
from collections import Counter
from pathlib import Path


INPUT_CSV = Path("outputs/crasar_hurricane_filter/crasar_hurricane_files.csv")
OUTPUT_DIR = Path("outputs/crasar_hurricane_filter")


def split_hurricanes(value):
    if not value:
        return []
    return [x for x in value.split(";") if x]


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}. Run scripts/filter_crasar_hurricane_files.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    hurricane_counter = Counter()
    source_counter = Counter()
    category_counter = Counter()
    annotation_type_counter = Counter()
    imagery_rows = []
    annotation_rows = []

    for row in rows:
        hurricanes = split_hurricanes(row.get("matched_hurricanes", ""))

        for hurricane in hurricanes:
            hurricane_counter[hurricane] += 1

        source_counter[row.get("source", "")] += 1
        category_counter[row.get("category", "")] += 1
        annotation_type_counter[row.get("annotation_type", "")] += 1

        if row.get("category") == "imagery":
            imagery_rows.append(row)

        if row.get("category") == "annotations":
            annotation_rows.append(row)

    imagery_csv = OUTPUT_DIR / "crasar_hurricane_imagery_files.csv"
    annotation_csv = OUTPUT_DIR / "crasar_hurricane_annotation_files.csv"
    summary_json = OUTPUT_DIR / "crasar_hurricane_summary.json"

    fieldnames = [
        "path",
        "extension",
        "source",
        "category",
        "annotation_type",
        "matched_hurricanes",
        "is_hurricane_related",
    ]

    with imagery_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(imagery_rows)

    with annotation_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotation_rows)

    summary = {
        "total_hurricane_related_files": len(rows),
        "hurricane_imagery_files": len(imagery_rows),
        "hurricane_annotation_files": len(annotation_rows),
        "by_hurricane": dict(hurricane_counter),
        "by_source": dict(source_counter),
        "by_category": dict(category_counter),
        "by_annotation_type": dict(annotation_type_counter),
        "imagery_csv": str(imagery_csv),
        "annotation_csv": str(annotation_csv),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("CRASAR hurricane summary")
    print("------------------------")
    print(f"Total hurricane-related files: {len(rows)}")
    print(f"Hurricane imagery files: {len(imagery_rows)}")
    print(f"Hurricane annotation files: {len(annotation_rows)}")
    print(f"By hurricane: {dict(hurricane_counter)}")
    print(f"By source: {dict(source_counter)}")
    print(f"By category: {dict(category_counter)}")
    print(f"By annotation type: {dict(annotation_type_counter)}")
    print(f"Saved imagery CSV to: {imagery_csv}")
    print(f"Saved annotation CSV to: {annotation_csv}")
    print(f"Saved summary JSON to: {summary_json}")

    print()
    print("First hurricane imagery files:")
    for row in imagery_rows[:20]:
        print(f"- {row['source']} | {row['matched_hurricanes']} | {row['path']}")


if __name__ == "__main__":
    main()
