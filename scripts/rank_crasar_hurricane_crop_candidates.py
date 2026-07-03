#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "CRASAR/CRASAR-U-DROIDs"
REPO_TYPE = "dataset"

USEFUL_LABELS = {"no damage", "minor damage", "major damage", "destroyed"}
EXCLUDED_LABELS = {"un-classified", "obscured"}


def parse_bool(value):
    return str(value).lower() in {"true", "1", "yes"}


def parse_size_bytes(value):
    try:
        return int(value)
    except Exception:
        return None


def size_mb(size_bytes):
    if size_bytes is None:
        return None
    return size_bytes / (1024 * 1024)


def download_annotation(repo_file):
    return hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=repo_file,
    )


def load_annotation_count(repo_file):
    local_path = download_annotation(repo_file)

    with open(local_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return {
            "records": 0,
            "label_counts": {},
            "error": f"Expected list, got {type(data)}",
        }

    labels = Counter()

    for record in data:
        if not isinstance(record, dict):
            continue
        labels[record.get("label", "missing")] += 1

    return {
        "records": len(data),
        "label_counts": dict(labels),
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rank hurricane-related CRASAR GeoTIFF/annotation pairs by usefulness for building-crop extraction."
    )

    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=Path("outputs/crasar_hurricane_filter/crasar_hurricane_image_annotation_pairs.csv"),
        help="CSV created by create_crasar_hurricane_image_annotation_pairs.py.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/crasar_hurricane_filter/crasar_hurricane_ranked_crop_candidates.csv"),
        help="Output ranked CSV.",
    )

    parser.add_argument(
        "--source",
        default="all",
        choices=["all", "UAS", "SATELLITE", "CREWED", "UAS_DSM"],
        help="Optional source filter.",
    )

    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=150.0,
        help="Only consider imagery files up to this size in MB.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=80,
        help="Maximum number of candidate annotation files to inspect.",
    )

    args = parser.parse_args()

    if not args.pairs_csv.exists():
        raise FileNotFoundError(f"Missing pairs CSV: {args.pairs_csv}")

    with args.pairs_csv.open(newline="", encoding="utf-8") as f:
        pair_rows = list(csv.DictReader(f))

    candidates = []

    for row in pair_rows:
        if not parse_bool(row.get("has_bda_annotation", "")):
            continue

        if args.source != "all" and row.get("source") != args.source:
            continue

        bytes_value = parse_size_bytes(row.get("size_bytes", ""))
        mb_value = size_mb(bytes_value)

        if mb_value is None:
            continue

        if mb_value > args.max_size_mb:
            continue

        candidates.append(row)

    candidates.sort(key=lambda r: parse_size_bytes(r.get("size_bytes", "")) or 10**18)
    candidates = candidates[: args.limit]

    print("Ranking hurricane crop candidates")
    print("---------------------------------")
    print(f"Source filter: {args.source}")
    print(f"Max size MB: {args.max_size_mb}")
    print(f"Candidate pairs selected for annotation inspection: {len(candidates)}")
    print()

    output_rows = []

    for i, row in enumerate(candidates, start=1):
        annotation_path = row["expected_bda_annotation_path"]
        print(f"[{i}/{len(candidates)}] Inspecting annotation: {annotation_path}")

        result = load_annotation_count(annotation_path)
        label_counts = result["label_counts"]

        useful_count = sum(label_counts.get(label, 0) for label in USEFUL_LABELS)
        excluded_count = sum(label_counts.get(label, 0) for label in EXCLUDED_LABELS)
        records = result["records"]

        useful_rate = useful_count / records if records else 0.0

        # Heuristic score: many useful labels, high useful rate, smaller files preferred.
        size_bytes_value = parse_size_bytes(row.get("size_bytes", ""))
        mb_value = size_mb(size_bytes_value) or 0.0
        size_penalty = math.log10(mb_value + 10.0)
        score = useful_count * useful_rate / size_penalty if size_penalty else useful_count * useful_rate

        output_rows.append({
            "score": round(score, 6),
            "imagery_path": row["imagery_path"],
            "annotation_path": annotation_path,
            "source": row.get("source", ""),
            "matched_hurricanes": row.get("matched_hurricanes", ""),
            "size_bytes": row.get("size_bytes", ""),
            "size_human": row.get("size_human", ""),
            "records": records,
            "useful_records": useful_count,
            "useful_rate": round(useful_rate, 6),
            "excluded_records": excluded_count,
            "no_damage": label_counts.get("no damage", 0),
            "minor_damage": label_counts.get("minor damage", 0),
            "major_damage": label_counts.get("major damage", 0),
            "destroyed": label_counts.get("destroyed", 0),
            "un_classified": label_counts.get("un-classified", 0),
            "obscured": label_counts.get("obscured", 0),
            "error": result["error"],
        })

    output_rows.sort(
        key=lambda r: (
            float(r["score"]),
            int(r["useful_records"]),
        ),
        reverse=True,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "score",
        "imagery_path",
        "annotation_path",
        "source",
        "matched_hurricanes",
        "size_bytes",
        "size_human",
        "records",
        "useful_records",
        "useful_rate",
        "excluded_records",
        "no_damage",
        "minor_damage",
        "major_damage",
        "destroyed",
        "un_classified",
        "obscured",
        "error",
    ]

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print()
    print(f"Saved ranked candidates to: {args.output_csv}")
    print()
    print("Top candidates:")
    for row in output_rows[:20]:
        print(
            f"score={row['score']:<10} "
            f"useful={row['useful_records']:<5} "
            f"rate={row['useful_rate']:<6} "
            f"size={row['size_human']:<10} "
            f"source={row['source']:<9} "
            f"hurricane={row['matched_hurricanes']:<18} "
            f"path={row['imagery_path']}"
        )


if __name__ == "__main__":
    main()
