#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "CRASAR/CRASAR-U-DROIDs"
REPO_TYPE = "dataset"


HURRICANE_KEYWORDS = {
    "Hurricane Michael": [
        "MexicoBeach",
        "Mexico-Beach",
        "10132018",
        "10142018",
    ],
    "Hurricane Ian": [
        "Ft-Myers",
        "Myers",
        "Boone",
        "DIRT",
        "1001-Ft-Myers",
    ],
    "Hurricane Harvey": [
        "Pecan",
        "Westpark",
        "Sienna",
        "Lancaster",
        "Canyon-Gate",
        "DMS-Assessment",
        "090302",
        "090401",
        "090402",
        "090403",
    ],
    "Hurricane Idalia": [
        "Steinhatchee",
        "Jena",
        "20230830",
        "20230831",
    ],
    "Hurricane Laura": [
        "Laura",
        "Lake-Charles",
        "LakeCharles",
        "Louisiana",
    ],
    "Hurricane Ida": [
        "Ida",
        "JeanLafitte",
        "Lafitte",
        "GrandIsle",
        "Grand-Isle",
    ],
}


def classify_source(path: str) -> str:
    for source in ["UAS", "SATELLITE", "CREWED", "UAS_DSM"]:
        if f"/{source}/" in path:
            return source
    return "unknown"


def classify_category(path: str) -> str:
    if "/imagery/" in path:
        return "imagery"
    if "/annotations/" in path:
        return "annotations"
    if path.endswith(".csv"):
        return "csv"
    return "other"


def classify_annotation_type(path: str) -> str:
    if "/building_damage_assessment/" in path:
        return "building_damage_assessment"
    if "/building_alignment_adjustments/" in path:
        return "building_alignment_adjustments"
    return ""


def match_hurricane(path: str):
    matches = []

    for hurricane, keywords in HURRICANE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in path.lower():
                matches.append(hurricane)
                break

    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Filter CRASAR-U-DROIDs files to identify hurricane-related imagery and annotations."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/crasar_hurricane_filter"),
        help="Output directory.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

    all_rows = []
    hurricane_rows = []

    for path in files:
        matches = match_hurricane(path)

        row = {
            "path": path,
            "extension": Path(path).suffix.lower(),
            "source": classify_source(path),
            "category": classify_category(path),
            "annotation_type": classify_annotation_type(path),
            "matched_hurricanes": ";".join(matches),
            "is_hurricane_related": bool(matches),
        }

        all_rows.append(row)

        if matches:
            hurricane_rows.append(row)

    all_csv = args.output_dir / "crasar_all_files_inventory.csv"
    hurricane_csv = args.output_dir / "crasar_hurricane_files.csv"

    fieldnames = [
        "path",
        "extension",
        "source",
        "category",
        "annotation_type",
        "matched_hurricanes",
        "is_hurricane_related",
    ]

    with all_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    with hurricane_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(hurricane_rows)

    by_hurricane = {}
    by_source = {}
    by_category = {}

    for row in hurricane_rows:
        for hurricane in row["matched_hurricanes"].split(";"):
            if not hurricane:
                continue
            by_hurricane[hurricane] = by_hurricane.get(hurricane, 0) + 1

        by_source[row["source"]] = by_source.get(row["source"], 0) + 1
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1

    print("CRASAR hurricane file filter")
    print("----------------------------")
    print(f"Total files listed: {len(all_rows)}")
    print(f"Hurricane-related files matched: {len(hurricane_rows)}")
    print(f"By hurricane: {by_hurricane}")
    print(f"By source: {by_source}")
    print(f"By category: {by_category}")
    print(f"Saved full inventory to: {all_csv}")
    print(f"Saved hurricane file list to: {hurricane_csv}")
    print()
    print("Note: this is a filename/keyword-based filter and should be manually checked.")


if __name__ == "__main__":
    main()
