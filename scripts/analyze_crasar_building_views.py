#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "CRASAR/CRASAR-U-DROIDs"
REPO_TYPE = "dataset"


def list_bda_json_files(source: str = "all") -> list:
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

    selected = []

    for path in files:
        if path.startswith("format/"):
            continue

        if not path.endswith(".json"):
            continue

        if "/annotations/" not in path:
            continue

        if "/building_damage_assessment/" not in path:
            continue

        if source != "all" and f"/{source}/" not in path:
            continue

        selected.append(path)

    return sorted(selected)


def load_json_from_hf(repo_file: str, cache_dir: Path):
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=repo_file,
        cache_dir=str(cache_dir),
    )

    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze repeated building IDs and view IDs in CRASAR-U-DROIDs."
    )

    parser.add_argument(
        "--source",
        default="UAS",
        choices=["all", "UAS", "SATELLITE", "CREWED", "UAS_DSM"],
        help="Imagery source to inspect. Default: UAS.",
    )

    parser.add_argument(
        "--max-json",
        type=int,
        default=20,
        help="Maximum number of annotation JSON files to inspect.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/crasar_building_view_analysis"),
        help="Output directory.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = list_bda_json_files(source=args.source)
    files_to_load = files[: args.max_json]

    print(f"Source filter: {args.source}")
    print(f"Building damage assessment JSON files found: {len(files)}")
    print(f"JSON files selected for sample analysis: {len(files_to_load)}")

    records = []

    for repo_file in files_to_load:
        print(f"Loading: {repo_file}")

        try:
            data = load_json_from_hf(repo_file, cache_dir=cache_dir)
        except Exception as exc:
            print(f"Failed to load {repo_file}: {exc}")
            continue

        if not isinstance(data, list):
            print(f"Skipping {repo_file}: expected list, got {type(data)}")
            continue

        for entry in data:
            if not isinstance(entry, dict):
                continue

            pixels = entry.get("pixels")
            boundary = entry.get("boundary")

            record = {
                "building_id": entry.get("building_id", ""),
                "view_id": entry.get("view_id", ""),
                "label": entry.get("label", ""),
                "source": entry.get("source", ""),
                "filename": entry.get("filename", ""),
                "annotation_file": repo_file,
                "num_pixels_points": len(pixels) if isinstance(pixels, list) else "",
                "num_boundary_points": len(boundary) if isinstance(boundary, list) else "",
            }

            records.append(record)

    building_to_records = defaultdict(list)

    for record in records:
        building_id = record["building_id"]
        if building_id:
            building_to_records[building_id].append(record)

    repeated_buildings = {
        building_id: recs
        for building_id, recs in building_to_records.items()
        if len(recs) > 1
    }

    repeated_summary_rows = []

    for building_id, recs in repeated_buildings.items():
        labels = sorted({r["label"] for r in recs if r["label"]})
        views = sorted({r["view_id"] for r in recs if r["view_id"]})
        sources = sorted({r["source"] for r in recs if r["source"]})
        filenames = sorted({r["filename"] for r in recs if r["filename"]})

        repeated_summary_rows.append({
            "building_id": building_id,
            "num_records": len(recs),
            "num_unique_views": len(views),
            "num_unique_labels": len(labels),
            "labels": ";".join(labels),
            "sources": ";".join(sources),
            "num_filenames": len(filenames),
        })

    records_csv = args.output_dir / f"crasar_{args.source}_building_view_records.csv"
    repeated_csv = args.output_dir / f"crasar_{args.source}_repeated_buildings.csv"
    summary_json = args.output_dir / f"crasar_{args.source}_building_view_summary.json"

    if records:
        with records_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    if repeated_summary_rows:
        with repeated_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(repeated_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(repeated_summary_rows)

    label_counts = Counter(r["label"] for r in records)
    source_counts = Counter(r["source"] for r in records)

    repeated_with_multiple_views = 0
    repeated_with_multiple_labels = 0
    repeated_with_multiple_filenames = 0

    for recs in repeated_buildings.values():
        if len({r["view_id"] for r in recs if r["view_id"]}) > 1:
            repeated_with_multiple_views += 1

        if len({r["label"] for r in recs if r["label"]}) > 1:
            repeated_with_multiple_labels += 1

        if len({r["filename"] for r in recs if r["filename"]}) > 1:
            repeated_with_multiple_filenames += 1

    summary = {
        "repo_id": REPO_ID,
        "source_filter": args.source,
        "json_files_found": len(files),
        "json_files_loaded": len(files_to_load),
        "annotation_records": len(records),
        "unique_buildings": len(building_to_records),
        "repeated_buildings": len(repeated_buildings),
        "repeated_buildings_with_multiple_views": repeated_with_multiple_views,
        "repeated_buildings_with_multiple_labels": repeated_with_multiple_labels,
        "repeated_buildings_with_multiple_filenames": repeated_with_multiple_filenames,
        "label_counts": dict(label_counts),
        "source_counts": dict(source_counts),
        "records_csv": str(records_csv),
        "repeated_buildings_csv": str(repeated_csv),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nCRASAR building-view summary")
    print("----------------------------")
    print(f"Annotation records: {len(records)}")
    print(f"Unique buildings: {len(building_to_records)}")
    print(f"Repeated buildings: {len(repeated_buildings)}")
    print(f"Repeated buildings with multiple views: {repeated_with_multiple_views}")
    print(f"Repeated buildings with multiple labels: {repeated_with_multiple_labels}")
    print(f"Repeated buildings with multiple filenames: {repeated_with_multiple_filenames}")
    print(f"Label counts: {dict(label_counts)}")
    print(f"Saved records CSV to: {records_csv}")
    print(f"Saved repeated buildings CSV to: {repeated_csv}")
    print(f"Saved summary JSON to: {summary_json}")


if __name__ == "__main__":
    main()
