#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "CRASAR/CRASAR-U-DROIDs"
REPO_TYPE = "dataset"


def classify_file(path: str) -> dict:
    parts = path.split("/")
    ext = Path(path).suffix.lower()

    split = parts[0] if parts and parts[0] in {"train", "test"} else "root"

    source = "unknown"
    for candidate in ["UAS", "UAS_DSM", "SATELLITE", "CREWED"]:
        if candidate in parts:
            source = candidate

    category = "other"
    if "imagery" in parts:
        category = "imagery"
    elif "annotations" in parts:
        category = "annotations"
    elif path.endswith("statistics.csv"):
        category = "statistics"

    annotation_type = "none"
    if "building_damage_assessment" in parts:
        annotation_type = "building_damage_assessment"
    elif "building_alignment_adjustments" in parts:
        annotation_type = "building_alignment_adjustments"

    return {
        "path": path,
        "split": split,
        "source": source,
        "category": category,
        "annotation_type": annotation_type,
        "extension": ext,
    }


def safe_load_json(repo_file: str, cache_dir: Path):
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
        description="Inspect CRASAR-U-DROIDs file structure and sample building annotations without downloading the full dataset."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/crasar_inspection"))
    parser.add_argument("--max-json", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()

    print(f"Listing files from Hugging Face dataset: {REPO_ID}")
    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Total files listed: {len(files)}")

    inventory_rows = [classify_file(path) for path in files]

    inventory_csv = args.output_dir / "crasar_file_inventory.csv"
    with inventory_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "split", "source", "category", "annotation_type", "extension"],
        )
        writer.writeheader()
        writer.writerows(inventory_rows)

    print(f"Saved file inventory to: {inventory_csv}")

    ext_counts = Counter(row["extension"] for row in inventory_rows)
    split_counts = Counter(row["split"] for row in inventory_rows)
    source_counts = Counter(row["source"] for row in inventory_rows)
    category_counts = Counter(row["category"] for row in inventory_rows)
    annotation_type_counts = Counter(row["annotation_type"] for row in inventory_rows)

    bda_json_files = [
        row["path"]
        for row in inventory_rows
        if row["annotation_type"] == "building_damage_assessment"
        and row["extension"] == ".json"
    ]

    print("\nFile summary")
    print("------------")
    print(f"Extensions: {dict(ext_counts)}")
    print(f"Splits: {dict(split_counts)}")
    print(f"Sources: {dict(source_counts)}")
    print(f"Categories: {dict(category_counts)}")
    print(f"Annotation types: {dict(annotation_type_counts)}")
    print(f"Building damage assessment JSON files: {len(bda_json_files)}")

    sample_files = bda_json_files[: args.max_json]

    label_counts = Counter()
    building_counts = Counter()
    view_ids = set()
    building_to_views = defaultdict(list)
    keys_seen = Counter()
    entries_total = 0
    files_loaded = []

    for repo_file in sample_files:
        print(f"\nDownloading sample annotation JSON: {repo_file}")
        try:
            data = safe_load_json(repo_file, cache_dir=cache_dir)
        except Exception as exc:
            print(f"Failed to load {repo_file}: {exc}")
            continue

        files_loaded.append(repo_file)

        if not isinstance(data, list):
            print(f"Skipping {repo_file}: expected list, got {type(data)}")
            continue

        for entry in data:
            if not isinstance(entry, dict):
                continue

            entries_total += 1
            keys_seen.update(entry.keys())

            building_id = entry.get("building_id")
            view_id = entry.get("view_id")
            label = entry.get("label")
            boundary = entry.get("boundary")

            if building_id:
                building_counts[building_id] += 1
            if view_id:
                view_ids.add(view_id)
            if label:
                label_counts[label] += 1
            if building_id and view_id:
                building_to_views[building_id].append({
                    "view_id": view_id,
                    "label": label,
                    "boundary": boundary,
                    "file": repo_file,
                })

    repeated_buildings = {
        bid: count for bid, count in building_counts.items() if count > 1
    }

    sample_summary = {
        "repo_id": REPO_ID,
        "total_files_listed": len(files),
        "inventory_csv": str(inventory_csv),
        "bda_json_files_found": len(bda_json_files),
        "sample_json_files_loaded": files_loaded,
        "sample_entries_total": entries_total,
        "sample_unique_buildings": len(building_counts),
        "sample_unique_views": len(view_ids),
        "sample_repeated_buildings": len(repeated_buildings),
        "sample_label_counts": dict(label_counts),
        "sample_keys_seen": dict(keys_seen),
        "top_repeated_buildings": building_counts.most_common(10),
        "extension_counts": dict(ext_counts),
        "split_counts": dict(split_counts),
        "source_counts": dict(source_counts),
        "category_counts": dict(category_counts),
        "annotation_type_counts": dict(annotation_type_counts),
    }

    summary_path = args.output_dir / "crasar_sample_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(sample_summary, f, indent=2)

    print("\nSample annotation summary")
    print("-------------------------")
    print(f"Sample entries total: {entries_total}")
    print(f"Unique building IDs in sample: {len(building_counts)}")
    print(f"Unique view IDs in sample: {len(view_ids)}")
    print(f"Repeated building IDs in sample: {len(repeated_buildings)}")
    print(f"Label counts in sample: {dict(label_counts)}")
    print(f"Keys seen: {dict(keys_seen)}")
    print(f"Saved sample summary to: {summary_path}")

    print("\nInterpretation hint")
    print("-------------------")
    print("If repeated building IDs appear with multiple view IDs, then CRASAR supports building-level redundancy analysis.")
    print("The key fields to look for are building_id, view_id, label, pixels, EPSG:4326, and boundary.")


if __name__ == "__main__":
    main()
