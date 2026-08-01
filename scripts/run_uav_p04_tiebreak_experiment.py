#!/usr/bin/env python3

"""Run the focused UAV-P04 ROI-margin tie-break experiment.

The experiment trains on the fixed historical TRAIN-30% buildings from
1001-San-Carlos-Island and 20210901-Cocodrie-1. It evaluates on one fixed,
deterministic, class-stratified 30% sample from two additional official TRAIN
orthomosaics: 1002-Ft-Myers-Beach-TFD and 20210902-LA-DIV-01.

Only three ROI treatments are compared: the current 0% bounding box, the
masked-building control, and +12.5% context. Five seeds are paired across the
three treatments. Internal test data and final-event data are never read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

try:
    from scripts import run_uav_p04_classifier_experiment as base
except ImportError:  # Support direct execution as ``python scripts/...py``.
    import run_uav_p04_classifier_experiment as base


EXPERIMENT_ID = "UAV-P04-CLASSIFIER-TIEBREAK"
CREATED_DATE = "2026-07-31"
DIRECTION = "current_two_to_new_two"
SUBSET = "train_30pct"
SELECTION_SALT = "uav-p04-tiebreak-evaluation-v1"
SOURCE_REPOSITORY = "CRASAR/CRASAR-U-DROIDs"
SOURCE_REVISION = "c3f5421e1167bb7e88a765e3afa38bebd9898479"

TRAIN_ORTHOMOSAICS = {
    "1001-San-Carlos-Island.geo.tif": base.EVENT_IAN,
    "20210901-Cocodrie-1.geo.tif": base.EVENT_IDA,
}
EVALUATION_ORTHOMOSAICS = {
    "1002-Ft-Myers-Beach-TFD.geo.tif": base.EVENT_IAN,
    "20210902-LA-DIV-01.geo.tif": base.EVENT_IDA,
}
EXPECTED_NEW_IMAGE_SIZES = {
    "1002-Ft-Myers-Beach-TFD.geo.tif": 3_536_113_635,
    "20210902-LA-DIV-01.geo.tif": 3_640_165_000,
}
EXPECTED_NEW_IMAGE_SHA256 = {
    "1002-Ft-Myers-Beach-TFD.geo.tif": "b741c319b874e498b1989d5ee9c93d919e15becae9dab86b64e60656133170bb",
    "20210902-LA-DIV-01.geo.tif": "35f6c7a41dacb99b5e0e796d153f4c7a2443f22f14468f5328b2aa0debf8c938",
}
EXPECTED_NEW_ANNOTATION_SIZES = {
    "1002-Ft-Myers-Beach-TFD.geo.tif": 1_289_511,
    "20210902-LA-DIV-01.geo.tif": 530_094,
}
EXPECTED_NEW_ANNOTATION_SHA256 = {
    "1002-Ft-Myers-Beach-TFD.geo.tif": "ba279479ef19c2bee7aecd9c71cfa52b9fbc4a62c311d424677eceb64e36b00c",
    "20210902-LA-DIV-01.geo.tif": "e7d5742d73a6b3889e6e4cc84ba85e20b172698aa2a516b035c50e4a7a72cbc9",
}
ALL_ORTHOMOSAICS = {**TRAIN_ORTHOMOSAICS, **EVALUATION_ORTHOMOSAICS}
EXPECTED_TRAIN_COUNT = 207
VARIANTS = (
    base.Variant("masked_building", "Masked building", None, True),
    base.margin_variant(0.0),
    base.margin_variant(0.125),
)
PRIMARY_METRIC = "equal_event_macro_f1"
EQUAL_EVENT_FIELDS = tuple(f"equal_event_{field}" for field in base.METRIC_FIELDS)
AGGREGATE_FIELDS = base.METRIC_FIELDS + EQUAL_EVENT_FIELDS
DUPLICATE_AUDIT_FIELDS = (
    "sample_id_1",
    "role_1",
    "event_1",
    "orthomosaic_1",
    "sample_id_2",
    "role_2",
    "event_2",
    "orthomosaic_2",
    "exact_duplicate",
    "phash_distance",
    "perceptual_candidate",
    "cross_split",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the focused leakage-audited UAV-P04 tie-break."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--evaluation-fraction", type=float, default=0.30)
    parser.add_argument("--seeds", default="17,29,43,59,71")
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one 1-epoch seed after all data and leakage checks.",
    )
    return parser.parse_args()


def annotation_path(data_root: Path, orthomosaic: str) -> Path:
    return (
        data_root
        / "train/annotations/UAS/building_damage_assessment"
        / f"{orthomosaic}.json"
    )


def imagery_path(data_root: Path, orthomosaic: str) -> Path:
    return data_root / "train/imagery/UAS" / orthomosaic


def load_json_records(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a JSON array of objects: {path}")
    return rows


def bounds_from_points(points: Sequence[tuple[int, int]]) -> base.Bounds:
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return base.Bounds(min(x_values), min(y_values), max(x_values), max(y_values))


def validate_record(
    record: dict[str, Any], orthomosaic: str, record_index: int
) -> None:
    required = ("id", "building_id", "label", "pixels")
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(
            f"Missing {missing} in {orthomosaic} annotation {record_index}."
        )


def sample_from_record(
    record: dict[str, Any],
    record_index: int,
    orthomosaic: str,
    event: str,
    role: str,
    manifest_bounds: base.Bounds | None = None,
) -> base.Sample:
    validate_record(record, orthomosaic, record_index)
    label = str(record["label"])
    if label not in base.LABEL_TO_ID:
        raise ValueError(f"Unexpected eligible label: {label}")
    points = base.points_from_record(record)
    return base.Sample(
        sample_id=str(record["id"]),
        building_id=str(record["building_id"]),
        record_index=record_index,
        event=event,
        orthomosaic=orthomosaic,
        label=label,
        class_id=base.LABEL_TO_ID[label],
        legacy_split="train" if role == "train" else "validation",
        subset_names=(SUBSET,) if role == "train" else (),
        points=points,
        base_bounds=manifest_bounds or bounds_from_points(points),
    )


def load_fixed_training_samples(
    manifest_path: Path, data_root: Path
) -> tuple[list[base.Sample], list[dict[str, str]]]:
    with manifest_path.open(newline="", encoding="utf-8") as source:
        all_rows = list(csv.DictReader(source))
    if any(row.get("legacy_split") == "test" for row in all_rows):
        raise ValueError(
            "The supplied source manifest contains internal-test rows. Use the "
            "leakage-safe UAV-P04 manifest, which contains no test rows."
        )
    selected_rows = [
        row
        for row in all_rows
        if row.get("legacy_split") == "train"
        and base.parse_bool(row.get("in_train_30pct", "False"))
    ]
    if len(selected_rows) != EXPECTED_TRAIN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_COUNT} fixed TRAIN-30% rows, found "
            f"{len(selected_rows)}."
        )
    if {row["orthomosaic"] for row in selected_rows} != set(TRAIN_ORTHOMOSAICS):
        raise ValueError(
            "The fixed training rows do not match the two approved sources."
        )

    requested: dict[str, set[int]] = defaultdict(set)
    for row in selected_rows:
        requested[row["orthomosaic"]].add(int(row["record_index"]))
    annotations = {
        orthomosaic: base.selected_json_array_records(
            annotation_path(data_root, orthomosaic), indices
        )
        for orthomosaic, indices in requested.items()
    }

    samples: list[base.Sample] = []
    for row in selected_rows:
        orthomosaic = row["orthomosaic"]
        record_index = int(row["record_index"])
        record = annotations[orthomosaic][record_index]
        for field, expected in (
            ("id", row["sample_id"]),
            ("building_id", row["building_id"]),
            ("label", row["label"]),
        ):
            if str(record.get(field)) != expected:
                raise ValueError(
                    f"Manifest/annotation mismatch for {row['sample_id']}: {field}."
                )
        manifest_bounds = base.Bounds(
            int(row["polygon_min_x"]),
            int(row["polygon_min_y"]),
            int(row["polygon_max_x"]),
            int(row["polygon_max_y"]),
        )
        samples.append(
            sample_from_record(
                record,
                record_index,
                orthomosaic,
                TRAIN_ORTHOMOSAICS[orthomosaic],
                "train",
                manifest_bounds,
            )
        )
    return sorted(samples, key=lambda sample: sample.sample_id), selected_rows


def load_evaluation_candidates(data_root: Path) -> list[base.Sample]:
    candidates: list[base.Sample] = []
    for orthomosaic, event in EVALUATION_ORTHOMOSAICS.items():
        records = load_json_records(annotation_path(data_root, orthomosaic))
        with rasterio.open(imagery_path(data_root, orthomosaic)) as source:
            for record_index, record in enumerate(records):
                if str(record.get("label")) not in base.LABEL_TO_ID:
                    continue
                points = base.points_from_record(record)
                raw_bounds = bounds_from_points(points)
                clipped_bounds = base.Bounds(
                    max(0, raw_bounds.min_x),
                    max(0, raw_bounds.min_y),
                    min(source.width - 1, raw_bounds.max_x),
                    min(source.height - 1, raw_bounds.max_y),
                )
                if clipped_bounds.width < 1 or clipped_bounds.height < 1:
                    raise ValueError(
                        f"Annotation {record_index} is outside {orthomosaic}."
                    )
                candidates.append(
                    sample_from_record(
                        record,
                        record_index,
                        orthomosaic,
                        event,
                        "evaluation",
                        clipped_bounds,
                    )
                )
    return sorted(candidates, key=lambda sample: sample.sample_id)


def calculate_alpha_invalid_fractions(
    samples: Sequence[base.Sample], image_paths: dict[str, Path]
) -> dict[str, float]:
    sources = {
        orthomosaic: rasterio.open(path) for orthomosaic, path in image_paths.items()
    }
    fractions: dict[str, float] = {}
    try:
        for index, sample in enumerate(samples, start=1):
            fractions[sample.sample_id] = base.polygon_invalid_fraction(
                sample, sources[sample.orthomosaic]
            )
            if index % 100 == 0 or index == len(samples):
                print(f"alpha audit: {index}/{len(samples)}", flush=True)
    finally:
        for source in sources.values():
            source.close()
    return fractions


def deterministic_score(sample: base.Sample) -> str:
    value = f"{SELECTION_SALT}:{sample.orthomosaic}:{sample.label}:{sample.sample_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rounded_fraction_count(count: int, fraction: float) -> int:
    return int(math.floor(count * fraction + 0.5))


def select_stratified_evaluation(
    candidates: Sequence[base.Sample], fraction: float
) -> tuple[list[base.Sample], dict[str, dict[str, int]]]:
    if not 0 < fraction < 1:
        raise ValueError("Evaluation fraction must be between zero and one.")
    grouped: dict[tuple[str, str], list[base.Sample]] = defaultdict(list)
    for sample in candidates:
        grouped[(sample.orthomosaic, sample.label)].append(sample)

    selected: list[base.Sample] = []
    selection_metadata: dict[str, dict[str, int]] = {}
    expected_groups = {
        (orthomosaic, label)
        for orthomosaic in EVALUATION_ORTHOMOSAICS
        for label in base.LABELS
    }
    if set(grouped) != expected_groups:
        raise ValueError("Every evaluation orthomosaic must contain all four classes.")
    for key, samples in sorted(grouped.items()):
        ordered = sorted(
            samples, key=lambda sample: (deterministic_score(sample), sample.sample_id)
        )
        target = rounded_fraction_count(len(ordered), fraction)
        if target < 1:
            raise ValueError(f"Evaluation group {key} would be empty.")
        for rank, sample in enumerate(ordered[:target], start=1):
            selected.append(sample)
            selection_metadata[sample.sample_id] = {
                "selection_rank_within_orthomosaic_class": rank,
                "eligible_count_within_orthomosaic_class": len(ordered),
                "selected_count_within_orthomosaic_class": target,
            }
    return sorted(selected, key=lambda sample: sample.sample_id), selection_metadata


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def assign_groups(values_by_id: dict[str, str], prefix: str) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id, value in values_by_id.items():
        grouped[value].append(sample_id)
    assigned: dict[str, str] = {}
    ordered_groups = sorted(grouped.values(), key=lambda members: min(members))
    for index, members in enumerate(ordered_groups, start=1):
        for sample_id in members:
            assigned[sample_id] = f"{prefix}_{index:04d}"
    return assigned


def build_leakage_audit(
    train_samples: Sequence[base.Sample],
    evaluation_samples: Sequence[base.Sample],
    image_paths: dict[str, Path],
    alpha_fractions: dict[str, float],
    input_size: int,
    selection_metadata: dict[str, dict[str, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    active_samples = sorted(
        [*train_samples, *evaluation_samples], key=lambda sample: sample.sample_id
    )
    ids = [sample.sample_id for sample in active_samples]
    building_ids = [sample.building_id for sample in active_samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs exist in the active comparison.")
    if len(building_ids) != len(set(building_ids)):
        raise ValueError("Duplicate building IDs exist in the active comparison.")

    sources = {
        orthomosaic: rasterio.open(path) for orthomosaic, path in image_paths.items()
    }
    exact_hashes: dict[str, str] = {}
    perceptual_hashes: dict[str, str] = {}
    bbox = base.margin_variant(0.0)
    try:
        for index, sample in enumerate(active_samples, start=1):
            image = base.render_model_input(
                sample, bbox, sources[sample.orthomosaic], input_size
            )
            exact_hashes[sample.sample_id] = hashlib.sha256(image.tobytes()).hexdigest()
            perceptual_hashes[sample.sample_id] = base.compute_phash_array(image)
            if index % 100 == 0 or index == len(active_samples):
                print(
                    f"duplicate audit inputs: {index}/{len(active_samples)}", flush=True
                )
    finally:
        for source in sources.values():
            source.close()

    union_find = UnionFind(ids)
    sample_by_id = {sample.sample_id: sample for sample in active_samples}
    role_by_id = {
        sample.sample_id: ("train" if sample in train_samples else "evaluation")
        for sample in active_samples
    }
    duplicate_rows: list[dict[str, Any]] = []
    for left_index, left_id in enumerate(ids):
        for right_id in ids[left_index + 1 :]:
            exact = exact_hashes[left_id] == exact_hashes[right_id]
            distance = base.hamming_distance(
                perceptual_hashes[left_id], perceptual_hashes[right_id]
            )
            perceptual = distance <= base.PHASH_THRESHOLD
            if not (exact or perceptual):
                continue
            union_find.union(left_id, right_id)
            duplicate_rows.append(
                {
                    "sample_id_1": left_id,
                    "role_1": role_by_id[left_id],
                    "event_1": sample_by_id[left_id].event,
                    "orthomosaic_1": sample_by_id[left_id].orthomosaic,
                    "sample_id_2": right_id,
                    "role_2": role_by_id[right_id],
                    "event_2": sample_by_id[right_id].event,
                    "orthomosaic_2": sample_by_id[right_id].orthomosaic,
                    "exact_duplicate": exact,
                    "phash_distance": distance,
                    "perceptual_candidate": perceptual,
                    "cross_split": role_by_id[left_id] != role_by_id[right_id],
                }
            )

    root_by_id = {sample_id: union_find.find(sample_id) for sample_id in ids}
    perceptual_groups = assign_groups(root_by_id, "phash_group")
    exact_groups = assign_groups(exact_hashes, "exact_group")
    train_ids = {sample.sample_id for sample in train_samples}

    manifest_rows: list[dict[str, Any]] = []
    for sample in active_samples:
        role = "train" if sample.sample_id in train_ids else "evaluation"
        metadata = selection_metadata.get(sample.sample_id, {})
        manifest_rows.append(
            {
                "sample_id": sample.sample_id,
                "building_id": sample.building_id,
                "record_index": sample.record_index,
                "role": role,
                "official_source_split": "Train",
                "event": sample.event,
                "orthomosaic": sample.orthomosaic,
                "label": sample.label,
                "class_id": sample.class_id,
                "in_fixed_train_30pct": role == "train",
                "in_fixed_evaluation_30pct": role == "evaluation",
                "polygon_min_x": sample.base_bounds.min_x,
                "polygon_min_y": sample.base_bounds.min_y,
                "polygon_max_x": sample.base_bounds.max_x,
                "polygon_max_y": sample.base_bounds.max_y,
                "orthomosaic_group": f"orthomosaic::{sample.orthomosaic}",
                "flight_group": f"orthomosaic_surrogate::{sample.orthomosaic}",
                "sequence_group": f"orthomosaic_surrogate::{sample.orthomosaic}",
                "exact_duplicate_group": exact_groups[sample.sample_id],
                "perceptual_duplicate_group": perceptual_groups[sample.sample_id],
                "exact_hash_sha256": exact_hashes[sample.sample_id],
                "phash64": perceptual_hashes[sample.sample_id],
                "polygon_alpha_invalid_fraction": alpha_fractions[sample.sample_id],
                **metadata,
            }
        )

    overlap_fields = (
        "sample_id",
        "building_id",
        "orthomosaic_group",
        "flight_group",
        "sequence_group",
        "exact_duplicate_group",
        "perceptual_duplicate_group",
    )
    train_rows = [row for row in manifest_rows if row["role"] == "train"]
    eval_rows = [row for row in manifest_rows if row["role"] == "evaluation"]
    overlaps: dict[str, list[str]] = {}
    for field in overlap_fields:
        overlaps[field] = sorted(
            {str(row[field]) for row in train_rows}
            & {str(row[field]) for row in eval_rows}
        )
    cross_split_pairs = [row for row in duplicate_rows if row["cross_split"]]
    audit = {
        "status": "pass"
        if all(not values for values in overlaps.values())
        else "blocked",
        "train_count": len(train_rows),
        "evaluation_count": len(eval_rows),
        "train_orthomosaics": sorted(TRAIN_ORTHOMOSAICS),
        "evaluation_orthomosaics": sorted(EVALUATION_ORTHOMOSAICS),
        "overlap_counts": {field: len(values) for field, values in overlaps.items()},
        "overlap_values": overlaps,
        "exact_duplicate_pair_count": sum(
            bool(row["exact_duplicate"]) for row in duplicate_rows
        ),
        "perceptual_candidate_pair_count": sum(
            bool(row["perceptual_candidate"]) for row in duplicate_rows
        ),
        "cross_split_duplicate_candidate_count": len(cross_split_pairs),
        "phash_threshold": base.PHASH_THRESHOLD,
        "native_flight_sequence_ids_available": False,
        "flight_sequence_surrogate": "orthomosaic",
        "internal_test_rows_read": 0,
        "internal_test_annotation_objects_read": 0,
        "internal_test_pixel_reads": 0,
        "final_event_reads": 0,
    }
    return manifest_rows, audit, duplicate_rows


def write_duplicate_audit(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if rows:
        base.write_csv(path, rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(DUPLICATE_AUDIT_FIELDS) + "\n", encoding="utf-8")


def validate_class_counts(samples: Sequence[base.Sample], role: str) -> dict[str, Any]:
    counts = Counter(sample.label for sample in samples)
    if set(counts) != set(base.LABELS) or any(
        counts[label] < 1 for label in base.LABELS
    ):
        raise ValueError(f"{role} does not contain all four classes: {dict(counts)}")
    return {label: counts[label] for label in base.LABELS}


def metrics_from_prediction_rows(
    prediction_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    labels = np.asarray([int(row["true_class_id"]) for row in prediction_rows])
    predictions = np.asarray(
        [int(row["predicted_class_id"]) for row in prediction_rows]
    )
    probabilities = np.asarray(
        [
            [
                float(row[f"probability_{label.replace(' ', '_')}"])
                for label in base.LABELS
            ]
            for row in prediction_rows
        ]
    )
    return base.evaluate_predictions(labels, predictions, probabilities)


def event_metrics_for_run(
    run_row: dict[str, Any], prediction_rows: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for event in (base.EVENT_IAN, base.EVENT_IDA):
        selected = [row for row in prediction_rows if row["event"] == event]
        metrics, matrix = metrics_from_prediction_rows(selected)
        event_row = {
            "run_id": run_row["run_id"],
            "variant": run_row["variant"],
            "seed": run_row["seed"],
            "event": event,
            "evaluation_count": len(selected),
            **metrics,
        }
        event_rows.append(event_row)
        for true_index, true_label in enumerate(base.LABELS):
            for predicted_index, predicted_label in enumerate(base.LABELS):
                matrix_rows.append(
                    {
                        "run_id": run_row["run_id"],
                        "variant": run_row["variant"],
                        "seed": run_row["seed"],
                        "event": event,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(matrix[true_index, predicted_index]),
                    }
                )
    for metric in base.METRIC_FIELDS:
        run_row[f"equal_event_{metric}"] = statistics.fmean(
            float(row[metric]) for row in event_rows
        )
    return event_rows, matrix_rows


def persist_raw_results(
    output_dir: Path,
    run_rows: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
    curve_rows: Sequence[dict[str, Any]],
    matrix_rows: Sequence[dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
    event_matrix_rows: Sequence[dict[str, Any]],
) -> None:
    base.write_csv(output_dir / "run_metrics.csv", run_rows)
    base.write_csv(output_dir / "predictions.csv", prediction_rows)
    base.write_csv(output_dir / "training_curves.csv", curve_rows)
    base.write_csv(output_dir / "confusion_matrices.csv", matrix_rows)
    base.write_csv(output_dir / "event_metrics.csv", event_rows)
    base.write_csv(output_dir / "event_confusion_matrices.csv", event_matrix_rows)


def aggregate_runs(run_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    by_variant = defaultdict(list)
    by_variant_seed = {}
    for row in run_rows:
        by_variant[row["variant"]].append(row)
        by_variant_seed[(row["variant"], int(row["seed"]))] = row
    for variant in VARIANTS:
        rows = by_variant[variant.variant_id]
        output: dict[str, Any] = {
            "variant": variant.variant_id,
            "variant_name": variant.display_name,
            "margin_fraction": variant.margin_fraction,
            "masked": variant.masked,
            "seed_count": len(rows),
        }
        for metric in AGGREGATE_FIELDS:
            summary = base.summarize_values([float(row[metric]) for row in rows])
            for statistic, value in summary.items():
                if statistic != "seed_count":
                    output[f"{metric}_{statistic}"] = value
        differences = [
            float(row[PRIMARY_METRIC])
            - float(by_variant_seed[("margin_0", int(row["seed"]))][PRIMARY_METRIC])
            for row in rows
        ]
        paired = base.summarize_values(differences)
        for statistic, value in paired.items():
            if statistic != "seed_count":
                output[f"{PRIMARY_METRIC}_diff_vs_margin_0_{statistic}"] = value
        result.append(output)
    baseline = next(row for row in result if row["variant"] == "margin_0")
    for output in result:
        output["major_recall_difference_vs_margin_0"] = (
            output["equal_event_recall_major_damage_mean"]
            - baseline["equal_event_recall_major_damage_mean"]
        )
        output["destroyed_recall_difference_vs_margin_0"] = (
            output["equal_event_recall_destroyed_mean"]
            - baseline["equal_event_recall_destroyed_mean"]
        )
        output["minority_recall_guardrail_passed"] = (
            output["major_recall_difference_vs_margin_0"] >= -base.RELEVANT_RECALL_DROP
            and output["destroyed_recall_difference_vs_margin_0"]
            >= -base.RELEVANT_RECALL_DROP
        )
    return result


def paired_comparisons(run_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["variant"], int(row["seed"])): row for row in run_rows}
    pairs = (
        ("masked_building", "margin_0"),
        ("margin_12p5", "margin_0"),
        ("margin_12p5", "masked_building"),
    )
    output: list[dict[str, Any]] = []
    seeds = sorted({int(row["seed"]) for row in run_rows})
    for left, right in pairs:
        for metric in (
            PRIMARY_METRIC,
            "equal_event_recall_major_damage",
            "equal_event_recall_destroyed",
        ):
            differences = [
                float(lookup[(left, seed)][metric])
                - float(lookup[(right, seed)][metric])
                for seed in seeds
            ]
            summary = base.summarize_values(differences)
            output.append(
                {
                    "left_variant": left,
                    "right_variant": right,
                    "metric": metric,
                    **summary,
                    "statistically_indistinguishable": summary["ci95_low"]
                    <= 0
                    <= summary["ci95_high"],
                    "paired_seed_wins": sum(value > 0 for value in differences),
                    "paired_seed_ties": sum(value == 0 for value in differences),
                }
            )
    return output


def aggregate_event_results(
    event_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[(row["variant"], row["event"])].append(row)
    output: list[dict[str, Any]] = []
    for (variant, event), rows in sorted(grouped.items()):
        result: dict[str, Any] = {
            "variant": variant,
            "event": event,
            "seed_count": len(rows),
            "evaluation_count": int(rows[0]["evaluation_count"]),
        }
        for metric in base.METRIC_FIELDS:
            summary = base.summarize_values([float(row[metric]) for row in rows])
            for statistic, value in summary.items():
                if statistic != "seed_count":
                    result[f"{metric}_{statistic}"] = value
        output.append(result)
    return output


def choose_result(
    aggregate_rows: Sequence[dict[str, Any]],
    paired_rows: Sequence[dict[str, Any]],
    event_aggregate_rows: Sequence[dict[str, Any]],
    run_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_variant = {row["variant"]: row for row in aggregate_rows}
    raw_best = max(aggregate_rows, key=lambda row: row[f"{PRIMARY_METRIC}_mean"])
    candidates_beating_baseline: list[dict[str, Any]] = []
    for row in aggregate_rows:
        if row["variant"] == "margin_0" or not row["minority_recall_guardrail_passed"]:
            continue
        paired = next(
            comparison
            for comparison in paired_rows
            if comparison["left_variant"] == row["variant"]
            and comparison["right_variant"] == "margin_0"
            and comparison["metric"] == PRIMARY_METRIC
        )
        if paired["ci95_low"] > 0:
            candidates_beating_baseline.append(row)
    selected = (
        max(candidates_beating_baseline, key=lambda row: row[f"{PRIMARY_METRIC}_mean"])
        if candidates_beating_baseline
        else by_variant["margin_0"]
    )

    event_winners = {}
    for event in (base.EVENT_IAN, base.EVENT_IDA):
        candidates = [row for row in event_aggregate_rows if row["event"] == event]
        event_winners[event] = max(candidates, key=lambda row: row["macro_f1_mean"])[
            "variant"
        ]
    seed_winners = {}
    for seed in sorted({int(row["seed"]) for row in run_rows}):
        candidates = [row for row in run_rows if int(row["seed"]) == seed]
        seed_winners[str(seed)] = max(
            candidates, key=lambda row: float(row[PRIMARY_METRIC])
        )["variant"]

    reliable_added_improvement = selected["variant"] != "margin_0"
    stable_across_events = set(event_winners.values()) == {selected["variant"]}
    stable_across_seeds = set(seed_winners.values()) == {selected["variant"]}
    if reliable_added_improvement and stable_across_events and stable_across_seeds:
        decision = "KEEP"
        conclusion = "STABLE EMPIRICAL WINNER AMONG THE THREE TESTED VARIANTS"
    elif not reliable_added_improvement:
        decision = "KEEP"
        conclusion = "NO ADDED ROI TREATMENT PROVED BETTER; RETAIN THE 0% BASELINE"
    else:
        decision = "NEEDS FULL DATA"
        conclusion = "NO STABLE WINNER WITH THE AVAILABLE ORTHOMOSAICS"
    return {
        "raw_best_mean_variant": raw_best["variant"],
        "raw_best_mean_equal_event_macro_f1": raw_best[f"{PRIMARY_METRIC}_mean"],
        "selected_variant": selected["variant"],
        "selected_variant_equal_event_macro_f1": selected[f"{PRIMARY_METRIC}_mean"],
        "decision": decision,
        "empirical_conclusion": conclusion,
        "selection_rule": (
            "An added treatment must have a positive paired 95% CI versus 0% "
            "for equal-event macro F1 and must not reduce equal-event major or "
            "destroyed recall by more than 0.05. Otherwise retain 0%."
        ),
        "event_winners": event_winners,
        "seed_winners": seed_winners,
        "selected_variant_wins_both_events": stable_across_events,
        "selected_variant_wins_all_five_seeds": stable_across_seeds,
        "reliable_added_improvement": reliable_added_improvement,
        "best_margin_scope": "best among 0%, masked building, and +12.5%; not a continuous optimum",
    }


def aggregate_roi_diagnostics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        role = "train" if row["legacy_split"] == "train" else "evaluation"
        grouped[(row["variant"], role)].append(row)
    output: list[dict[str, Any]] = []
    for (variant, role), selected in sorted(grouped.items()):
        result: dict[str, Any] = {
            "variant": variant,
            "role": role,
            "sample_count": len(selected),
        }
        for field in base.ROI_DIAGNOSTIC_FIELDS:
            values = [float(row[field]) for row in selected]
            result[f"{field}_mean"] = statistics.fmean(values)
            result[f"{field}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        output.append(result)
    return output


def create_result_figure(
    path: Path,
    run_rows: Sequence[dict[str, Any]],
    aggregate_rows: Sequence[dict[str, Any]],
    paired_rows: Sequence[dict[str, Any]],
) -> None:
    display = {variant.variant_id: variant.display_name for variant in VARIANTS}
    order = ["margin_0", "masked_building", "margin_12p5"]
    colors = {
        "margin_0": "#35618f",
        "masked_building": "#8b8f97",
        "margin_12p5": "#c28b2c",
    }
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    axis = axes[0]
    for index, variant in enumerate(order):
        values = [
            float(row[PRIMARY_METRIC]) for row in run_rows if row["variant"] == variant
        ]
        summary = base.summarize_values(values)
        axis.scatter(
            [index] * len(values), values, color=colors[variant], alpha=0.65, zorder=3
        )
        axis.errorbar(
            index,
            summary["mean"],
            yerr=[
                [summary["mean"] - summary["ci95_low"]],
                [summary["ci95_high"] - summary["mean"]],
            ],
            fmt="o",
            color="#20252b",
            capsize=5,
            zorder=4,
        )
    axis.set_xticks(range(len(order)), [display[value] for value in order])
    axis.set_ylabel("Equal-event macro F1")
    axis.set_title("Classification performance")
    axis.grid(axis="y", color="#d9dde2", linewidth=0.8)

    axis = axes[1]
    comparisons = [
        row
        for row in paired_rows
        if row["right_variant"] == "margin_0" and row["metric"] == PRIMARY_METRIC
    ]
    for index, row in enumerate(comparisons):
        axis.errorbar(
            row["mean"],
            index,
            xerr=[[row["mean"] - row["ci95_low"]], [row["ci95_high"] - row["mean"]]],
            fmt="o",
            color=colors[row["left_variant"]],
            capsize=5,
        )
    axis.axvline(0, color="#20252b", linewidth=1)
    axis.set_yticks(
        range(len(comparisons)), [display[row["left_variant"]] for row in comparisons]
    )
    axis.set_xlabel("Paired macro F1 difference vs 0%")
    axis.set_title("Paired improvement (95% CI)")
    axis.grid(axis="x", color="#d9dde2", linewidth=0.8)

    axis = axes[2]
    positions = np.arange(len(order))
    width = 0.35
    major = [
        next(row for row in aggregate_rows if row["variant"] == variant)[
            "equal_event_recall_major_damage_mean"
        ]
        for variant in order
    ]
    destroyed = [
        next(row for row in aggregate_rows if row["variant"] == variant)[
            "equal_event_recall_destroyed_mean"
        ]
        for variant in order
    ]
    axis.bar(positions - width / 2, major, width, label="Major damage", color="#35618f")
    axis.bar(
        positions + width / 2, destroyed, width, label="Destroyed", color="#c28b2c"
    )
    axis.set_xticks(positions, [display[value] for value in order])
    axis.set_ylabel("Equal-event recall")
    axis.set_ylim(0, 1)
    axis.set_title("Severe-class recall")
    axis.legend(frameon=False, fontsize=9)
    axis.grid(axis="y", color="#d9dde2", linewidth=0.8)

    figure.suptitle(
        "UAV-P04 focused tie-break: five paired seeds, fixed TRAIN-30% and evaluation",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_visual_comparison(
    path: Path,
    evaluation_samples: Sequence[base.Sample],
    image_cache: dict[str, dict[str, np.ndarray]],
) -> list[str]:
    chosen = []
    event_for_label = {
        "no damage": base.EVENT_IAN,
        "minor damage": base.EVENT_IAN,
        "major damage": base.EVENT_IDA,
        "destroyed": base.EVENT_IDA,
    }
    for label in base.LABELS:
        candidates = sorted(
            [
                sample
                for sample in evaluation_samples
                if sample.label == label and sample.event == event_for_label[label]
            ],
            key=lambda sample: (deterministic_score(sample), sample.sample_id),
        )
        chosen.append(candidates[0])
    columns = ["margin_0", "masked_building", "margin_12p5"]
    display = {variant.variant_id: variant.display_name for variant in VARIANTS}
    figure, axes = plt.subplots(len(chosen), len(columns), figsize=(9, 11))
    for row_index, sample in enumerate(chosen):
        for column_index, variant in enumerate(columns):
            axis = axes[row_index, column_index]
            axis.imshow(image_cache[variant][sample.sample_id])
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(display[variant])
            if column_index == 0:
                axis.set_ylabel(sample.label.title())
    figure.suptitle("UAV-P04: identical evaluation buildings under each ROI treatment")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return [sample.sample_id for sample in chosen]


def create_confusion_figure(
    path: Path, event_matrix_rows: Sequence[dict[str, Any]]
) -> None:
    order = ["margin_0", "masked_building", "margin_12p5"]
    display = {variant.variant_id: variant.display_name for variant in VARIANTS}
    matrices: dict[tuple[str, str, int], np.ndarray] = defaultdict(
        lambda: np.zeros((len(base.LABELS), len(base.LABELS)), dtype=np.float64)
    )
    for row in event_matrix_rows:
        key = (str(row["variant"]), str(row["event"]), int(row["seed"]))
        true_index = base.LABEL_TO_ID[str(row["true_label"])]
        predicted_index = base.LABEL_TO_ID[str(row["predicted_label"])]
        matrices[key][true_index, predicted_index] = float(row["count"])

    figure, axes = plt.subplots(
        1, len(order), figsize=(14.5, 4.4), sharex=True, sharey=True
    )
    image = None
    for axis, variant in zip(axes, order):
        normalized = []
        for key, matrix in matrices.items():
            if key[0] != variant:
                continue
            row_totals = matrix.sum(axis=1, keepdims=True)
            normalized.append(
                np.divide(
                    matrix, row_totals, out=np.zeros_like(matrix), where=row_totals > 0,
                )
            )
        equal_event_seed_matrix = np.mean(normalized, axis=0)
        image = axis.imshow(equal_event_seed_matrix, vmin=0, vmax=1, cmap="Blues")
        for true_index in range(len(base.LABELS)):
            for predicted_index in range(len(base.LABELS)):
                value = equal_event_seed_matrix[true_index, predicted_index]
                axis.text(
                    predicted_index,
                    true_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "#20252b",
                    fontsize=9,
                )
        axis.set_title(display[variant])
        axis.set_xticks(
            range(len(base.LABELS)),
            ("No", "Minor", "Major", "Destroyed"),
            rotation=35,
            ha="right",
        )
        axis.set_yticks(range(len(base.LABELS)), ("No", "Minor", "Major", "Destroyed"))
        axis.set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    color_axis = figure.add_axes((0.925, 0.20, 0.015, 0.62))
    figure.colorbar(image, cax=color_axis, label="Row-normalized share")
    figure.suptitle(
        "UAV-P04 confusion matrices: equal weight per event and seed", fontsize=14
    )
    figure.subplots_adjust(left=0.07, right=0.89, bottom=0.22, top=0.84, wspace=0.22)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def validate_artifacts(
    run_rows: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
    curve_rows: Sequence[dict[str, Any]],
    matrix_rows: Sequence[dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
    event_matrix_rows: Sequence[dict[str, Any]],
    manifest_rows: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    epochs: int,
) -> dict[str, Any]:
    expected_keys = {
        (variant.variant_id, seed) for variant in VARIANTS for seed in seeds
    }
    actual_keys = {(row["variant"], int(row["seed"])) for row in run_rows}
    run_ids = {row["run_id"] for row in run_rows}
    predictions_per_run = Counter(row["run_id"] for row in prediction_rows)
    curves_per_run = Counter(row["run_id"] for row in curve_rows)
    matrices_per_run = Counter(row["run_id"] for row in matrix_rows)
    events_per_run = Counter(row["run_id"] for row in event_rows)
    event_matrices_per_run = Counter(row["run_id"] for row in event_matrix_rows)
    initialization_hashes: dict[int, set[str]] = defaultdict(set)
    for row in run_rows:
        initialization_hashes[int(row["seed"])].add(row["initialization_hash"])
    checks = {
        "complete_15_run_grid": actual_keys == expected_keys,
        "unique_run_ids": len(run_ids) == len(run_rows),
        "same_initialization_within_each_paired_seed": all(
            len(values) == 1 for values in initialization_hashes.values()
        ),
        "prediction_count_matches_fixed_evaluation": all(
            predictions_per_run[row["run_id"]] == int(row["eval_count"])
            for row in run_rows
        ),
        "fixed_epoch_count": all(
            curves_per_run[run_id] == epochs for run_id in run_ids
        ),
        "sixteen_confusion_cells_per_run": all(
            matrices_per_run[run_id] == len(base.LABELS) ** 2 for run_id in run_ids
        ),
        "two_event_metric_rows_per_run": all(
            events_per_run[run_id] == 2 for run_id in run_ids
        ),
        "thirty_two_event_confusion_cells_per_run": all(
            event_matrices_per_run[run_id] == 2 * len(base.LABELS) ** 2
            for run_id in run_ids
        ),
        "manifest_has_zero_test_rows": all(
            row["official_source_split"] == "Train" for row in manifest_rows
        ),
        "manifest_uses_exactly_four_approved_orthomosaics": {
            row["orthomosaic"] for row in manifest_rows
        }
        == set(ALL_ORTHOMOSAICS),
    }
    result = {
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
        "counts": {
            "runs": len(run_rows),
            "predictions": len(prediction_rows),
            "training_curve_rows": len(curve_rows),
            "confusion_matrix_rows": len(matrix_rows),
            "event_metric_rows": len(event_rows),
            "event_confusion_matrix_rows": len(event_matrix_rows),
            "manifest_rows": len(manifest_rows),
        },
        "missing_run_keys": sorted(expected_keys - actual_keys),
        "unexpected_run_keys": sorted(actual_keys - expected_keys),
    }
    if result["status"] != "pass":
        raise RuntimeError(json.dumps(base.json_ready(result), sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    if args.data_root.name != base.DATASET_NAME:
        raise ValueError(f"--data-root must be the {base.DATASET_NAME} directory.")
    seeds = base.parse_seeds(args.seeds)
    if not args.smoke_test and len(seeds) != 5:
        raise ValueError("The tie-break requires exactly five paired seeds.")
    if args.smoke_test:
        seeds = seeds[:1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    image_paths = {
        orthomosaic: imagery_path(args.data_root, orthomosaic)
        for orthomosaic in ALL_ORTHOMOSAICS
    }
    required_paths = [
        *image_paths.values(),
        *(
            annotation_path(args.data_root, orthomosaic)
            for orthomosaic in ALL_ORTHOMOSAICS
        ),
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing approved local source files: " + ", ".join(missing)
        )
    wrong_sizes = {
        orthomosaic: {
            "actual": image_paths[orthomosaic].stat().st_size,
            "expected": expected,
        }
        for orthomosaic, expected in EXPECTED_NEW_IMAGE_SIZES.items()
        if image_paths[orthomosaic].stat().st_size != expected
    }
    if wrong_sizes:
        raise ValueError(
            "The new GeoTIFF download is incomplete or does not match the approved "
            f"source revision: {wrong_sizes}"
        )
    wrong_annotation_sizes = {
        orthomosaic: {
            "actual": annotation_path(args.data_root, orthomosaic).stat().st_size,
            "expected": expected,
        }
        for orthomosaic, expected in EXPECTED_NEW_ANNOTATION_SIZES.items()
        if annotation_path(args.data_root, orthomosaic).stat().st_size != expected
    }
    if wrong_annotation_sizes:
        raise ValueError(
            "The new annotation download does not match the approved source "
            f"revision: {wrong_annotation_sizes}"
        )

    train_samples, _ = load_fixed_training_samples(args.source_manifest, args.data_root)
    evaluation_candidates = load_evaluation_candidates(args.data_root)
    all_alpha_fractions = calculate_alpha_invalid_fractions(
        [*train_samples, *evaluation_candidates], image_paths
    )
    invalid_train = [
        sample.sample_id
        for sample in train_samples
        if all_alpha_fractions[sample.sample_id] > 0.50
    ]
    if invalid_train:
        raise RuntimeError(
            "The fixed training subset changed under the confirmed alpha rule: "
            + ", ".join(invalid_train)
        )
    valid_evaluation_candidates = [
        sample
        for sample in evaluation_candidates
        if all_alpha_fractions[sample.sample_id] <= 0.50
    ]
    rejected_evaluation_candidates = [
        sample
        for sample in evaluation_candidates
        if all_alpha_fractions[sample.sample_id] > 0.50
    ]
    evaluation_samples, selection_metadata = select_stratified_evaluation(
        valid_evaluation_candidates, args.evaluation_fraction
    )
    train_counts = validate_class_counts(train_samples, "training")
    evaluation_counts = validate_class_counts(evaluation_samples, "evaluation")

    manifest_rows, leakage_audit, duplicate_rows = build_leakage_audit(
        train_samples,
        evaluation_samples,
        image_paths,
        all_alpha_fractions,
        args.input_size,
        selection_metadata,
    )
    base.write_csv(args.output_dir / "uav_p04_tiebreak_manifest.csv", manifest_rows)
    write_duplicate_audit(args.output_dir / "duplicate_audit_pairs.csv", duplicate_rows)
    (args.output_dir / "leakage_audit.json").write_text(
        json.dumps(base.json_ready(leakage_audit), indent=2) + "\n", encoding="utf-8"
    )
    if leakage_audit["status"] != "pass":
        raise RuntimeError("Leakage audit blocked training; see leakage_audit.json.")

    config = base.ExperimentConfig(
        input_size=args.input_size,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        initial_seeds=tuple(seeds),
        final_seeds=tuple(seeds),
        normalization="uint8 / 127.5 - 1.0",
        augmentation="none",
        loss="cross_entropy",
        class_weighting="n / (4 * class_count), computed from fixed TRAIN-30%",
        optimizer="AdamW",
        scheduler="none",
        checkpoint_policy="final epoch; no early stopping",
        phash_threshold=base.PHASH_THRESHOLD,
        relevant_recall_drop=base.RELEVANT_RECALL_DROP,
        torch_version=torch.__version__,
        device="cpu",
    )
    configuration = {
        "experiment_id": EXPERIMENT_ID,
        "created_date": CREATED_DATE,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "config": asdict(config),
        "architecture": {
            "name": "SmallDamageCNN",
            "reference": "identical to UAV-P04-CLASSIFIER-LOOO",
            "initialization": "PyTorch seeded default initialization, paired by seed",
            "pretrained_weights": False,
            "all_weights_trainable": True,
            "input_size": args.input_size,
        },
        "comparison": {
            "variants": [asdict(variant) for variant in VARIANTS],
            "training_subset": "fixed historical TRAIN-30% from the two original orthomosaics",
            "evaluation_subset": (
                "fixed deterministic class-stratified 30% from each of the two new "
                "official-Train orthomosaics after UAV-P01 alpha rejection"
            ),
            "evaluation_selection_salt": SELECTION_SALT,
            "evaluation_fraction": args.evaluation_fraction,
            "primary_metric": PRIMARY_METRIC,
            "event_weighting": "Ian and Ida receive equal weight",
        },
        "roi_rendering": {
            "margin_definition": (
                "add margin_fraction * bbox width to left/right and margin_fraction * "
                "bbox height to top/bottom"
            ),
            "resize": "bilinear RGB; nearest alpha",
            "aspect_ratio": "preserved",
            "square_padding": "symmetric black",
            "invalid_pixels": "alpha == 0 RGB black-filled",
            "masked_control": "0% box with pixels outside polygon black",
            "global_autocontrast": False,
            "median_filter": False,
        },
        "counts": {
            "train_total": len(train_samples),
            "train_by_class": train_counts,
            "evaluation_eligible_before_alpha": len(evaluation_candidates),
            "evaluation_rejected_by_alpha": len(rejected_evaluation_candidates),
            "evaluation_total": len(evaluation_samples),
            "evaluation_by_class": evaluation_counts,
            "evaluation_by_event": dict(
                Counter(sample.event for sample in evaluation_samples)
            ),
        },
        "source_files": {
            orthomosaic: {
                "imagery": str(image_paths[orthomosaic]),
                "annotation": str(annotation_path(args.data_root, orthomosaic)),
                "file_size_bytes": image_paths[orthomosaic].stat().st_size,
                **(
                    {
                        "verified_sha256": EXPECTED_NEW_IMAGE_SHA256[orthomosaic],
                        "annotation_verified_sha256": EXPECTED_NEW_ANNOTATION_SHA256[
                            orthomosaic
                        ],
                    }
                    if orthomosaic in EXPECTED_NEW_IMAGE_SHA256
                    else {}
                ),
            }
            for orthomosaic in ALL_ORTHOMOSAICS
        },
        "test_access": {
            "internal_test_manifest_rows_read": 0,
            "internal_test_annotation_objects_read": 0,
            "internal_test_pixel_reads": 0,
            "final_event_reads": 0,
        },
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(base.json_ready(configuration), indent=2) + "\n", encoding="utf-8"
    )

    original_direction = base.DIRECTIONS.get(DIRECTION)
    base.DIRECTIONS[DIRECTION] = {
        "train_event": "Hurricane Ian + Hurricane Ida",
        "eval_event": "Hurricane Ian + Hurricane Ida",
    }
    run_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    event_matrix_rows: list[dict[str, Any]] = []
    roi_rows: list[dict[str, Any]] = []
    image_cache: dict[str, dict[str, np.ndarray]] = {}
    active_samples = sorted(
        [*train_samples, *evaluation_samples], key=lambda sample: sample.sample_id
    )
    experiment_start = time.perf_counter()
    try:
        for variant in VARIANTS:
            images, diagnostics = base.render_variant_images(
                variant, active_samples, image_paths, config.input_size
            )
            image_cache[variant.variant_id] = images
            roi_rows.extend(diagnostics)
            for seed in seeds:
                run_id = f"{variant.variant_id}__{SUBSET}__{DIRECTION}__seed{seed}"
                print(f"RUN {run_id}", flush=True)
                run_row, predictions, curves, matrices = base.train_and_evaluate(
                    run_id=run_id,
                    variant=variant,
                    subset_name=SUBSET,
                    direction=DIRECTION,
                    seed=seed,
                    train_samples=train_samples,
                    eval_samples=evaluation_samples,
                    images_by_id=images,
                    config=config,
                    phase="tiebreak",
                )
                per_event, per_event_matrices = event_metrics_for_run(
                    run_row, predictions
                )
                run_rows.append(run_row)
                prediction_rows.extend(predictions)
                curve_rows.extend(curves)
                matrix_rows.extend(matrices)
                event_rows.extend(per_event)
                event_matrix_rows.extend(per_event_matrices)
                persist_raw_results(
                    args.output_dir,
                    run_rows,
                    prediction_rows,
                    curve_rows,
                    matrix_rows,
                    event_rows,
                    event_matrix_rows,
                )
                print(
                    f"DONE {run_id} equal_event_macro_f1="
                    f"{run_row[PRIMARY_METRIC]:.4f}",
                    flush=True,
                )
    finally:
        if original_direction is None:
            base.DIRECTIONS.pop(DIRECTION, None)
        else:
            base.DIRECTIONS[DIRECTION] = original_direction

    base.write_csv(args.output_dir / "roi_diagnostics.csv", roi_rows)
    base.write_csv(
        args.output_dir / "roi_diagnostic_summary.csv",
        aggregate_roi_diagnostics(roi_rows),
    )
    if args.smoke_test:
        print("SMOKE TEST COMPLETE", flush=True)
        return

    aggregate_rows = aggregate_runs(run_rows)
    paired_rows = paired_comparisons(run_rows)
    event_aggregate_rows = aggregate_event_results(event_rows)
    selection = choose_result(
        aggregate_rows, paired_rows, event_aggregate_rows, run_rows
    )
    base.write_csv(args.output_dir / "margin_comparison.csv", aggregate_rows)
    base.write_csv(args.output_dir / "paired_differences.csv", paired_rows)
    base.write_csv(args.output_dir / "event_comparison.csv", event_aggregate_rows)

    result_figure = args.assets_dir / "UAV-P04_tiebreak_classifier-results.png"
    visual_figure = args.assets_dir / "UAV-P04_tiebreak_visual-comparison.png"
    confusion_figure = args.assets_dir / "UAV-P04_tiebreak_confusion-matrices.png"
    create_result_figure(result_figure, run_rows, aggregate_rows, paired_rows)
    visual_ids = create_visual_comparison(
        visual_figure, evaluation_samples, image_cache
    )
    create_confusion_figure(confusion_figure, event_matrix_rows)
    artifact_validation = validate_artifacts(
        run_rows,
        prediction_rows,
        curve_rows,
        matrix_rows,
        event_rows,
        event_matrix_rows,
        manifest_rows,
        seeds,
        config.epochs,
    )
    (args.output_dir / "artifact_validation.json").write_text(
        json.dumps(base.json_ready(artifact_validation), indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "decision": selection["decision"],
        "selection": selection,
        "counts": configuration["counts"],
        "seeds": list(seeds),
        "runs": len(run_rows),
        "failed_runs": 0,
        "leakage_audit": leakage_audit,
        "artifact_validation": artifact_validation,
        "test_access": configuration["test_access"],
        "visual_sample_ids": visual_ids,
        "execution": {
            "duration_seconds": time.perf_counter() - experiment_start,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "limitations": [
            "The fixed training subset contains 20 destroyed buildings.",
            "Only four development orthomosaics from two hurricane events are used.",
            "Evaluation is orthomosaic-held-out, not disaster-event-held-out.",
            "The reference CNN is trained from scratch with a 96x96 input.",
            "Native flight/sequence IDs are unavailable; orthomosaic is the surrogate group.",
            "The result compares only 0%, masked building, and +12.5%; it is not a continuous optimum.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(base.json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    print("EXPERIMENT COMPLETE", flush=True)
    print(json.dumps(base.json_ready(selection), indent=2), flush=True)


if __name__ == "__main__":
    main()
