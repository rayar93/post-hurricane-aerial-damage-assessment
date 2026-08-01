#!/usr/bin/env python3

"""Run the empirical UAV-P04 classifier margin experiment.

This experiment uses a bidirectional leave-one-orthomosaic-out design:

* train on historical Ian TRAIN subset IDs and evaluate on fixed Ida VAL IDs;
* train on historical Ida TRAIN subset IDs and evaluate on fixed Ian VAL IDs.

Internal-test rows, final-event data, source-image copies, crops, and model
checkpoints are never materialized. The only comparable run difference is the
ROI treatment. All other model and training settings are frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.windows import Window
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn


DATASET_NAME = "crasar_uas_preprocessing_dev_v1"
EVENT_IAN = "Hurricane Ian"
EVENT_IDA = "Hurricane Ida"
ORTHOMOSAIC_IAN = "1001-San-Carlos-Island.geo.tif"
ORTHOMOSAIC_IDA = "20210901-Cocodrie-1.geo.tif"
ALLOWED_ORTHOMOSAICS = {
    ORTHOMOSAIC_IAN: EVENT_IAN,
    ORTHOMOSAIC_IDA: EVENT_IDA,
}
LABELS = ("no damage", "minor damage", "major damage", "destroyed")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
SUBSET_FLAGS = {
    "train_10pct": "in_train_10pct",
    "train_20pct": "in_train_20pct",
    "train_30pct": "in_train_30pct",
}
EXPECTED_TRAIN_COUNTS = {
    "ian_to_ida": {
        "train_10pct": 58,
        "train_20pct": 116,
        "train_30pct": 174,
    },
    "ida_to_ian": {
        "train_10pct": 11,
        "train_20pct": 22,
        "train_30pct": 33,
    },
}
EXPECTED_EVAL_COUNTS = {"ian_to_ida": 22, "ida_to_ian": 116}
DIRECTIONS = {
    "ian_to_ida": {"train_event": EVENT_IAN, "eval_event": EVENT_IDA},
    "ida_to_ian": {"train_event": EVENT_IDA, "eval_event": EVENT_IAN},
}
INITIAL_VARIANTS = (
    ("masked_building", "Masked building", None, True),
    ("margin_0", "0%", 0.0, False),
    ("margin_5", "+5%", 0.05, False),
    ("margin_10", "+10%", 0.10, False),
    ("margin_15", "+15%", 0.15, False),
    ("margin_20", "+20%", 0.20, False),
    ("margin_25", "+25%", 0.25, False),
    ("margin_35", "+35%", 0.35, False),
    ("margin_50", "+50%", 0.50, False),
)
PHASH_THRESHOLD = 6
RELEVANT_RECALL_DROP = 0.05
IGNORE_INTERNAL_TEST = True
PIL_LANCZOS = (
    Image.Resampling.LANCZOS
    if hasattr(Image, "Resampling")
    else Image.LANCZOS
)


@dataclass(frozen=True)
class Bounds:
    min_x: int
    min_y: int
    max_x: int
    max_y: int

    @property
    def width(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def height(self) -> int:
        return self.max_y - self.min_y + 1

    def window(self) -> Window:
        return Window(self.min_x, self.min_y, self.width, self.height)


@dataclass(frozen=True)
class Variant:
    variant_id: str
    display_name: str
    margin_fraction: float | None
    masked: bool


@dataclass(frozen=True)
class Sample:
    sample_id: str
    building_id: str
    record_index: int
    event: str
    orthomosaic: str
    label: str
    class_id: int
    legacy_split: str
    subset_names: tuple[str, ...]
    points: tuple[tuple[int, int], ...]
    base_bounds: Bounds


@dataclass(frozen=True)
class ExperimentConfig:
    input_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    dropout: float
    initial_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    normalization: str
    augmentation: str
    loss: str
    class_weighting: str
    optimizer: str
    scheduler: str
    checkpoint_policy: str
    phash_threshold: int
    relevant_recall_drop: float
    torch_version: str
    device: str


class SmallDamageCNN(nn.Module):
    """Compact all-trainable reference CNN for controlled ROI comparisons."""

    def __init__(self, dropout: float = 0.20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(2, 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, len(LABELS)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_left] = root_right


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-audited UAV-P04 classifier experiments."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--initial-seeds", default="17,29,43")
    parser.add_argument("--final-seeds", default="17,29,43,59,71")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one combination and do not select a margin.",
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=1,
        help="Epoch count for the single-combination smoke test.",
    )
    return parser.parse_args()


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")
    return seeds


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Expected True or False, got {value!r}")
    return normalized == "true"


def variant_from_tuple(values: tuple[str, str, float | None, bool]) -> Variant:
    return Variant(*values)


def margin_variant(fraction: float) -> Variant:
    percentage = 100 * fraction
    text = f"{percentage:g}".replace(".", "p")
    return Variant(
        variant_id=f"margin_{text}",
        display_name=f"+{percentage:g}%" if fraction > 0 else "0%",
        margin_fraction=fraction,
        masked=False,
    )


def resolve_source_path(data_root: Path, manifest_path: str) -> Path:
    parts = Path(manifest_path).parts
    if DATASET_NAME in parts:
        dataset_index = parts.index(DATASET_NAME)
        return data_root.joinpath(*parts[dataset_index + 1 :])
    return data_root.joinpath(*parts)


def load_development_rows(
    manifest_path: Path,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    retained: list[dict[str, str]] = []
    counts = Counter()
    with manifest_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        clean_manifest = "legacy_split" in (reader.fieldnames or ())
        for row in reader:
            if clean_manifest:
                row = dict(row)
                row["id"] = row["sample_id"]
                row["split"] = row["legacy_split"]
            split = row["split"].strip()
            counts[f"legacy_{split}_rows"] += 1
            if split == "test":
                counts["internal_test_rows_excluded_before_selection"] += 1
                continue
            if split == "train" and not parse_bool(row["in_train_30pct"]):
                counts["train_rows_above_30pct_excluded"] += 1
                continue
            retained.append(row)
    return retained, dict(counts)


def selected_json_array_records(
    path: Path, requested_indices: set[int]
) -> dict[int, dict[str, Any]]:
    """Decode only requested top-level objects from a monolithic JSON list."""
    if not requested_indices:
        return {}
    selected: dict[int, dict[str, Any]] = {}
    object_index = -1
    brace_depth = 0
    in_string = False
    escaped = False
    capturing = False
    buffer: list[str] = []
    with path.open(encoding="utf-8") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            for character in chunk:
                if in_string:
                    if capturing:
                        buffer.append(character)
                    if escaped:
                        escaped = False
                    elif character == "\\":
                        escaped = True
                    elif character == '"':
                        in_string = False
                    continue
                if character == '"':
                    in_string = True
                    if capturing:
                        buffer.append(character)
                    continue
                if character == "{":
                    if brace_depth == 0:
                        object_index += 1
                        capturing = object_index in requested_indices
                        buffer = []
                    brace_depth += 1
                    if capturing:
                        buffer.append(character)
                    continue
                if character == "}":
                    if capturing:
                        buffer.append(character)
                    brace_depth -= 1
                    if brace_depth < 0:
                        raise ValueError(f"Malformed JSON object nesting in {path}")
                    if brace_depth == 0 and capturing:
                        record = json.loads("".join(buffer))
                        if not isinstance(record, dict):
                            raise ValueError(
                                f"Expected object at index {object_index} in {path}"
                            )
                        selected[object_index] = record
                        capturing = False
                        buffer = []
                        if set(selected) == requested_indices:
                            return selected
                    continue
                if capturing:
                    buffer.append(character)
    missing = sorted(requested_indices - set(selected))
    if missing:
        raise IndexError(f"Missing requested annotation indices in {path}: {missing}")
    return selected


def load_selected_annotations(
    rows: Sequence[dict[str, str]], data_root: Path
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Path]]:
    requested_indices: dict[str, set[int]] = defaultdict(set)
    annotation_paths: dict[str, Path] = {}
    for row in rows:
        orthomosaic = row["orthomosaic"]
        if orthomosaic not in ALLOWED_ORTHOMOSAICS:
            raise ValueError(f"Unapproved orthomosaic: {orthomosaic}")
        requested_indices[orthomosaic].add(int(row["record_index"]))
        if row.get("annotation_path"):
            path = resolve_source_path(data_root, row["annotation_path"])
        else:
            path = (
                data_root
                / "train/annotations/UAS/building_damage_assessment"
                / f"{orthomosaic}.json"
            )
        existing = annotation_paths.setdefault(orthomosaic, path)
        if existing != path:
            raise ValueError(f"Multiple annotation paths for {orthomosaic}")

    selected: dict[str, dict[int, dict[str, Any]]] = {}
    for orthomosaic, path in sorted(annotation_paths.items()):
        if not path.is_file():
            raise FileNotFoundError(path)
        selected[orthomosaic] = selected_json_array_records(
            path, requested_indices[orthomosaic]
        )
    return selected, annotation_paths


def image_paths_from_rows(
    rows: Sequence[dict[str, str]], data_root: Path
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for row in rows:
        orthomosaic = row["orthomosaic"]
        if row.get("imagery_path"):
            path = resolve_source_path(data_root, row["imagery_path"])
        else:
            path = data_root / "train/imagery/UAS" / orthomosaic
        existing = paths.setdefault(orthomosaic, path)
        if existing != path:
            raise ValueError(f"Multiple image paths for {orthomosaic}")
    if set(paths) != set(ALLOWED_ORTHOMOSAICS):
        raise ValueError("Expected exactly the two approved UAV orthomosaics.")
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def points_from_record(record: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    raw_points = record.get("pixels")
    if not isinstance(raw_points, list) or len(raw_points) < 3:
        raise ValueError("Annotation polygon has fewer than three points.")
    return tuple(
        (int(round(float(point["x"]))), int(round(float(point["y"]))))
        for point in raw_points
    )


def build_samples(
    rows: Sequence[dict[str, str]],
    annotations: dict[str, dict[int, dict[str, Any]]],
) -> list[Sample]:
    samples: list[Sample] = []
    seen_ids: set[str] = set()
    for row in rows:
        sample_id = row["id"]
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample ID: {sample_id}")
        seen_ids.add(sample_id)
        orthomosaic = row["orthomosaic"]
        event = row["event"]
        if ALLOWED_ORTHOMOSAICS.get(orthomosaic) != event:
            raise ValueError(f"Event/orthomosaic mismatch for {sample_id}")
        record_index = int(row["record_index"])
        record = annotations[orthomosaic][record_index]
        for field in ("id", "building_id", "label"):
            if str(record.get(field, "")) != row[field]:
                raise ValueError(
                    f"Manifest/annotation mismatch for {sample_id}: {field}"
                )
        label = row["label"]
        if label not in LABEL_TO_ID:
            raise ValueError(f"Unexpected label for {sample_id}: {label}")
        points = points_from_record(record)
        point_x = [point[0] for point in points]
        point_y = [point[1] for point in points]
        base_bounds = Bounds(
            int(row["polygon_min_x"])
            if row.get("polygon_min_x")
            else min(point_x),
            int(row["polygon_min_y"])
            if row.get("polygon_min_y")
            else min(point_y),
            int(row["polygon_max_x"])
            if row.get("polygon_max_x")
            else max(point_x),
            int(row["polygon_max_y"])
            if row.get("polygon_max_y")
            else max(point_y),
        )
        subset_names = tuple(
            subset_name
            for subset_name, flag in SUBSET_FLAGS.items()
            if row["split"] == "train" and parse_bool(row[flag])
        )
        samples.append(
            Sample(
                sample_id=sample_id,
                building_id=row["building_id"],
                record_index=record_index,
                event=event,
                orthomosaic=orthomosaic,
                label=label,
                class_id=LABEL_TO_ID[label],
                legacy_split=row["split"],
                subset_names=subset_names,
                points=points,
                base_bounds=base_bounds,
            )
        )
    return sorted(samples, key=lambda sample: sample.sample_id)


def validate_sample_counts(samples: Sequence[Sample]) -> None:
    for direction, definition in DIRECTIONS.items():
        train_event = definition["train_event"]
        eval_event = definition["eval_event"]
        for subset_name, expected in EXPECTED_TRAIN_COUNTS[direction].items():
            count = sum(
                sample.legacy_split == "train"
                and sample.event == train_event
                and subset_name in sample.subset_names
                for sample in samples
            )
            if count != expected:
                raise ValueError(
                    f"{direction}/{subset_name}: {count} train rows; "
                    f"expected {expected}."
                )
        eval_count = sum(
            sample.legacy_split == "val" and sample.event == eval_event
            for sample in samples
        )
        if eval_count != EXPECTED_EVAL_COUNTS[direction]:
            raise ValueError(
                f"{direction}: {eval_count} evaluation rows; expected "
                f"{EXPECTED_EVAL_COUNTS[direction]}."
            )


def expand_bounds(
    bounds: Bounds,
    margin_fraction: float,
    image_width: int,
    image_height: int,
) -> Bounds:
    margin_x = int(math.ceil(bounds.width * margin_fraction))
    margin_y = int(math.ceil(bounds.height * margin_fraction))
    return Bounds(
        min_x=max(0, bounds.min_x - margin_x),
        min_y=max(0, bounds.min_y - margin_y),
        max_x=min(image_width - 1, bounds.max_x + margin_x),
        max_y=min(image_height - 1, bounds.max_y + margin_y),
    )


def alpha_band_index(source: rasterio.io.DatasetReader) -> int:
    for index, interpretation in enumerate(source.colorinterp, start=1):
        if interpretation.name.lower() == "alpha":
            return index
    if source.count >= 4:
        return 4
    raise ValueError(f"{source.name} has no alpha band.")


def local_polygon_points(
    sample: Sample, bounds: Bounds, output_width: int, output_height: int
) -> list[tuple[float, float]]:
    return [
        (
            (x - bounds.min_x) * output_width / bounds.width,
            (y - bounds.min_y) * output_height / bounds.height,
        )
        for x, y in sample.points
    ]


def render_model_input(
    sample: Sample,
    variant: Variant,
    source: rasterio.io.DatasetReader,
    input_size: int,
) -> np.ndarray:
    image, _ = render_model_input_with_diagnostics(
        sample, variant, source, input_size
    )
    return image


def render_model_input_with_diagnostics(
    sample: Sample,
    variant: Variant,
    source: rasterio.io.DatasetReader,
    input_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    margin = 0.0 if variant.margin_fraction is None else variant.margin_fraction
    bounds = expand_bounds(sample.base_bounds, margin, source.width, source.height)
    side = max(bounds.width, bounds.height)
    output_width = max(1, int(round(input_size * bounds.width / side)))
    output_height = max(1, int(round(input_size * bounds.height / side)))
    rgb = source.read(
        [1, 2, 3],
        window=bounds.window(),
        out_shape=(3, output_height, output_width),
        resampling=Resampling.bilinear,
    ).transpose(1, 2, 0)
    alpha = source.read(
        alpha_band_index(source),
        window=bounds.window(),
        out_shape=(output_height, output_width),
        resampling=Resampling.nearest,
    )
    rgb = np.asarray(rgb, dtype=np.uint8)
    rgb[alpha == 0] = 0

    mask_image = Image.new("L", (output_width, output_height), 0)
    ImageDraw.Draw(mask_image).polygon(
        local_polygon_points(sample, bounds, output_width, output_height),
        fill=1,
    )
    building_mask = np.asarray(mask_image, dtype=bool)
    alpha_valid = alpha > 0
    if variant.masked:
        mask = building_mask
        rgb[~mask] = 0

    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    offset_x = (input_size - output_width) // 2
    offset_y = (input_size - output_height) // 2
    canvas[
        offset_y : offset_y + output_height,
        offset_x : offset_x + output_width,
    ] = rgb
    canvas_pixels = input_size * input_size
    content_pixels = output_width * output_height
    building_pixels = int(building_mask.sum())
    valid_building_pixels = int(np.logical_and(building_mask, alpha_valid).sum())
    invalid_building_pixels = building_pixels - valid_building_pixels
    valid_context_pixels = int(np.logical_and(~building_mask, alpha_valid).sum())
    invalid_context_pixels = content_pixels - building_pixels - valid_context_pixels
    visible_context_pixels = 0 if variant.masked else valid_context_pixels
    visible_alpha_invalid_pixels = (
        invalid_building_pixels
        if variant.masked
        else invalid_building_pixels + invalid_context_pixels
    )
    diagnostics = {
        "variant": variant.variant_id,
        "variant_name": variant.display_name,
        "margin_fraction": variant.margin_fraction,
        "masked": variant.masked,
        "sample_id": sample.sample_id,
        "building_id": sample.building_id,
        "event": sample.event,
        "orthomosaic": sample.orthomosaic,
        "label": sample.label,
        "legacy_split": sample.legacy_split,
        "roi_width_source_pixels": bounds.width,
        "roi_height_source_pixels": bounds.height,
        "roi_area_source_pixels": bounds.width * bounds.height,
        "rendered_content_pixels": content_pixels,
        "rendered_building_pixels": building_pixels,
        "valid_building_pixels": valid_building_pixels,
        "valid_context_pixels_before_masking": valid_context_pixels,
        "visible_context_pixels": visible_context_pixels,
        "alpha_invalid_pixels_in_roi": int((~alpha_valid).sum()),
        "visible_alpha_invalid_pixels": visible_alpha_invalid_pixels,
        "square_padding_pixels": canvas_pixels - content_pixels,
        "valid_building_fraction_canvas": valid_building_pixels / canvas_pixels,
        "visible_context_fraction_canvas": visible_context_pixels / canvas_pixels,
        "alpha_invalid_fraction_roi": float((~alpha_valid).mean()),
        "visible_alpha_invalid_fraction_canvas": (
            visible_alpha_invalid_pixels / canvas_pixels
        ),
        "square_padding_fraction_canvas": (canvas_pixels - content_pixels)
        / canvas_pixels,
        "building_alpha_invalid_fraction": (
            invalid_building_pixels / building_pixels if building_pixels else 0.0
        ),
    }
    return canvas, diagnostics


def polygon_invalid_fraction(
    sample: Sample, source: rasterio.io.DatasetReader
) -> float:
    bounds = sample.base_bounds
    alpha = source.read(alpha_band_index(source), window=bounds.window())
    mask_image = Image.new("L", (bounds.width, bounds.height), 0)
    relative_points = [
        (x - bounds.min_x, y - bounds.min_y) for x, y in sample.points
    ]
    ImageDraw.Draw(mask_image).polygon(relative_points, fill=1)
    mask = np.asarray(mask_image, dtype=bool)
    polygon_pixels = int(mask.sum())
    if polygon_pixels == 0:
        raise ValueError(f"Empty polygon mask for {sample.sample_id}")
    return float(np.logical_and(mask, alpha == 0).sum() / polygon_pixels)


def dct_matrix(size: int) -> np.ndarray:
    matrix = np.empty((size, size), dtype=np.float64)
    factor = np.pi / (2 * size)
    for frequency in range(size):
        alpha = math.sqrt(1 / size) if frequency == 0 else math.sqrt(2 / size)
        for position in range(size):
            matrix[frequency, position] = alpha * math.cos(
                (2 * position + 1) * frequency * factor
            )
    return matrix


PHASH_DCT = dct_matrix(32)


def compute_phash_array(image: np.ndarray, hash_size: int = 8) -> str:
    gray = np.asarray(Image.fromarray(image).convert("L"), dtype=np.float64)
    if gray.shape != (32, 32):
        gray = np.asarray(
            Image.fromarray(gray.astype(np.uint8)).resize(
                (32, 32), PIL_LANCZOS
            ),
            dtype=np.float64,
        )
    transformed = PHASH_DCT @ gray @ PHASH_DCT.T
    low_frequency = transformed[:hash_size, :hash_size].flatten()
    median = np.median(low_frequency[1:])
    bits = low_frequency > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_distance(left: str, right: str) -> int:
    return bin(int(left, 16) ^ int(right, 16)).count("1")


def build_leakage_manifest(
    samples: Sequence[Sample],
    image_paths: dict[str, Path],
    input_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    bbox_variant = margin_variant(0.0)
    hashes: dict[str, dict[str, Any]] = {}
    invalid_fractions: dict[str, float] = {}
    sources = {
        orthomosaic: rasterio.open(path)
        for orthomosaic, path in image_paths.items()
    }
    try:
        for index, sample in enumerate(samples, start=1):
            source = sources[sample.orthomosaic]
            invalid_fraction = polygon_invalid_fraction(sample, source)
            invalid_fractions[sample.sample_id] = invalid_fraction
            if invalid_fraction > 0.50:
                continue
            image = render_model_input(sample, bbox_variant, source, input_size)
            hashes[sample.sample_id] = {
                "exact_hash": hashlib.sha256(image.tobytes()).hexdigest(),
                "phash": compute_phash_array(image),
            }
            if index % 50 == 0 or index == len(samples):
                print(
                    f"leakage-audit inputs: {index}/{len(samples)}",
                    flush=True,
                )
    finally:
        for source in sources.values():
            source.close()

    rejected = [
        sample_id
        for sample_id, fraction in invalid_fractions.items()
        if fraction > 0.50
    ]
    if rejected:
        raise ValueError(
            "Active development samples exceeded the confirmed alpha-invalid "
            f"threshold: {rejected}"
        )

    sample_ids = sorted(hashes)
    union_find = UnionFind(sample_ids)
    duplicate_pairs: list[dict[str, Any]] = []
    cross_event_nearest: list[dict[str, Any]] = []
    sample_by_id = {sample.sample_id: sample for sample in samples}

    for left_index, left_id in enumerate(sample_ids):
        left_sample = sample_by_id[left_id]
        for right_id in sample_ids[left_index + 1 :]:
            right_sample = sample_by_id[right_id]
            distance = hamming_distance(
                hashes[left_id]["phash"], hashes[right_id]["phash"]
            )
            exact_duplicate = (
                hashes[left_id]["exact_hash"] == hashes[right_id]["exact_hash"]
            )
            perceptual_candidate = distance <= PHASH_THRESHOLD
            if exact_duplicate or perceptual_candidate:
                union_find.union(left_id, right_id)
                duplicate_pairs.append(
                    {
                        "sample_id_1": left_id,
                        "event_1": left_sample.event,
                        "sample_id_2": right_id,
                        "event_2": right_sample.event,
                        "exact_duplicate": exact_duplicate,
                        "phash_distance": distance,
                        "perceptual_candidate": perceptual_candidate,
                        "cross_event": left_sample.event != right_sample.event,
                    }
                )
            if left_sample.event != right_sample.event:
                cross_event_nearest.append(
                    {
                        "sample_id_1": left_id,
                        "event_1": left_sample.event,
                        "sample_id_2": right_id,
                        "event_2": right_sample.event,
                        "phash_distance": distance,
                    }
                )

    cross_event_nearest = sorted(
        cross_event_nearest,
        key=lambda row: (
            row["phash_distance"],
            row["sample_id_1"],
            row["sample_id_2"],
        ),
    )[:50]

    root_members: dict[str, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        root_members[union_find.find(sample_id)].append(sample_id)
    group_by_id: dict[str, str] = {}
    for group_index, members in enumerate(
        sorted(root_members.values(), key=lambda values: values[0]), start=1
    ):
        group_id = f"phash_group_{group_index:04d}"
        for sample_id in members:
            group_by_id[sample_id] = group_id

    exact_groups: dict[str, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        exact_groups[hashes[sample_id]["exact_hash"]].append(sample_id)
    exact_group_by_id: dict[str, str] = {}
    for group_index, members in enumerate(
        sorted(exact_groups.values(), key=lambda values: values[0]), start=1
    ):
        group_id = f"exact_group_{group_index:04d}"
        for sample_id in members:
            exact_group_by_id[sample_id] = group_id

    manifest_rows: list[dict[str, Any]] = []
    for sample in samples:
        if sample.sample_id not in hashes:
            continue
        manifest_rows.append(
            {
                "sample_id": sample.sample_id,
                "building_id": sample.building_id,
                "record_index": sample.record_index,
                "event": sample.event,
                "orthomosaic": sample.orthomosaic,
                "label": sample.label,
                "class_id": sample.class_id,
                "legacy_split": sample.legacy_split,
                "polygon_min_x": sample.base_bounds.min_x,
                "polygon_min_y": sample.base_bounds.min_y,
                "polygon_max_x": sample.base_bounds.max_x,
                "polygon_max_y": sample.base_bounds.max_y,
                "in_train_10pct": "train_10pct" in sample.subset_names,
                "in_train_20pct": "train_20pct" in sample.subset_names,
                "in_train_30pct": "train_30pct" in sample.subset_names,
                "ian_to_ida_role": role_for_direction(sample, "ian_to_ida"),
                "ida_to_ian_role": role_for_direction(sample, "ida_to_ian"),
                "orthomosaic_group": f"orthomosaic::{sample.orthomosaic}",
                "flight_group": f"orthomosaic_surrogate::{sample.orthomosaic}",
                "sequence_group": f"orthomosaic_surrogate::{sample.orthomosaic}",
                "exact_duplicate_group": exact_group_by_id[sample.sample_id],
                "perceptual_duplicate_group": group_by_id[sample.sample_id],
                "exact_hash_sha256": hashes[sample.sample_id]["exact_hash"],
                "phash64": hashes[sample.sample_id]["phash"],
                "polygon_alpha_invalid_fraction": invalid_fractions[
                    sample.sample_id
                ],
            }
        )

    cross_event_candidates = [
        row for row in duplicate_pairs if row["cross_event"]
    ]
    audit = {
        "status": "pass" if not cross_event_candidates else "blocked",
        "phash_threshold": PHASH_THRESHOLD,
        "active_rows": len(manifest_rows),
        "internal_test_rows_in_new_manifest": 0,
        "native_flight_sequence_ids_available": False,
        "flight_sequence_surrogate": "orthomosaic",
        "exact_duplicate_pair_count": sum(
            bool(row["exact_duplicate"]) for row in duplicate_pairs
        ),
        "perceptual_candidate_pair_count": sum(
            bool(row["perceptual_candidate"]) for row in duplicate_pairs
        ),
        "cross_event_duplicate_candidate_count": len(cross_event_candidates),
        "minimum_cross_event_phash_distance": (
            cross_event_nearest[0]["phash_distance"]
            if cross_event_nearest
            else None
        ),
        "fold_checks": {},
    }
    for direction in DIRECTIONS:
        audit["fold_checks"][direction] = audit_direction(
            manifest_rows, direction
        )
        if audit["fold_checks"][direction]["status"] != "pass":
            audit["status"] = "blocked"

    if cross_event_candidates:
        audit["blocking_cross_event_candidates"] = cross_event_candidates
    return manifest_rows, audit, duplicate_pairs + cross_event_nearest


def role_for_direction(sample: Sample, direction: str) -> str:
    definition = DIRECTIONS[direction]
    if sample.event == definition["train_event"] and sample.legacy_split == "train":
        return "train_pool"
    if sample.event == definition["eval_event"] and sample.legacy_split == "val":
        return "evaluation"
    return "excluded"


def audit_direction(
    manifest_rows: Sequence[dict[str, Any]], direction: str
) -> dict[str, Any]:
    train_rows = [
        row
        for row in manifest_rows
        if row[f"{direction}_role"] == "train_pool"
        and bool(row["in_train_30pct"])
    ]
    eval_rows = [
        row
        for row in manifest_rows
        if row[f"{direction}_role"] == "evaluation"
    ]
    fields = (
        "sample_id",
        "building_id",
        "orthomosaic_group",
        "flight_group",
        "sequence_group",
        "exact_duplicate_group",
        "perceptual_duplicate_group",
    )
    overlaps: dict[str, list[str]] = {}
    for field in fields:
        train_values = {str(row[field]) for row in train_rows}
        eval_values = {str(row[field]) for row in eval_rows}
        overlaps[field] = sorted(train_values & eval_values)
    status = "pass" if all(not values for values in overlaps.values()) else "blocked"
    return {
        "status": status,
        "train_30pct_count": len(train_rows),
        "evaluation_count": len(eval_rows),
        "overlap_counts": {field: len(values) for field, values in overlaps.items()},
        "overlap_values": overlaps,
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def model_initialization_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=len(LABELS)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Training subset lacks a class: counts={counts.tolist()}")
    weights = len(labels) / (len(LABELS) * counts)
    return weights.astype(np.float32)


def normalized_tensor(images: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    return tensor.div_(127.5).sub_(1.0)


def multiclass_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.eye(len(LABELS), dtype=np.float64)[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    result = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index == bins - 1:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence >= lower) & (confidence < upper)
        if not in_bin.any():
            continue
        result += float(in_bin.mean()) * abs(
            float(correct[in_bin].mean()) - float(confidence[in_bin].mean())
        )
    return result


def evaluate_predictions(
    labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(len(LABELS)),
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "multiclass_brier": multiclass_brier(probabilities, labels),
        "ece_10bin": expected_calibration_error(probabilities, labels, bins=10),
        "calibration_viable": len(labels) >= 100,
    }
    for class_index, label in enumerate(LABELS):
        key = label.replace(" ", "_")
        metrics[f"precision_{key}"] = float(precision[class_index])
        metrics[f"recall_{key}"] = float(recall[class_index])
        metrics[f"f1_{key}"] = float(per_class_f1[class_index])
        metrics[f"support_{key}"] = int(support[class_index])
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(LABELS)))
    return metrics, matrix


def train_and_evaluate(
    run_id: str,
    variant: Variant,
    subset_name: str,
    direction: str,
    seed: int,
    train_samples: Sequence[Sample],
    eval_samples: Sequence[Sample],
    images_by_id: dict[str, np.ndarray],
    config: ExperimentConfig,
    phase: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    start = time.perf_counter()
    seed_everything(seed)
    model = SmallDamageCNN(dropout=config.dropout)
    initialization_hash = model_initialization_hash(model)
    model.train()

    train_images = np.stack([images_by_id[sample.sample_id] for sample in train_samples])
    train_labels = np.asarray([sample.class_id for sample in train_samples], dtype=np.int64)
    eval_images = np.stack([images_by_id[sample.sample_id] for sample in eval_samples])
    eval_labels = np.asarray([sample.class_id for sample in eval_samples], dtype=np.int64)
    weights = torch.from_numpy(class_weights(train_labels))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_tensor = normalized_tensor(train_images)
    eval_tensor = normalized_tensor(eval_images)

    curve_rows: list[dict[str, Any]] = []
    generator = np.random.default_rng(seed)
    for epoch in range(1, config.epochs + 1):
        permutation = generator.permutation(len(train_samples))
        total_loss = 0.0
        correct = 0
        seen = 0
        for start_index in range(0, len(permutation), config.batch_size):
            indices = permutation[start_index : start_index + config.batch_size]
            batch_images = train_tensor[indices]
            batch_labels = torch.from_numpy(train_labels[indices])
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            batch_size = len(indices)
            total_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
            seen += batch_size
        curve_rows.append(
            {
                "run_id": run_id,
                "phase": phase,
                "variant": variant.variant_id,
                "subset": subset_name,
                "direction": direction,
                "seed": seed,
                "epoch": epoch,
                "train_loss": total_loss / seen,
                "train_accuracy": correct / seen,
            }
        )

    model.eval()
    probability_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start_index in range(0, len(eval_samples), config.batch_size):
            logits = model(eval_tensor[start_index : start_index + config.batch_size])
            probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
    probabilities = np.concatenate(probability_batches, axis=0)
    predictions = probabilities.argmax(axis=1)
    metrics, matrix = evaluate_predictions(eval_labels, predictions, probabilities)

    run_row: dict[str, Any] = {
        "run_id": run_id,
        "phase": phase,
        "variant": variant.variant_id,
        "variant_name": variant.display_name,
        "margin_fraction": variant.margin_fraction,
        "masked": variant.masked,
        "subset": subset_name,
        "direction": direction,
        "train_event": DIRECTIONS[direction]["train_event"],
        "eval_event": DIRECTIONS[direction]["eval_event"],
        "seed": seed,
        "train_count": len(train_samples),
        "eval_count": len(eval_samples),
        "initialization_hash": initialization_hash,
        "class_weights": json.dumps(class_weights(train_labels).tolist()),
        "final_train_loss": curve_rows[-1]["train_loss"],
        "final_train_accuracy": curve_rows[-1]["train_accuracy"],
        "duration_seconds": time.perf_counter() - start,
        **metrics,
    }

    prediction_rows: list[dict[str, Any]] = []
    for sample, predicted, probability in zip(eval_samples, predictions, probabilities):
        row: dict[str, Any] = {
            "run_id": run_id,
            "phase": phase,
            "variant": variant.variant_id,
            "subset": subset_name,
            "direction": direction,
            "seed": seed,
            "sample_id": sample.sample_id,
            "building_id": sample.building_id,
            "event": sample.event,
            "orthomosaic": sample.orthomosaic,
            "true_class_id": sample.class_id,
            "true_label": sample.label,
            "predicted_class_id": int(predicted),
            "predicted_label": LABELS[int(predicted)],
            "correct": int(predicted) == sample.class_id,
        }
        for class_index, label in enumerate(LABELS):
            row[f"probability_{label.replace(' ', '_')}"] = float(
                probability[class_index]
            )
        prediction_rows.append(row)

    matrix_rows: list[dict[str, Any]] = []
    for true_index, true_label in enumerate(LABELS):
        for predicted_index, predicted_label in enumerate(LABELS):
            matrix_rows.append(
                {
                    "run_id": run_id,
                    "phase": phase,
                    "variant": variant.variant_id,
                    "subset": subset_name,
                    "direction": direction,
                    "seed": seed,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": int(matrix[true_index, predicted_index]),
                }
            )
    return run_row, prediction_rows, curve_rows, matrix_rows


def render_variant_images(
    variant: Variant,
    samples: Sequence[Sample],
    image_paths: dict[str, Path],
    input_size: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    print(f"rendering {variant.variant_id} for {len(samples)} samples", flush=True)
    images: dict[str, np.ndarray] = {}
    diagnostic_rows: list[dict[str, Any]] = []
    sources = {
        orthomosaic: rasterio.open(path)
        for orthomosaic, path in image_paths.items()
    }
    try:
        for index, sample in enumerate(samples, start=1):
            image, diagnostics = render_model_input_with_diagnostics(
                sample, variant, sources[sample.orthomosaic], input_size
            )
            images[sample.sample_id] = image
            diagnostic_rows.append(diagnostics)
            if index % 100 == 0 or index == len(samples):
                print(
                    f"rendered {variant.variant_id}: {index}/{len(samples)}",
                    flush=True,
                )
    finally:
        for source in sources.values():
            source.close()
    return images, diagnostic_rows


def samples_for_run(
    samples: Sequence[Sample], subset_name: str, direction: str
) -> tuple[list[Sample], list[Sample]]:
    definition = DIRECTIONS[direction]
    train_samples = [
        sample
        for sample in samples
        if sample.event == definition["train_event"]
        and sample.legacy_split == "train"
        and subset_name in sample.subset_names
    ]
    eval_samples = [
        sample
        for sample in samples
        if sample.event == definition["eval_event"]
        and sample.legacy_split == "val"
    ]
    return train_samples, eval_samples


def run_grid(
    variants: Sequence[Variant],
    seeds: Sequence[int],
    phase: str,
    samples: Sequence[Sample],
    image_paths: dict[str, Path],
    config: ExperimentConfig,
    output_dir: Path,
    all_run_rows: list[dict[str, Any]],
    all_prediction_rows: list[dict[str, Any]],
    all_curve_rows: list[dict[str, Any]],
    all_matrix_rows: list[dict[str, Any]],
    completed_keys: set[tuple[str, str, str, int]],
    image_cache: dict[str, dict[str, np.ndarray]],
    all_roi_diagnostic_rows: list[dict[str, Any]],
) -> None:
    for variant in variants:
        if variant.variant_id not in image_cache:
            images, diagnostics = render_variant_images(
                variant, samples, image_paths, config.input_size
            )
            image_cache[variant.variant_id] = images
            all_roi_diagnostic_rows.extend(diagnostics)
        images_by_id = image_cache[variant.variant_id]
        for subset_name in SUBSET_FLAGS:
            for direction in DIRECTIONS:
                train_samples, eval_samples = samples_for_run(
                    samples, subset_name, direction
                )
                expected_train = EXPECTED_TRAIN_COUNTS[direction][subset_name]
                expected_eval = EXPECTED_EVAL_COUNTS[direction]
                if len(train_samples) != expected_train or len(eval_samples) != expected_eval:
                    raise ValueError(
                        f"Unexpected run counts for {subset_name}/{direction}: "
                        f"train={len(train_samples)}, eval={len(eval_samples)}"
                    )
                for seed in seeds:
                    key = (variant.variant_id, subset_name, direction, seed)
                    if key in completed_keys:
                        continue
                    run_id = (
                        f"{variant.variant_id}__{subset_name}__{direction}__seed{seed}"
                    )
                    print(f"RUN {run_id} phase={phase}", flush=True)
                    run_row, predictions, curves, matrices = train_and_evaluate(
                        run_id=run_id,
                        variant=variant,
                        subset_name=subset_name,
                        direction=direction,
                        seed=seed,
                        train_samples=train_samples,
                        eval_samples=eval_samples,
                        images_by_id=images_by_id,
                        config=config,
                        phase=phase,
                    )
                    all_run_rows.append(run_row)
                    all_prediction_rows.extend(predictions)
                    all_curve_rows.extend(curves)
                    all_matrix_rows.extend(matrices)
                    completed_keys.add(key)
                    print(
                        f"DONE {run_id} macro_f1={run_row['macro_f1']:.4f} "
                        f"seconds={run_row['duration_seconds']:.2f}",
                        flush=True,
                    )
        persist_raw_results(
            output_dir,
            all_run_rows,
            all_prediction_rows,
            all_curve_rows,
            all_matrix_rows,
        )
        write_csv(output_dir / "roi_diagnostics.csv", all_roi_diagnostic_rows)


def persist_raw_results(
    output_dir: Path,
    run_rows: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
    curve_rows: Sequence[dict[str, Any]],
    matrix_rows: Sequence[dict[str, Any]],
) -> None:
    write_csv(output_dir / "run_metrics.csv", run_rows)
    write_csv(output_dir / "predictions.csv", prediction_rows)
    write_csv(output_dir / "training_curves.csv", curve_rows)
    write_csv(output_dir / "confusion_matrices.csv", matrix_rows)


METRIC_FIELDS = (
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "precision_no_damage",
    "recall_no_damage",
    "f1_no_damage",
    "precision_minor_damage",
    "recall_minor_damage",
    "f1_minor_damage",
    "precision_major_damage",
    "recall_major_damage",
    "f1_major_damage",
    "precision_destroyed",
    "recall_destroyed",
    "f1_destroyed",
    "multiclass_brier",
    "ece_10bin",
)

ROI_DIAGNOSTIC_FIELDS = (
    "roi_area_source_pixels",
    "valid_building_fraction_canvas",
    "visible_context_fraction_canvas",
    "alpha_invalid_fraction_roi",
    "visible_alpha_invalid_fraction_canvas",
    "square_padding_fraction_canvas",
    "building_alpha_invalid_fraction",
)


def t_critical_95(sample_count: int) -> float:
    values = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    return values.get(sample_count, 1.96)


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty value list.")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = (
        t_critical_95(len(values)) * std / math.sqrt(len(values))
        if len(values) > 1
        else 0.0
    )
    return {
        "seed_count": len(values),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def aggregate_roi_diagnostics(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["event"])].append(row)
        grouped[(row["variant"], "equal_event_mean")].append(row)

    result: list[dict[str, Any]] = []
    event_names = tuple(ALLOWED_ORTHOMOSAICS.values())
    for (variant_id, event), group_rows in sorted(grouped.items()):
        if event == "equal_event_mean":
            event_groups = {
                event_name: [
                    row for row in group_rows if row["event"] == event_name
                ]
                for event_name in event_names
            }
            if any(not values for values in event_groups.values()):
                raise ValueError(
                    f"Missing event diagnostics for {variant_id}."
                )
            output: dict[str, Any] = {
                "variant": variant_id,
                "event": event,
                "sample_count": sum(len(values) for values in event_groups.values()),
            }
            for field in ROI_DIAGNOSTIC_FIELDS:
                output[f"{field}_mean"] = statistics.fmean(
                    statistics.fmean(float(row[field]) for row in values)
                    for values in event_groups.values()
                )
            result.append(output)
            continue

        output = {
            "variant": variant_id,
            "event": event,
            "sample_count": len(group_rows),
        }
        for field in ROI_DIAGNOSTIC_FIELDS:
            values = [float(row[field]) for row in group_rows]
            output[f"{field}_mean"] = statistics.fmean(values)
            output[f"{field}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        result.append(output)
    return result


def equal_direction_seed_rows(
    run_rows: Sequence[dict[str, Any]],
    variant_ids: set[str],
    seeds: set[int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        if row["variant"] not in variant_ids or int(row["seed"]) not in seeds:
            continue
        grouped[(row["variant"], row["subset"], int(row["seed"]))].append(row)
    result: list[dict[str, Any]] = []
    for (variant_id, subset_name, seed), rows in sorted(grouped.items()):
        if {row["direction"] for row in rows} != set(DIRECTIONS):
            raise ValueError(
                f"Missing direction for {variant_id}/{subset_name}/seed{seed}"
            )
        combined: dict[str, Any] = {
            "variant": variant_id,
            "subset": subset_name,
            "seed": seed,
        }
        for metric in METRIC_FIELDS:
            combined[metric] = statistics.fmean(float(row[metric]) for row in rows)
        result.append(combined)
    return result


def aggregate_seed_rows(
    seed_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(row["variant"], row["subset"])].append(row)
    aggregate: list[dict[str, Any]] = []
    by_key = {(row["variant"], row["subset"], row["seed"]): row for row in seed_rows}
    for (variant_id, subset_name), rows in sorted(grouped.items()):
        output: dict[str, Any] = {
            "variant": variant_id,
            "subset": subset_name,
            "seed_count": len(rows),
        }
        for metric in METRIC_FIELDS:
            summary = summarize_values([float(row[metric]) for row in rows])
            for statistic, value in summary.items():
                if statistic == "seed_count":
                    continue
                output[f"{metric}_{statistic}"] = value

        for baseline_variant, suffix in (("margin_0", "vs_margin_0"), ("margin_10", "vs_margin_10")):
            paired_differences: list[float] = []
            for row in rows:
                baseline = by_key.get((baseline_variant, subset_name, row["seed"]))
                if baseline is None:
                    continue
                paired_differences.append(
                    float(row["macro_f1"]) - float(baseline["macro_f1"])
                )
            if paired_differences:
                summary = summarize_values(paired_differences)
                output[f"macro_f1_diff_{suffix}_mean"] = summary["mean"]
                output[f"macro_f1_diff_{suffix}_std"] = summary["std"]
                output[f"macro_f1_diff_{suffix}_ci95_low"] = summary["ci95_low"]
                output[f"macro_f1_diff_{suffix}_ci95_high"] = summary["ci95_high"]
        aggregate.append(output)
    return aggregate


def initial_winners(
    aggregate_rows: Sequence[dict[str, Any]],
) -> dict[str, str]:
    winners: dict[str, str] = {}
    for subset_name in SUBSET_FLAGS:
        candidates = [row for row in aggregate_rows if row["subset"] == subset_name]
        winner = max(candidates, key=lambda row: row["macro_f1_mean"])
        winners[subset_name] = winner["variant"]
    return winners


def select_refinement_variants(
    aggregate_rows: Sequence[dict[str, Any]],
) -> tuple[list[Variant], set[str], dict[str, Any]]:
    rows_30 = [row for row in aggregate_rows if row["subset"] == "train_30pct"]
    numeric_rows = [row for row in rows_30 if row["variant"].startswith("margin_")]
    numeric_winner = max(numeric_rows, key=lambda row: row["macro_f1_mean"])
    variant_by_id = {
        variant.variant_id: variant for variant in map(variant_from_tuple, INITIAL_VARIANTS)
    }
    winner_variant = variant_by_id[numeric_winner["variant"]]
    winning_margin = float(winner_variant.margin_fraction)
    refined_fractions = sorted(
        {
            max(0.0, winning_margin - 0.025),
            min(0.50, winning_margin + 0.025),
        }
        - {
            float(variant.margin_fraction)
            for variant in variant_by_id.values()
            if variant.margin_fraction is not None
        }
    )
    refined = [margin_variant(fraction) for fraction in refined_fractions]

    winners = initial_winners(aggregate_rows)
    top_two_30 = sorted(
        rows_30, key=lambda row: row["macro_f1_mean"], reverse=True
    )[:2]
    finalist_ids = {
        "margin_0",
        "margin_10",
        "masked_building",
        numeric_winner["variant"],
        *winners.values(),
        *(row["variant"] for row in top_two_30),
        *(variant.variant_id for variant in refined),
    }
    selection = {
        "initial_winners_by_subset": winners,
        "initial_numeric_winner_30pct": numeric_winner["variant"],
        "initial_numeric_winner_margin_fraction": winning_margin,
        "refined_margin_fractions": refined_fractions,
        "finalist_variant_ids": sorted(finalist_ids),
    }
    return refined, finalist_ids, selection


def find_aggregate_row(
    rows: Sequence[dict[str, Any]], variant: str, subset: str
) -> dict[str, Any]:
    return next(
        row for row in rows if row["variant"] == variant and row["subset"] == subset
    )


def choose_empirical_best(
    aggregate_rows: Sequence[dict[str, Any]],
    seed_rows: Sequence[dict[str, Any]],
    variant_by_id: dict[str, Variant],
) -> dict[str, Any]:
    best_by_subset: dict[str, Any] = {}
    for subset_name in SUBSET_FLAGS:
        candidates = [row for row in aggregate_rows if row["subset"] == subset_name]
        raw_best = max(candidates, key=lambda row: row["macro_f1_mean"])
        baseline = find_aggregate_row(aggregate_rows, "margin_0", subset_name)
        for row in candidates:
            row["major_recall_difference_vs_margin_0"] = (
                row["recall_major_damage_mean"]
                - baseline["recall_major_damage_mean"]
            )
            row["destroyed_recall_difference_vs_margin_0"] = (
                row["recall_destroyed_mean"]
                - baseline["recall_destroyed_mean"]
            )
            row["minority_recall_guardrail_passed"] = (
                row["major_recall_difference_vs_margin_0"]
                >= -RELEVANT_RECALL_DROP
                and row["destroyed_recall_difference_vs_margin_0"]
                >= -RELEVANT_RECALL_DROP
            )
        eligible_candidates = [
            row for row in candidates if row["minority_recall_guardrail_passed"]
        ]
        guardrail_best = max(
            eligible_candidates, key=lambda row: row["macro_f1_mean"]
        )
        numeric_candidates = [
            row
            for row in eligible_candidates
            if variant_by_id[row["variant"]].margin_fraction is not None
        ]
        numeric_best = max(numeric_candidates, key=lambda row: row["macro_f1_mean"])
        indistinguishable_numeric: list[dict[str, Any]] = []
        for row in numeric_candidates:
            variant = variant_by_id[row["variant"]]
            best_variant = variant_by_id[numeric_best["variant"]]
            if variant.margin_fraction is None or best_variant.margin_fraction is None:
                continue
            if row["variant"] == numeric_best["variant"]:
                indistinguishable_numeric.append(row)
                continue
            comparison = paired_difference(
                seed_rows,
                row["variant"],
                numeric_best["variant"],
                subset_name,
                "macro_f1",
            )
            if comparison["statistically_indistinguishable"]:
                indistinguishable_numeric.append(row)
        preferred_numeric = min(
            indistinguishable_numeric,
            key=lambda row: float(variant_by_id[row["variant"]].margin_fraction),
        )
        selected_variant = (
            guardrail_best["variant"]
            if variant_by_id[guardrail_best["variant"]].masked
            else preferred_numeric["variant"]
        )
        best_by_subset[subset_name] = {
            "raw_best_variant": raw_best["variant"],
            "raw_best_macro_f1": raw_best["macro_f1_mean"],
            "raw_best_major_recall_difference_vs_margin_0": raw_best[
                "major_recall_difference_vs_margin_0"
            ],
            "raw_best_destroyed_recall_difference_vs_margin_0": raw_best[
                "destroyed_recall_difference_vs_margin_0"
            ],
            "raw_best_minority_recall_guardrail_passed": raw_best[
                "minority_recall_guardrail_passed"
            ],
            "best_guardrail_eligible_variant": guardrail_best["variant"],
            "best_guardrail_eligible_macro_f1": guardrail_best[
                "macro_f1_mean"
            ],
            "best_guardrail_eligible_numeric_margin_variant": numeric_best[
                "variant"
            ],
            "best_guardrail_eligible_numeric_margin_macro_f1": numeric_best[
                "macro_f1_mean"
            ],
            "smaller_numeric_margin_preferred_by_paired_ci": preferred_numeric[
                "variant"
            ],
            "selected_variant_after_guardrail_and_tie_rule": selected_variant,
        }
    return best_by_subset


def event_aggregate(
    run_rows: Sequence[dict[str, Any]],
    finalist_ids: set[str],
    final_seeds: set[int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        if row["variant"] in finalist_ids and int(row["seed"]) in final_seeds:
            grouped[(row["variant"], row["subset"], row["direction"])].append(row)
    result: list[dict[str, Any]] = []
    for (variant_id, subset_name, direction), rows in sorted(grouped.items()):
        output: dict[str, Any] = {
            "variant": variant_id,
            "subset": subset_name,
            "direction": direction,
            "eval_event": DIRECTIONS[direction]["eval_event"],
            "seed_count": len(rows),
        }
        for metric in METRIC_FIELDS:
            summary = summarize_values([float(row[metric]) for row in rows])
            for statistic, value in summary.items():
                if statistic != "seed_count":
                    output[f"{metric}_{statistic}"] = value
        result.append(output)
    return result


def paired_difference(
    seed_rows: Sequence[dict[str, Any]],
    left_variant: str,
    right_variant: str,
    subset_name: str,
    metric: str,
) -> dict[str, Any]:
    by_key = {
        (row["variant"], row["subset"], row["seed"]): row for row in seed_rows
    }
    seeds = sorted(
        row["seed"]
        for row in seed_rows
        if row["variant"] == left_variant and row["subset"] == subset_name
    )
    differences = [
        float(by_key[(left_variant, subset_name, seed)][metric])
        - float(by_key[(right_variant, subset_name, seed)][metric])
        for seed in seeds
    ]
    summary = summarize_values(differences)
    return {
        "left_variant": left_variant,
        "right_variant": right_variant,
        "subset": subset_name,
        "metric": metric,
        **summary,
        "statistically_indistinguishable": summary["ci95_low"] <= 0 <= summary["ci95_high"],
    }


def stability_assessment(
    aggregate_rows: Sequence[dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
    seed_rows: Sequence[dict[str, Any]],
    best_by_subset: dict[str, Any],
    variant_by_id: dict[str, Variant],
) -> dict[str, Any]:
    selected_variant = best_by_subset["train_30pct"][
        "selected_variant_after_guardrail_and_tie_rule"
    ]
    raw_best_variant = best_by_subset["train_30pct"]["raw_best_variant"]
    selected_30 = find_aggregate_row(aggregate_rows, selected_variant, "train_30pct")
    baseline_30 = find_aggregate_row(aggregate_rows, "margin_0", "train_30pct")
    major_drop = (
        selected_30["recall_major_damage_mean"]
        - baseline_30["recall_major_damage_mean"]
    )
    destroyed_drop = (
        selected_30["recall_destroyed_mean"]
        - baseline_30["recall_destroyed_mean"]
    )
    raw_subset_winners = {
        subset: details["raw_best_variant"]
        for subset, details in best_by_subset.items()
    }
    selected_subset_winners = {
        subset: details["selected_variant_after_guardrail_and_tie_rule"]
        for subset, details in best_by_subset.items()
    }
    seed_winners: dict[str, str] = {}
    for seed in sorted({int(row["seed"]) for row in seed_rows}):
        candidates = [
            row
            for row in seed_rows
            if row["subset"] == "train_30pct" and int(row["seed"]) == seed
        ]
        seed_winners[str(seed)] = max(
            candidates, key=lambda row: row["macro_f1"]
        )["variant"]
    direction_winners: dict[str, str] = {}
    for direction in DIRECTIONS:
        candidates = [
            row
            for row in event_rows
            if row["subset"] == "train_30pct" and row["direction"] == direction
        ]
        direction_winners[direction] = max(
            candidates, key=lambda row: row["macro_f1_mean"]
        )["variant"]
    paired_vs_bbox = paired_difference(
        seed_rows, selected_variant, "margin_0", "train_30pct", "macro_f1"
    )
    paired_vs_10 = paired_difference(
        seed_rows, selected_variant, "margin_10", "train_30pct", "macro_f1"
    )
    minority_ok = (
        major_drop >= -RELEVANT_RECALL_DROP
        and destroyed_drop >= -RELEVANT_RECALL_DROP
    )
    same_raw_subset_winner = len(set(raw_subset_winners.values())) == 1
    same_selected_subset_winner = len(set(selected_subset_winners.values())) == 1
    same_direction_winner = len(set(direction_winners.values())) == 1
    same_seed_winner = len(set(seed_winners.values())) == 1
    stable = (
        same_raw_subset_winner
        and same_selected_subset_winner
        and same_direction_winner
        and same_seed_winner
        and minority_ok
        and not paired_vs_bbox["statistically_indistinguishable"]
    )
    return {
        "raw_best_30pct": raw_best_variant,
        "selected_variant_30pct_after_guardrail_and_tie_rule": selected_variant,
        "selected_margin_fraction": variant_by_id[selected_variant].margin_fraction,
        "raw_winners_by_subset": raw_subset_winners,
        "selected_after_guardrail_and_tie_rule_by_subset": selected_subset_winners,
        "direction_winners_30pct": direction_winners,
        "seed_winners_30pct_equal_direction": seed_winners,
        "major_recall_difference_vs_margin_0": major_drop,
        "destroyed_recall_difference_vs_margin_0": destroyed_drop,
        "minority_recall_guardrail_passed": minority_ok,
        "same_raw_winner_across_subsets": same_raw_subset_winner,
        "same_selected_winner_across_subsets": same_selected_subset_winner,
        "same_winner_across_directions": same_direction_winner,
        "same_winner_across_seeds": same_seed_winner,
        "paired_macro_f1_vs_margin_0": paired_vs_bbox,
        "paired_macro_f1_vs_margin_10": paired_vs_10,
        "stable_winner": stable,
        "empirical_conclusion": (
            "STABLE EMPIRICAL WINNER AMONG TESTED VARIANTS"
            if stable
            else "NO STABLE WINNER WITH CURRENT DATA"
        ),
        "decision": "NEEDS FULL DATA",
    }


def aggregate_confusion_for_plot(
    matrix_rows: Sequence[dict[str, Any]],
    variant_id: str,
    direction: str,
    seeds: set[int],
) -> np.ndarray:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=float)
    for row in matrix_rows:
        if (
            row["variant"] == variant_id
            and row["subset"] == "train_30pct"
            and row["direction"] == direction
            and int(row["seed"]) in seeds
        ):
            true_index = LABEL_TO_ID[row["true_label"]]
            predicted_index = LABEL_TO_ID[row["predicted_label"]]
            matrix[true_index, predicted_index] += int(row["count"])
    row_totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals > 0)


def create_professor_summary_figure(
    assets_dir: Path,
    final_aggregate: Sequence[dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
    variant_by_id: dict[str, Variant],
) -> None:
    """Create the single decision figure used by the concise UAV-P04 report."""
    ink = "#24323d"
    muted = "#5d6d78"
    blue = "#2d6f8e"
    blue_light = "#8eb9cc"
    gold = "#d69f35"
    neutral = "#aeb8bf"

    final_rows_30 = {
        row["variant"]: row
        for row in final_aggregate
        if row["subset"] == "train_30pct"
    }
    ordered_variants = [
        "masked_building",
        "margin_0",
        "margin_5",
        "margin_10",
        "margin_12p5",
        "margin_15",
        "margin_17p5",
        "margin_50",
    ]

    figure, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    figure.patch.set_facecolor("white")

    # Panel A: conceptual ROI treatments without reproducing source imagery.
    axis = axes[0, 0]
    axis.set_title("A. ROI treatments (conceptual)", loc="left", color=ink)
    axis.set_xlim(0, 3)
    axis.set_ylim(0, 1)
    axis.axis("off")
    building = np.asarray(
        [[0.19, 0.18], [0.15, 0.50], [0.32, 0.76], [0.67, 0.70],
         [0.82, 0.40], [0.66, 0.17]]
    )
    for index, label in enumerate(("Masked control", "0% box", "+12.5% margin")):
        left = index + 0.08
        if index == 2:
            axis.add_patch(
                Rectangle(
                    (left, 0.12), 0.82, 0.70, facecolor="#f3e5bd",
                    edgecolor=gold, linewidth=2,
                )
            )
            axis.add_patch(
                Rectangle(
                    (left + 0.10, 0.20), 0.62, 0.54, fill=False,
                    edgecolor=muted, linewidth=1.3, linestyle="--",
                )
            )
            polygon = building * [0.54, 0.46] + [left + 0.15, 0.24]
        else:
            background = ink if index == 0 else "#dce7ec"
            axis.add_patch(
                Rectangle(
                    (left + 0.10, 0.20), 0.62, 0.54,
                    facecolor=background, edgecolor=ink, linewidth=1.5,
                )
            )
            polygon = building * [0.54, 0.46] + [left + 0.15, 0.24]
        axis.add_patch(
            PlotPolygon(polygon, closed=True, facecolor=blue_light,
                        edgecolor=blue, linewidth=1.5)
        )
        axis.text(left + 0.41, 0.04, label, ha="center", va="bottom", color=ink)
    axis.text(
        0.02,
        0.92,
        "Margin is added to every side of the current building box;\n"
        "aspect ratio and square padding remain fixed.",
        color=muted,
        va="top",
    )

    # Panel B: final TRAIN-30% comparison with uncertainty.
    axis = axes[0, 1]
    positions = np.arange(len(ordered_variants))
    means = np.asarray(
        [float(final_rows_30[variant]["macro_f1_mean"]) for variant in ordered_variants]
    )
    lower = np.asarray(
        [float(final_rows_30[variant]["macro_f1_ci95_low"]) for variant in ordered_variants]
    )
    upper = np.asarray(
        [float(final_rows_30[variant]["macro_f1_ci95_high"]) for variant in ordered_variants]
    )
    colors = [
        blue if variant == "margin_0" else gold if variant == "margin_12p5" else neutral
        for variant in ordered_variants
    ]
    for position, mean, low, high, color in zip(positions, means, lower, upper, colors):
        axis.errorbar(
            mean,
            position,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=7,
            linewidth=1.8,
        )
    axis.set_yticks(
        positions,
        [variant_by_id[variant].display_name for variant in ordered_variants],
    )
    axis.invert_yaxis()
    axis.set_xlim(0.08, 0.34)
    axis.set_xlabel("Macro F1 (focused scale)")
    paired_mean = float(
        final_rows_30["margin_12p5"]["macro_f1_diff_vs_margin_0_mean"]
    )
    paired_low = float(
        final_rows_30["margin_12p5"]["macro_f1_diff_vs_margin_0_ci95_low"]
    )
    paired_high = float(
        final_rows_30["margin_12p5"]["macro_f1_diff_vs_margin_0_ci95_high"]
    )
    axis.set_title("B. TRAIN-30% macro F1", loc="left", color=ink, pad=42)
    axis.text(
        0.0,
        1.02,
        "Five paired seeds; equal weight per event; 95% t intervals\n"
        f"+12.5% minus 0%: {paired_mean:+.3f}, 95% CI [{paired_low:+.3f}, {paired_high:+.3f}]",
        transform=axis.transAxes,
        color=muted,
    )
    axis.axvline(
        float(final_rows_30["margin_0"]["macro_f1_mean"]),
        color=blue,
        linewidth=1,
        linestyle="--",
        alpha=0.65,
    )
    axis.grid(axis="x", alpha=0.2)

    # Panel C: the two held-out events disagree.
    axis = axes[1, 0]
    event_lookup = {
        (row["variant"], row["direction"]): row
        for row in event_rows
        if row["subset"] == "train_30pct"
    }
    event_labels = ["Ian→Ida", "Ida→Ian"]
    directions = ["ian_to_ida", "ida_to_ian"]
    x = np.arange(len(directions))
    width = 0.34
    for offset, variant, color, label in (
        (-width / 2, "margin_0", blue, "0%"),
        (width / 2, "margin_12p5", gold, "+12.5%"),
    ):
        values = [
            float(event_lookup[(variant, direction)]["macro_f1_mean"])
            for direction in directions
        ]
        axis.bar(
            x + offset,
            values,
            width,
            color=color,
            edgecolor=ink,
            linewidth=0.8,
            label=label,
        )
        for position, value in zip(x + offset, values):
            axis.text(position, value + 0.008, f"{value:.3f}", ha="center", color=ink)
    axis.set_xticks(x, event_labels)
    axis.set_ylim(0, 0.38)
    axis.set_ylabel("Macro F1")
    axis.set_title("C. Macro F1 by held-out event", loc="left", color=ink, pad=25)
    axis.text(
        0.0,
        1.02,
        "The apparent +12.5% gain comes from only one direction",
        transform=axis.transAxes,
        color=muted,
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, loc="upper left")

    # Panel D: the numerical leader violates the minority-recall guardrail.
    axis = axes[1, 1]
    class_labels = ["Major damage", "Destroyed"]
    metrics = ["recall_major_damage_mean", "recall_destroyed_mean"]
    x = np.arange(len(metrics))
    for offset, variant, color, label in (
        (-width / 2, "margin_0", blue, "0%"),
        (width / 2, "margin_12p5", gold, "+12.5%"),
    ):
        values = [float(final_rows_30[variant][metric]) for metric in metrics]
        axis.bar(
            x + offset,
            values,
            width,
            color=color,
            edgecolor=ink,
            linewidth=0.8,
            label=label,
        )
        for position, value in zip(x + offset, values):
            axis.text(position, value + 0.012, f"{value:.3f}", ha="center", color=ink)
    axis.set_xticks(x, class_labels)
    axis.set_ylim(0, 0.62)
    axis.set_ylabel("Recall")
    axis.set_title("D. Minority-class recall", loc="left", color=ink, pad=25)
    axis.text(
        0.0,
        1.02,
        "Major damage: −9.3 points (fails the predeclared −5-point guardrail)",
        transform=axis.transAxes,
        color=muted,
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, loc="upper left")

    for axis in axes.flat:
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.suptitle("UAV-P04 ROI-margin selection evidence", fontsize=18, color=ink, y=0.985)
    figure.text(
        0.5,
        0.945,
        "Current recommendation: retain the 0% building box provisionally — no stable improvement was demonstrated",
        ha="center",
        color=muted,
    )
    figure.text(
        0.5,
        0.015,
        "Decision status: NEEDS FULL DATA. The internal test and final reserved event were not read or used.",
        ha="center",
        color=muted,
    )
    figure.tight_layout(rect=(0.03, 0.055, 0.98, 0.91), h_pad=2.6, w_pad=2.8)
    figure.savefig(
        assets_dir / "UAV-P04_classifier_decision-summary.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_figures(
    assets_dir: Path,
    initial_aggregate: Sequence[dict[str, Any]],
    final_aggregate: Sequence[dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
    matrix_rows: Sequence[dict[str, Any]],
    curve_rows: Sequence[dict[str, Any]],
    stability: dict[str, Any],
    finalist_ids: set[str],
    final_seeds: set[int],
    variant_by_id: dict[str, Variant],
) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    create_professor_summary_figure(
        assets_dir=assets_dir,
        final_aggregate=final_aggregate,
        event_rows=event_rows,
        variant_by_id=variant_by_id,
    )
    palette = {"train_10pct": "#9ec4d8", "train_20pct": "#4f8fad", "train_30pct": "#174f6b"}

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, rows, title in (
        (axes[0], initial_aggregate, "Initial coarse search"),
        (axes[1], final_aggregate, "Finalists and 2.5-point refinement"),
    ):
        variant_ids = sorted(
            {row["variant"] for row in rows},
            key=lambda variant_id: (
                variant_by_id[variant_id].margin_fraction is not None,
                variant_by_id[variant_id].margin_fraction
                if variant_by_id[variant_id].margin_fraction is not None
                else -1,
            ),
        )
        x = np.arange(len(variant_ids))
        for subset_name in SUBSET_FLAGS:
            subset_rows = {
                row["variant"]: row for row in rows if row["subset"] == subset_name
            }
            values = [subset_rows[variant_id]["macro_f1_mean"] for variant_id in variant_ids]
            errors = [subset_rows[variant_id]["macro_f1_std"] for variant_id in variant_ids]
            axis.errorbar(
                x,
                values,
                yerr=errors,
                marker="o",
                linewidth=1.5,
                capsize=3,
                color=palette[subset_name],
                label=subset_name.replace("train_", "").replace("pct", "%"),
            )
        axis.set_xticks(
            x,
            [variant_by_id[variant_id].display_name for variant_id in variant_ids],
            rotation=45,
            ha="right",
        )
        axis.set_ylim(0, 0.45)
        axis.set_ylabel("Macro F1 (equal weight per direction)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("UAV-P04 classifier margin comparison")
    figure.text(
        0.5,
        0.92,
        "Points are seed means; error bars are ±1 standard deviation; internal test excluded",
        ha="center",
        color="#455a64",
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.88))
    figure.savefig(
        assets_dir / "UAV-P04_classifier_margin-comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    final_rows_30 = [
        row for row in event_rows if row["subset"] == "train_30pct"
    ]
    variants = sorted(
        finalist_ids,
        key=lambda variant_id: (
            variant_by_id[variant_id].margin_fraction is not None,
            variant_by_id[variant_id].margin_fraction
            if variant_by_id[variant_id].margin_fraction is not None
            else -1,
        ),
    )
    x = np.arange(len(variants))
    width = 0.36
    figure, axis = plt.subplots(figsize=(13, 6))
    direction_colors = {"ian_to_ida": "#d69f35", "ida_to_ian": "#285f7a"}
    direction_hatches = {"ian_to_ida": "//", "ida_to_ian": ""}
    for offset, direction in zip((-width / 2, width / 2), DIRECTIONS):
        lookup = {
            row["variant"]: row
            for row in final_rows_30
            if row["direction"] == direction
        }
        values = [lookup[variant_id]["macro_f1_mean"] for variant_id in variants]
        errors = [lookup[variant_id]["macro_f1_std"] for variant_id in variants]
        axis.bar(
            x + offset,
            values,
            width=width,
            yerr=errors,
            capsize=3,
            color=direction_colors[direction],
            edgecolor="#263238",
            hatch=direction_hatches[direction],
            label=f"{DIRECTIONS[direction]['train_event'].replace('Hurricane ', '')}→{DIRECTIONS[direction]['eval_event'].replace('Hurricane ', '')}",
        )
    axis.set_xticks(
        x,
        [variant_by_id[variant_id].display_name for variant_id in variants],
        rotation=45,
        ha="right",
    )
    axis.set_ylim(0, 0.50)
    axis.set_ylabel("Macro F1")
    axis.set_title("UAV-P04 TRAIN-30% performance by held-out orthomosaic")
    axis.text(
        0.5,
        1.02,
        "Five paired seeds; bars are event-specific means; error bars are ±1 standard deviation",
        transform=axis.transAxes,
        ha="center",
        color="#455a64",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        assets_dir / "UAV-P04_classifier_event-comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    final_lookup = {
        row["variant"]: row
        for row in final_aggregate
        if row["subset"] == "train_30pct"
    }
    for axis, metric, title in (
        (axes[0], "recall_major_damage", "Major damage recall"),
        (axes[1], "recall_destroyed", "Destroyed recall"),
    ):
        values = [final_lookup[variant_id][f"{metric}_mean"] for variant_id in variants]
        errors = [final_lookup[variant_id][f"{metric}_std"] for variant_id in variants]
        axis.bar(
            x,
            values,
            yerr=errors,
            capsize=3,
            color="#6f9fb5",
            edgecolor="#263238",
        )
        axis.axhline(
            final_lookup["margin_0"][f"{metric}_mean"] - RELEVANT_RECALL_DROP,
            color="#d69f35",
            linestyle="--",
            linewidth=1.3,
            label="0% minus 5 points",
        )
        axis.set_xticks(
            x,
            [variant_by_id[variant_id].display_name for variant_id in variants],
            rotation=45,
            ha="right",
        )
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Recall (equal weight per direction)")
    axes[1].legend(frameon=False)
    figure.suptitle("UAV-P04 minority-class recall at TRAIN-30%")
    figure.text(
        0.5,
        0.92,
        "Five paired seeds; error bars are ±1 standard deviation",
        ha="center",
        color="#455a64",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(
        assets_dir / "UAV-P04_classifier_minority-recall.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    winner = stability[
        "selected_variant_30pct_after_guardrail_and_tie_rule"
    ]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for axis, direction in zip(axes, DIRECTIONS):
        matrix = aggregate_confusion_for_plot(
            matrix_rows, winner, direction, final_seeds
        )
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        for row_index in range(len(LABELS)):
            for column_index in range(len(LABELS)):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "#263238",
                )
        axis.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
        axis.set_yticks(range(len(LABELS)), LABELS)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(
            f"{DIRECTIONS[direction]['train_event'].replace('Hurricane ', '')}→{DIRECTIONS[direction]['eval_event'].replace('Hurricane ', '')}"
        )
    colorbar_axis = figure.add_axes((0.92, 0.22, 0.018, 0.56))
    figure.colorbar(image, cax=colorbar_axis)
    figure.suptitle(
        f"UAV-P04 normalized confusion matrices: {variant_by_id[winner].display_name}"
        ,
        y=0.98,
    )
    figure.text(
        0.5,
        0.91,
        "TRAIN-30%, counts pooled over five seeds and normalized within each true class",
        ha="center",
        color="#455a64",
    )
    figure.subplots_adjust(
        top=0.78, bottom=0.20, left=0.08, right=0.88, wspace=0.55
    )
    figure.savefig(
        assets_dir / "UAV-P04_classifier_confusion-matrix.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    selected_curves = [
        row
        for row in curve_rows
        if row["variant"] == winner
        and row["subset"] == "train_30pct"
        and int(row["seed"]) in final_seeds
    ]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for direction, color in direction_colors.items():
        direction_rows = [row for row in selected_curves if row["direction"] == direction]
        by_epoch: dict[int, list[float]] = defaultdict(list)
        for row in direction_rows:
            by_epoch[int(row["epoch"])].append(float(row["train_loss"]))
        epochs = sorted(by_epoch)
        means = [statistics.fmean(by_epoch[epoch]) for epoch in epochs]
        stds = [
            statistics.stdev(by_epoch[epoch]) if len(by_epoch[epoch]) > 1 else 0
            for epoch in epochs
        ]
        axis.plot(
            epochs,
            means,
            color=color,
            linewidth=1.8,
            label=direction.replace("_to_", "→"),
        )
        axis.fill_between(
            epochs,
            np.asarray(means) - np.asarray(stds),
            np.asarray(means) + np.asarray(stds),
            color=color,
            alpha=0.18,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Class-weighted training loss")
    axis.set_title(
        f"UAV-P04 training curves: {variant_by_id[winner].display_name}, TRAIN-30%"
    )
    axis.text(
        0.5,
        1.02,
        "Mean ±1 standard deviation over five seeds; fixed final-epoch checkpoint",
        transform=axis.transAxes,
        ha="center",
        color="#455a64",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        assets_dir / "UAV-P04_classifier_training-curves.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def validate_completed_artifacts(
    run_rows: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
    curve_rows: Sequence[dict[str, Any]],
    matrix_rows: Sequence[dict[str, Any]],
    manifest_rows: Sequence[dict[str, Any]],
    initial_variant_ids: set[str],
    finalist_ids: set[str],
    initial_seeds: set[int],
    final_seeds: set[int],
    epochs: int,
) -> dict[str, Any]:
    initial_expected = {
        (variant, subset, direction, seed)
        for variant in initial_variant_ids
        for subset in SUBSET_FLAGS
        for direction in DIRECTIONS
        for seed in initial_seeds
    }
    finalist_expected = {
        (variant, subset, direction, seed)
        for variant in finalist_ids
        for subset in SUBSET_FLAGS
        for direction in DIRECTIONS
        for seed in final_seeds
    }
    expected_keys = initial_expected | finalist_expected
    actual_keys = {
        (
            str(row["variant"]),
            str(row["subset"]),
            str(row["direction"]),
            int(row["seed"]),
        )
        for row in run_rows
    }
    run_ids = [str(row["run_id"]) for row in run_rows]
    predictions_per_run = Counter(str(row["run_id"]) for row in prediction_rows)
    curves_per_run = Counter(str(row["run_id"]) for row in curve_rows)
    matrices_per_run = Counter(str(row["run_id"]) for row in matrix_rows)
    expected_eval_by_run = {
        str(row["run_id"]): int(row["eval_count"]) for row in run_rows
    }
    initialization_hashes: dict[int, set[str]] = defaultdict(set)
    for row in run_rows:
        initialization_hashes[int(row["seed"])].add(
            str(row["initialization_hash"])
        )

    checks = {
        "unique_run_ids": len(set(run_ids)) == len(run_ids),
        "complete_expected_grid": actual_keys == expected_keys,
        "one_initialization_hash_per_seed": all(
            len(values) == 1 for values in initialization_hashes.values()
        ),
        "prediction_count_matches_each_evaluation": all(
            predictions_per_run[run_id] == expected
            for run_id, expected in expected_eval_by_run.items()
        ),
        "curve_count_matches_fixed_epochs": all(
            curves_per_run[run_id] == epochs for run_id in run_ids
        ),
        "confusion_matrix_has_16_cells_per_run": all(
            matrices_per_run[run_id] == len(LABELS) ** 2 for run_id in run_ids
        ),
        "manifest_has_only_train_or_val": {
            str(row["legacy_split"]) for row in manifest_rows
        }
        <= {"train", "val"},
        "manifest_has_only_approved_orthomosaics": {
            str(row["orthomosaic"]) for row in manifest_rows
        }
        == set(ALLOWED_ORTHOMOSAICS),
        "manifest_has_zero_internal_test_rows": sum(
            str(row["legacy_split"]) == "test" for row in manifest_rows
        )
        == 0,
    }
    status = "pass" if all(checks.values()) else "blocked"
    result = {
        "status": status,
        "checks": checks,
        "counts": {
            "expected_unique_runs": len(expected_keys),
            "actual_unique_runs": len(actual_keys),
            "run_rows": len(run_rows),
            "prediction_rows": len(prediction_rows),
            "curve_rows": len(curve_rows),
            "confusion_matrix_rows": len(matrix_rows),
            "manifest_rows": len(manifest_rows),
        },
        "missing_run_keys": sorted(expected_keys - actual_keys),
        "unexpected_run_keys": sorted(actual_keys - expected_keys),
    }
    if status != "pass":
        raise RuntimeError(
            "Completed artifact validation failed: "
            + json.dumps(json_ready(result), sort_keys=True)
        )
    return result


def main() -> None:
    args = parse_args()
    if args.data_root.name != DATASET_NAME:
        raise ValueError(f"--data-root must be the {DATASET_NAME} directory.")
    initial_seeds = parse_seeds(args.initial_seeds)
    final_seeds = parse_seeds(args.final_seeds)
    if not set(initial_seeds) <= set(final_seeds):
        raise ValueError("Every initial seed must be included in final seeds.")
    if args.smoke_epochs < 1:
        raise ValueError("--smoke-epochs must be at least 1.")
    if not args.smoke_test and len(initial_seeds) < 3:
        raise ValueError("The initial search requires at least three seeds.")
    if not args.smoke_test and len(final_seeds) < 5:
        raise ValueError("Finalists require at least five seeds.")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    config = ExperimentConfig(
        input_size=args.input_size,
        epochs=args.smoke_epochs if args.smoke_test else args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        initial_seeds=initial_seeds,
        final_seeds=final_seeds,
        normalization="uint8 / 127.5 - 1.0",
        augmentation="none",
        loss="cross_entropy",
        class_weighting="n / (4 * class_count), computed from each TRAIN subset",
        optimizer="AdamW",
        scheduler="none",
        checkpoint_policy="final epoch; no early stopping",
        phash_threshold=PHASH_THRESHOLD,
        relevant_recall_drop=RELEVANT_RECALL_DROP,
        torch_version=torch.__version__,
        device="cpu",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.assets_dir.mkdir(parents=True, exist_ok=True)

    rows, legacy_counts = load_development_rows(args.manifest)
    annotations, annotation_paths = load_selected_annotations(rows, args.data_root)
    image_paths = image_paths_from_rows(rows, args.data_root)
    samples = build_samples(rows, annotations)
    validate_sample_counts(samples)

    manifest_rows, leakage_audit, duplicate_rows = build_leakage_manifest(
        samples, image_paths, config.input_size
    )
    write_csv(args.output_dir / "uav_p04_looo_manifest.csv", manifest_rows)
    write_csv(args.output_dir / "duplicate_audit_pairs.csv", duplicate_rows)
    (args.output_dir / "leakage_audit.json").write_text(
        json.dumps(json_ready(leakage_audit), indent=2) + "\n",
        encoding="utf-8",
    )
    if leakage_audit["status"] != "pass":
        raise RuntimeError(
            "Leakage audit is blocked. Training was not started. See leakage_audit.json."
        )

    eligible_ids = {row["sample_id"] for row in manifest_rows}
    samples = [sample for sample in samples if sample.sample_id in eligible_ids]
    validate_sample_counts(samples)

    configuration = {
        "experiment_id": "UAV-P04-CLASSIFIER-LOOO",
        "created_date": "2026-07-31",
        "config": asdict(config),
        "architecture": {
            "name": "SmallDamageCNN",
            "initialization": "PyTorch seeded default initialization; paired by seed",
            "pretrained_weights": False,
            "all_weights_trainable": True,
            "blocks": [
                "Conv3x3 3->8, GroupNorm(2), ReLU, MaxPool2",
                "Conv3x3 8->16, GroupNorm(4), ReLU, MaxPool2",
                "Conv3x3 16->32, GroupNorm(8), ReLU, MaxPool2",
                "Conv3x3 32->64, GroupNorm(8), ReLU",
            ],
            "pooling": "adaptive average pooling to 1x1",
            "head": "flatten, dropout 0.20, linear 64->4",
        },
        "roi_rendering": {
            "margin_definition": (
                "add margin_fraction * bbox width to left/right and "
                "margin_fraction * bbox height to top/bottom"
            ),
            "boundary_policy": "clip to GeoTIFF extent",
            "resize": "bilinear RGB; nearest alpha",
            "aspect_ratio": "preserved",
            "square_padding": "symmetric black",
            "invalid_pixels": "alpha == 0 RGB black-filled",
            "masked_control": "0% box with pixels outside polygon black",
            "global_autocontrast": False,
            "median_filter": False,
        },
        "initial_variants": [
            asdict(variant_from_tuple(values)) for values in INITIAL_VARIANTS
        ],
        "directions": DIRECTIONS,
        "expected_train_counts": EXPECTED_TRAIN_COUNTS,
        "expected_eval_counts": EXPECTED_EVAL_COUNTS,
        "legacy_counts": legacy_counts,
        "source_paths": {
            "manifest": str(args.manifest.resolve()),
            "imagery": {key: str(value) for key, value in image_paths.items()},
            "annotations": {
                key: str(value) for key, value in annotation_paths.items()
            },
        },
        "test_access": {
            "internal_test_rows_in_new_manifest": 0,
            "internal_test_annotation_records_selected": 0,
            "internal_test_annotation_objects_decoded": 0,
            "internal_test_pixel_reads": 0,
            "final_event_reads": 0,
        },
        "calibration_policy": (
            "Brier and 10-bin ECE are computed for every run; calibration is "
            "marked viable only when evaluation n >= 100."
        ),
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(json_ready(configuration), indent=2) + "\n",
        encoding="utf-8",
    )

    initial_variants = [variant_from_tuple(values) for values in INITIAL_VARIANTS]
    if args.smoke_test:
        initial_variants = [margin_variant(0.0)]
        initial_seeds = initial_seeds[:1]
        original_subsets = dict(SUBSET_FLAGS)
        original_directions = dict(DIRECTIONS)
        try:
            SUBSET_FLAGS.clear()
            SUBSET_FLAGS["train_10pct"] = "in_train_10pct"
            DIRECTIONS.clear()
            DIRECTIONS["ian_to_ida"] = original_directions["ian_to_ida"]
            run_rows: list[dict[str, Any]] = []
            prediction_rows: list[dict[str, Any]] = []
            curve_rows: list[dict[str, Any]] = []
            matrix_rows: list[dict[str, Any]] = []
            roi_diagnostic_rows: list[dict[str, Any]] = []
            run_grid(
                initial_variants,
                initial_seeds,
                "smoke_test",
                samples,
                image_paths,
                config,
                args.output_dir,
                run_rows,
                prediction_rows,
                curve_rows,
                matrix_rows,
                set(),
                {},
                roi_diagnostic_rows,
            )
            write_csv(
                args.output_dir / "roi_diagnostic_summary.csv",
                aggregate_roi_diagnostics(roi_diagnostic_rows),
            )
        finally:
            SUBSET_FLAGS.clear()
            SUBSET_FLAGS.update(original_subsets)
            DIRECTIONS.clear()
            DIRECTIONS.update(original_directions)
        print("SMOKE TEST COMPLETE", flush=True)
        return

    all_run_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    all_curve_rows: list[dict[str, Any]] = []
    all_matrix_rows: list[dict[str, Any]] = []
    all_roi_diagnostic_rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, str, str, int]] = set()
    image_cache: dict[str, dict[str, np.ndarray]] = {}

    experiment_start = time.perf_counter()
    run_grid(
        variants=initial_variants,
        seeds=initial_seeds,
        phase="initial_search",
        samples=samples,
        image_paths=image_paths,
        config=config,
        output_dir=args.output_dir,
        all_run_rows=all_run_rows,
        all_prediction_rows=all_prediction_rows,
        all_curve_rows=all_curve_rows,
        all_matrix_rows=all_matrix_rows,
        completed_keys=completed_keys,
        image_cache=image_cache,
        all_roi_diagnostic_rows=all_roi_diagnostic_rows,
    )

    initial_seed_rows = equal_direction_seed_rows(
        all_run_rows,
        {variant.variant_id for variant in initial_variants},
        set(initial_seeds),
    )
    initial_aggregate = aggregate_seed_rows(initial_seed_rows)
    write_csv(args.output_dir / "initial_equal_direction_seed_metrics.csv", initial_seed_rows)
    write_csv(args.output_dir / "initial_margin_comparison.csv", initial_aggregate)

    refined_variants, finalist_ids, preliminary_selection = select_refinement_variants(
        initial_aggregate
    )
    variant_by_id = {variant.variant_id: variant for variant in initial_variants}
    variant_by_id.update(
        {variant.variant_id: variant for variant in refined_variants}
    )
    finalist_variants = [variant_by_id[variant_id] for variant_id in sorted(finalist_ids)]
    print(
        "PRELIMINARY SELECTION "
        + json.dumps(json_ready(preliminary_selection), sort_keys=True),
        flush=True,
    )

    run_grid(
        variants=finalist_variants,
        seeds=final_seeds,
        phase="finalist_and_refinement",
        samples=samples,
        image_paths=image_paths,
        config=config,
        output_dir=args.output_dir,
        all_run_rows=all_run_rows,
        all_prediction_rows=all_prediction_rows,
        all_curve_rows=all_curve_rows,
        all_matrix_rows=all_matrix_rows,
        completed_keys=completed_keys,
        image_cache=image_cache,
        all_roi_diagnostic_rows=all_roi_diagnostic_rows,
    )

    final_seed_rows = equal_direction_seed_rows(
        all_run_rows, finalist_ids, set(final_seeds)
    )
    final_aggregate = aggregate_seed_rows(final_seed_rows)
    event_rows = event_aggregate(all_run_rows, finalist_ids, set(final_seeds))
    best_by_subset = choose_empirical_best(
        final_aggregate, final_seed_rows, variant_by_id
    )
    stability = stability_assessment(
        final_aggregate,
        event_rows,
        final_seed_rows,
        best_by_subset,
        variant_by_id,
    )

    paired_rows: list[dict[str, Any]] = []
    for subset_name in SUBSET_FLAGS:
        for variant_id in sorted(finalist_ids):
            for baseline in ("margin_0", "margin_10"):
                paired_rows.append(
                    paired_difference(
                        final_seed_rows,
                        variant_id,
                        baseline,
                        subset_name,
                        "macro_f1",
                    )
                )
    write_csv(args.output_dir / "final_equal_direction_seed_metrics.csv", final_seed_rows)
    write_csv(args.output_dir / "final_margin_comparison.csv", final_aggregate)
    write_csv(args.output_dir / "event_metrics.csv", event_rows)
    write_csv(args.output_dir / "paired_differences.csv", paired_rows)
    write_csv(
        args.output_dir / "roi_diagnostic_summary.csv",
        aggregate_roi_diagnostics(all_roi_diagnostic_rows),
    )
    persist_raw_results(
        args.output_dir,
        all_run_rows,
        all_prediction_rows,
        all_curve_rows,
        all_matrix_rows,
    )

    create_figures(
        assets_dir=args.assets_dir,
        initial_aggregate=initial_aggregate,
        final_aggregate=final_aggregate,
        event_rows=event_rows,
        matrix_rows=all_matrix_rows,
        curve_rows=all_curve_rows,
        stability=stability,
        finalist_ids=finalist_ids,
        final_seeds=set(final_seeds),
        variant_by_id=variant_by_id,
    )

    initialization_hashes: dict[str, set[str]] = defaultdict(set)
    for row in all_run_rows:
        initialization_hashes[str(row["seed"])].add(row["initialization_hash"])
    if any(len(values) != 1 for values in initialization_hashes.values()):
        raise RuntimeError("Paired runs did not share one initialization per seed.")

    artifact_validation = validate_completed_artifacts(
        run_rows=all_run_rows,
        prediction_rows=all_prediction_rows,
        curve_rows=all_curve_rows,
        matrix_rows=all_matrix_rows,
        manifest_rows=manifest_rows,
        initial_variant_ids={
            variant.variant_id for variant in initial_variants
        },
        finalist_ids=finalist_ids,
        initial_seeds=set(initial_seeds),
        final_seeds=set(final_seeds),
        epochs=config.epochs,
    )
    (args.output_dir / "artifact_validation.json").write_text(
        json.dumps(json_ready(artifact_validation), indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "experiment_id": "UAV-P04-CLASSIFIER-LOOO",
        "status": "complete",
        "decision": stability["decision"],
        "best_by_subset": best_by_subset,
        "stability": stability,
        "preliminary_selection": preliminary_selection,
        "variants": [asdict(variant) for variant in variant_by_id.values()],
        "run_counts": {
            "completed_unique_runs": len(all_run_rows),
            "initial_search_expected": len(initial_variants)
            * len(SUBSET_FLAGS)
            * len(DIRECTIONS)
            * len(initial_seeds),
            "failed_runs": 0,
        },
        "seeds": {
            "initial": list(initial_seeds),
            "final": list(final_seeds),
            "initialization_hashes": {
                seed: next(iter(values))
                for seed, values in initialization_hashes.items()
            },
        },
        "leakage_audit": leakage_audit,
        "artifact_validation": artifact_validation,
        "test_access": configuration["test_access"],
        "limitations": [
            "Only two orthomosaics are available.",
            "Ida TRAIN subsets contain only 11, 22, and 33 buildings.",
            "The Ida evaluation fold contains only 22 buildings.",
            "The reference CNN is trained from scratch because no local pretrained weights are available.",
            "Native flight/sequence identifiers are unavailable; orthomosaic is used as a conservative surrogate group.",
            "Calibration for the 22-building Ida evaluation fold is not statistically viable.",
            "The selected result is the best among tested margins, not a continuous mathematical optimum.",
        ],
        "execution": {
            "duration_seconds": time.perf_counter() - experiment_start,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    print("EXPERIMENT COMPLETE", flush=True)
    print(json.dumps(json_ready(summary["stability"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
