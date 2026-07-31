#!/usr/bin/env python3

"""
Finalize and document UAV-P01: alpha-based missing-data handling.

This script:
1. writes the final JSON and CSV results,
2. creates the quantitative comparison figure,
3. writes the detailed Markdown report,
4. updates the UAV preprocessing registry,
5. adds the report link to the README.

It does not download data or inspect the frozen test split.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = ROOT / "UAV_PREPROCESSING_EXPERIMENTS.md"
README_PATH = ROOT / "README.md"

REPORT_DIR = ROOT / "reports/preprocessing/uav_missing_data"
DOCS_DIR = ROOT / "docs/preprocessing"
ASSETS_DIR = ROOT / "docs/assets/uav-preprocessing"

SUMMARY_PATH = REPORT_DIR / "summary.json"
FINDINGS_PATH = REPORT_DIR / "key_findings.csv"
REPORT_PATH = DOCS_DIR / "uav_missing_data_validation.md"
FIGURE_PATH = ASSETS_DIR / "UAV-P01_metric-comparison.png"


RESULTS = {
    "experiment_id": "UAV-P01",
    "name": "Alpha validity filtering and invalid-region fill",
    "status": "COMPLETE",
    "decision": "KEEP",
    "dataset": {
        "events": ["Hurricane Ian", "Hurricane Ida"],
        "source_annotations": 699,
        "eligible_buildings": 690,
        "class_counts": {
            "no damage": 256,
            "minor damage": 242,
            "major damage": 126,
            "destroyed": 66,
        },
    },
    "split": {
        "train": 414,
        "validation": 138,
        "test": 138,
    },
    "nested_train_subsets": {
        "10_percent": 69,
        "20_percent": 138,
        "30_percent": 207,
    },
    "train_findings": {
        "valid_dark_gt_1pct_count": 144,
        "valid_dark_gt_1pct_fraction": 144 / 414,
        "valid_dark_gt_10pct_count": 3,
        "valid_dark_gt_10pct_fraction": 3 / 414,
        "polygon_any_nodata_count": 15,
        "polygon_any_nodata_fraction": 15 / 414,
        "polygon_gt_40pct_count": 5,
        "polygon_gt_40pct_fraction": 5 / 414,
        "polygon_gt_50pct_count": 0,
        "polygon_gt_50pct_fraction": 0.0,
        "crop_nodata_mean": 0.013592,
        "crop_nodata_median": 0.0,
        "crop_nodata_max": 0.482577,
        "polygon_nodata_mean": 0.012548,
        "polygon_nodata_median": 0.0,
        "polygon_nodata_p95": 0.0,
        "polygon_nodata_max": 0.472647,
    },
    "materialization_10pct": {
        "train_samples": 69,
        "validation_samples": 138,
        "test_samples": 0,
        "total_samples": 207,
        "samples_with_ignore_255": 6,
        "maximum_ignore_fraction": 0.437805,
        "train_validation_overlap": 0,
        "mask_values": [0, 1, 2, 3, 4, 255],
    },
    "final_policy": {
        "validity_source": "GeoTIFF alpha",
        "valid_condition": "alpha > 0",
        "invalid_condition": "alpha == 0",
        "rgb_fill": [0, 0, 0],
        "target_ignore_index": 255,
        "current_samples_rejected": 0,
        "future_rejection_condition": (
            "building_polygon_invalid_fraction > 0.50"
        ),
        "dark_pixel_heuristic": "REJECT",
    },
}


def percentage(count: int, denominator: int) -> float:
    return 100.0 * count / denominator


def ensure_directories() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def write_summary() -> None:
    SUMMARY_PATH.write_text(
        json.dumps(RESULTS, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_findings_csv() -> None:
    rows = [
        {
            "metric": "eligible_buildings",
            "count": 690,
            "denominator": "",
            "percentage": "",
            "value": 690,
            "unit": "buildings",
        },
        {
            "metric": "valid_dark_gt_1pct",
            "count": 144,
            "denominator": 414,
            "percentage": percentage(144, 414),
            "value": 144,
            "unit": "buildings",
        },
        {
            "metric": "valid_dark_gt_10pct",
            "count": 3,
            "denominator": 414,
            "percentage": percentage(3, 414),
            "value": 3,
            "unit": "buildings",
        },
        {
            "metric": "polygon_any_alpha_nodata",
            "count": 15,
            "denominator": 414,
            "percentage": percentage(15, 414),
            "value": 15,
            "unit": "buildings",
        },
        {
            "metric": "polygon_alpha_nodata_gt_40pct",
            "count": 5,
            "denominator": 414,
            "percentage": percentage(5, 414),
            "value": 5,
            "unit": "buildings",
        },
        {
            "metric": "polygon_alpha_nodata_gt_50pct",
            "count": 0,
            "denominator": 414,
            "percentage": 0.0,
            "value": 0,
            "unit": "buildings",
        },
        {
            "metric": "crop_nodata_mean",
            "count": "",
            "denominator": "",
            "percentage": 1.3592,
            "value": 0.013592,
            "unit": "fraction",
        },
        {
            "metric": "crop_nodata_max",
            "count": "",
            "denominator": "",
            "percentage": 48.2577,
            "value": 0.482577,
            "unit": "fraction",
        },
        {
            "metric": "polygon_nodata_mean",
            "count": "",
            "denominator": "",
            "percentage": 1.2548,
            "value": 0.012548,
            "unit": "fraction",
        },
        {
            "metric": "polygon_nodata_max",
            "count": "",
            "denominator": "",
            "percentage": 47.2647,
            "value": 0.472647,
            "unit": "fraction",
        },
        {
            "metric": "samples_with_ignore_255",
            "count": 6,
            "denominator": 207,
            "percentage": percentage(6, 207),
            "value": 6,
            "unit": "samples",
        },
        {
            "metric": "maximum_ignore_fraction",
            "count": "",
            "denominator": "",
            "percentage": 43.7805,
            "value": 0.437805,
            "unit": "fraction",
        },
    ]

    with FINDINGS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "count",
                "denominator",
                "percentage",
                "value",
                "unit",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def create_figure() -> None:
    labels = [
        ">1% valid dark pixels",
        ">10% valid dark pixels",
        "Any polygon alpha no-data",
        ">40% polygon alpha no-data",
        ">50% polygon alpha no-data",
    ]
    values = [
        percentage(144, 414),
        percentage(3, 414),
        percentage(15, 414),
        percentage(5, 414),
        0.0,
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(labels, values)

    ax.invert_yaxis()
    ax.set_xlabel("Percentage of TRAIN buildings")
    ax.set_title("UAV-P01: RGB darkness versus alpha no-data")
    ax.set_xlim(0, 40)
    ax.grid(axis="x", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
        )

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report() -> None:
    report = """# UAV missing-data validation

## Experiment

**Registry ID:** UAV-P01  
**Preprocessing step:** alpha validity filtering and invalid-region fill  
**Final status:** `COMPLETE — KEEP`

## Objective

Determine how unavailable pixels should be detected and handled in UAV
building-damage imagery.

The previous approach treated very dark RGB pixels as missing data. This
experiment evaluates the GeoTIFF alpha channel as the authoritative
validity signal.

## Dataset

The development corpus contains 690 eligible buildings from Hurricane Ian
and Hurricane Ida.

| Damage class | Buildings |
|---|---:|
| No damage | 256 |
| Minor damage | 242 |
| Major damage | 126 |
| Destroyed | 66 |
| **Total** | **690** |

The frozen split contains 414 train, 138 validation, and 138 test buildings.
Nested training subsets contain 69 buildings at 10%, 138 at 20%, and 207 at
30%.

The frozen test split was not inspected or materialized while selecting this
preprocessing rule.

## Hypothesis

The GeoTIFF alpha channel identifies pixels where imagery does not exist.

RGB darkness cannot be used as a missing-data indicator because roofs,
shadows, roads, vegetation, and background areas can be naturally dark.

## Quantitative findings

All preprocessing-selection statistics were calculated from TRAIN.

### RGB darkness versus alpha validity

| TRAIN check | Count | Percentage |
|---|---:|---:|
| More than 1% valid dark pixels | 144 / 414 | 34.78% |
| More than 10% valid dark pixels | 3 / 414 | 0.72% |
| Any alpha-zero pixel inside the building polygon | 15 / 414 | 3.62% |
| More than 40% alpha-zero inside the polygon | 5 / 414 | 1.21% |
| More than 50% alpha-zero inside the polygon | 0 / 414 | 0.00% |

Valid dark pixels were much more common than actual alpha no-data inside
building polygons. A dark-pixel filter would therefore remove or penalize
valid visual evidence.

### Missing-data distribution

| Statistic | Complete crop | Building polygon |
|---|---:|---:|
| Mean missing fraction | 1.3592% | 1.2548% |
| Median missing fraction | 0.0000% | 0.0000% |
| 95th percentile | — | 0.0000% |
| Maximum missing fraction | 48.2577% | 47.2647% |

No TRAIN building exceeded 50% invalid coverage inside its polygon.

The proposed 50%, 75%, and 90% rejection policies therefore remove zero
TRAIN samples and produce identical datasets. Separate model runs for those
thresholds would not test a real intervention.

## Materialization validation

The nested 10% materialization contained:

- 69 training samples;
- all 138 validation samples;
- zero test samples;
- zero train/validation overlap;
- target-mask values `[0, 1, 2, 3, 4, 255]`;
- six samples containing `IGNORE=255`;
- maximum resized ignore fraction of 43.7805%.

This confirms that alpha-invalid pixels remain excluded from the learning
target after materialization and resizing.

## Final implementation

```python
valid = alpha > 0
rgb[~valid] = (0, 0, 0)
target_mask[~valid] = 255
```

Black is only a deterministic RGB placeholder. It is not used to decide
whether a pixel is valid.

For future data, the conservative safeguard is:

```python
reject = building_polygon_invalid_fraction > 0.50
```

This safeguard removes zero samples from the current development corpus.

## Fill-method decision

Training-mean and nearest-valid fills were not retained. They synthesize
unavailable visual content without demonstrated need, while alpha-invalid
target pixels are already excluded from the loss using `IGNORE=255`.

Black fill is deterministic, simple, and clearly represents unavailable
imagery when the alpha mask is preserved.

## Final decision

| Component | Decision |
|---|---|
| Alpha validity mechanism | `KEEP` |
| RGB dark-pixel heuristic | `REJECT` |
| Current building rejection | Keep all 690 eligible buildings |
| Future rejection threshold | More than 50% invalid polygon coverage |
| Invalid RGB fill | Black `(0, 0, 0)` |
| Invalid target handling | `IGNORE=255` |
| UAV-P01 status | `COMPLETE` |

## Evidence files

- `docs/assets/uav-preprocessing/UAV-P01_metric-comparison.png`
- `reports/preprocessing/uav_missing_data/key_findings.csv`
- `reports/preprocessing/uav_missing_data/summary.json`

## Reproduction

```bash
python -m unittest tests/test_uav_alpha_validity.py
python scripts/export_uav_missing_data_report.py
```
"""

    REPORT_PATH.write_text(report, encoding="utf-8")


def update_registry() -> None:
    text = REGISTRY_PATH.read_text(encoding="utf-8")

    table_pattern = re.compile(
        r"\| UAV-P01 \| Alpha validity filtering and "
        r"invalid-region fill \|[^\n]*"
    )
    table_replacement = (
        "| UAV-P01 | Alpha validity filtering and invalid-region fill | "
        "Deterministic | 1 | Complete | `KEEP` | Internal UAV audit |"
    )

    if table_pattern.search(text):
        text = table_pattern.sub(table_replacement, text, count=1)

    start_marker = (
        "## UAV-P01 — Alpha validity filtering and invalid-region fill"
    )
    end_marker = "## UAV-P02 — Global auto-contrast"

    if start_marker not in text:
        raise RuntimeError("UAV-P01 section not found in registry.")

    if end_marker not in text:
        raise RuntimeError("UAV-P02 section not found in registry.")

    start = text.index(start_marker)
    end = text.index(end_marker)

    replacement = """## UAV-P01 — Alpha validity filtering and invalid-region fill

**Status:** `COMPLETE`

**Decision:** `KEEP`

**Hypothesis:** GeoTIFF alpha distinguishes unavailable imagery from valid
dark RGB pixels.

**Development corpus:** 690 eligible UAV buildings from Hurricane Ian and
Hurricane Ida. The frozen split contains 414 train, 138 validation, and
138 test buildings.

### TRAIN findings

| Check | Result |
|---|---:|
| More than 1% valid dark pixels | 144 / 414 (34.78%) |
| More than 10% valid dark pixels | 3 / 414 (0.72%) |
| Any polygon alpha no-data | 15 / 414 (3.62%) |
| More than 40% polygon alpha no-data | 5 / 414 (1.21%) |
| More than 50% polygon alpha no-data | 0 / 414 (0.00%) |
| Maximum polygon alpha no-data | 47.2647% |

The complete-crop no-data mean was 1.3592%. The building-polygon no-data
mean was 1.2548%. Polygon median and 95th percentile were both zero.

The 50%, 75%, and 90% rejection policies remove zero TRAIN buildings and
would produce identical training datasets.

### Materialization validation

The nested 10% materialization contained 69 training samples, 138 validation
samples, zero test samples, zero train/validation overlap, six samples with
`IGNORE=255`, and a maximum resized ignore fraction of 43.7805%.

### Final policy

- use `alpha > 0` as the valid-pixel mask;
- reject the RGB dark-pixel heuristic;
- black-fill RGB pixels where `alpha == 0`;
- assign `IGNORE=255` to invalid target-mask pixels;
- retain every building in the current development corpus;
- reject future crops only when more than 50% of the building polygon is
  alpha-invalid.

Black is a deterministic placeholder, not a missing-data detector.

**Evidence:**

- `docs/preprocessing/uav_missing_data_validation.md`
- `docs/assets/uav-preprocessing/UAV-P01_metric-comparison.png`
- `reports/preprocessing/uav_missing_data/key_findings.csv`
- `reports/preprocessing/uav_missing_data/summary.json`

**Final result:** alpha validity `KEEP`; dark-pixel heuristic `REJECT`;
current samples rejected: `0`.

"""

    updated = text[:start] + replacement + text[end:]
    REGISTRY_PATH.write_text(updated, encoding="utf-8")


def update_readme() -> None:
    text = README_PATH.read_text(encoding="utf-8")

    report_line = (
        "- [UAV missing-data validation]"
        "(docs/preprocessing/uav_missing_data_validation.md): "
        "alpha validity findings and final UAV-P01 decision."
    )

    if report_line in text:
        return

    registry_token = "UAV_PREPROCESSING_EXPERIMENTS.md"
    position = text.find(registry_token)

    if position == -1:
        text += "\n## Research documentation\n\n"
        text += report_line + "\n"
    else:
        line_end = text.find("\n", position)

        if line_end == -1:
            line_end = len(text)

        text = (
            text[: line_end + 1]
            + report_line
            + "\n"
            + text[line_end + 1 :]
        )

    README_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_directories()
    write_summary()
    write_findings_csv()
    create_figure()
    write_report()
    update_registry()
    update_readme()

    print("UAV-P01 export complete")
    print("-----------------------")
    print(REPORT_PATH.relative_to(ROOT))
    print(FIGURE_PATH.relative_to(ROOT))
    print(FINDINGS_PATH.relative_to(ROOT))
    print(SUMMARY_PATH.relative_to(ROOT))
    print()
    print("Final decision: COMPLETE — KEEP")


if __name__ == "__main__":
    main()
