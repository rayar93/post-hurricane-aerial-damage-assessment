# UAV missing-data validation

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
