# UAV Preprocessing Experiment Registry

**Scope:** high-resolution UAV imagery for building-level post-hurricane damage classification  
**Owner:** Miguel Ángel  
**Created:** 2026-07-30  
**Last updated:** 2026-07-30

This document is the single source of truth for UAV preprocessing decisions. Every candidate must be documented here before it enters the final pipeline. After each experiment, add the quantitative results, one representative image, and a final `KEEP` or `REJECT` decision.

## Decision rules

A step is retained only when it has a dataset-specific hypothesis, a controlled comparison against the same baseline, a consistent quantitative or operational benefit, no unacceptable regression in minority classes, and visual evidence of its effect.

Decision labels:

- `KEEP`: include in the frozen UAV pipeline.
- `REJECT`: exclude because it degrades performance, adds no meaningful benefit, or adds unnecessary complexity.
- `PENDING`: not evaluated yet.
- `NEEDS FULL DATA`: promising on subsets but not confirmed on the complete development set.
- `BLOCKED`: required data or code is unavailable.

A step that leaves results effectively unchanged should normally be rejected.

## Experimental protocol

Unless explicitly stated otherwise, all comparisons must use the same grouped train/validation/test manifests, model, pretrained weights, optimizer, scheduler, epoch budget, seeds, and metrics.

1. Test on stratified 10% data.
2. Re-test viable candidates on stratified 20% and 30% data.
3. Confirm only the strongest candidates on the full development set with at least three seeds.
4. Never use the reserved unseen disaster event for preprocessing selection.
5. Apply random augmentation to training data only.
6. Compute dataset statistics from training data only.

Required metrics:

- macro F1 and weighted F1;
- balanced accuracy;
- per-class precision, recall, and F1;
- confusion matrix;
- ordinal MAE when using ordered damage labels;
- training/inference cost when materially affected.

## Visual evidence convention

Every completed step must include at least one raw-versus-transformed comparison. Store figures under:

```text
docs/assets/uav-preprocessing/
```

Use these filenames:

```text
<STEP-ID>_visual-comparison.png
<STEP-ID>_metric-comparison.png
<STEP-ID>_confusion-matrix.png
```

The visual comparison must use the same samples across variants and include relevant classes or failure modes.

## Mandatory leakage controls

These are evaluation requirements, not optional ablations:

- keep the same building, orthomosaic, flight/sequence, duplicate cluster, and disaster event inside one split;
- detect exact and perceptual near-duplicates before splitting;
- restrict augmentation to training data;
- keep the final test event frozen until preprocessing and model selection are complete.

## Candidate summary

| ID | Candidate | Type | Priority | Status | Decision | Main source |
|---|---|---|---:|---|---|---|
| UAV-P00 | Dark-pixel invalid-image heuristic | Deterministic | Complete | Complete | `REJECT` | Internal UAV audit |
| UAV-P01 | Alpha validity filtering and invalid-region fill | Deterministic | 1 | Complete | `KEEP` | Internal UAV audit |
| UAV-P02 | Global auto-contrast | Deterministic | Complete | Complete | `REJECT` | Internal ablation |
| UAV-P03 | Median 3×3 denoising | Deterministic | Complete | Complete | `REJECT` | Internal 10/20/30% ablation |
| UAV-P04 | Building ROI and context margin | Deterministic | 1 | Planned | `PENDING` | Hasan et al. (2026) |
| UAV-P05 | Polygon alignment and buffer | Deterministic | 1 | Planned | `PENDING` | Manzini et al. (2024–2026) |
| UAV-P06 | Input resolution, aspect ratio, padding, interpolation | Deterministic | 1 | Planned | `PENDING` | Corley et al. (2024) |
| UAV-P07 | GSD normalization and scale robustness | Deterministic / augmentation | 1 | Planned | `PENDING` | Manzini et al. (2025–2026) |
| UAV-P08 | RGB normalization matched to pretraining | Deterministic | 1 | Planned | `PENDING` | Corley et al. (2024); LADI v2 |
| UAV-P09 | Exterior treatment for masked buildings | Deterministic | 2 | Planned | `PENDING` | Dataset-specific hypothesis |
| UAV-A01 | Flips and rotations | Train-only augmentation | 1 | Planned | `PENDING` | Fan (2026) |
| UAV-A02 | Brightness, contrast, gamma, saturation | Train-only augmentation | 2 | Planned | `PENDING` | Fan (2026) |
| UAV-A03 | Translation and crop jitter | Train-only augmentation | 1 | Planned | `PENDING` | Fan (2026); alignment studies |
| UAV-A04 | Zoom and resolution degradation | Train-only augmentation | 1 | Planned | `PENDING` | Fan (2026); operational studies |
| UAV-A05 | Synthetic shadows | Train-only augmentation | 2 | Planned | `PENDING` | Fan (2026) |
| UAV-M01 | DSM height and slope channels | Additional modality | 3 | Planned | `PENDING` | Dataset-specific hypothesis |

# Completed decisions

## UAV-P00 — Dark-pixel invalid-image heuristic

**Hypothesis:** crops with many near-black pixels may be unusable.

**Finding:** the rule confused valid dark structures/backgrounds with nonexistent orthomosaic regions. Approximately 34% of crops had more than 1% very dark pixels, while only a small fraction was genuinely unusable.

**Image:** pending migration to `docs/assets/uav-preprocessing/UAV-P00_visual-comparison.png`.

**Decision:** `REJECT`. RGB darkness is not a reliable validity mask.

## UAV-P01 — Alpha validity filtering and invalid-region fill

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

## UAV-P02 — Global auto-contrast

**Hypothesis:** contrast expansion may reveal subtle damage.

**Finding:** no useful visual or model improvement.

**Image:** `docs/assets/uav-preprocessing/UAV-P02_visual-comparison.png`.

**Decision:** `REJECT`.

## UAV-P03 — Median 3×3 denoising

**Hypothesis:** removing impulse noise may make damage features easier to learn.

**Finding:** inconsistent results: improvement on some 10% and 20% comparisons but degradation at 30%. Noise is not the main dataset problem and fine roof/debris information may be lost.

**Image:** `docs/assets/uav-preprocessing/UAV-P03_visual-comparison.png`.

**Decision:** `REJECT`.

# Priority-1 deterministic experiments

## UAV-P04 — Building ROI and context margin

**Source and rationale:** Hasan et al. (2026) use building-centric damage representations and show that surrounding spatial context can add useful information. Moderate context may preserve debris, fallen vegetation, flooding, displaced materials, and nearby damage; excessive context may distract the classifier.

Variants:

- current rectangular bounding box;
- exact building mask only;
- building plus 10% margin;
- building plus 25% margin;
- building plus 50% margin.

Preserve aspect ratio and pad to square instead of stretching.

**Image:** `docs/assets/uav-preprocessing/UAV-P04_visual-comparison.png`.

| Variant | 10% macro F1 | 20% macro F1 | 30% macro F1 | Full-data macro F1 | Minority-class effect |
|---|---:|---:|---:|---:|---|
| Bounding box | — | — | — | — | — |
| Exact mask | — | — | — | — | — |
| +10% context | — | — | — | — | — |
| +25% context | — | — | — | — | — |
| +50% context | — | — | — | — | — |

**Decision:** `PENDING`.

## UAV-P05 — Polygon alignment and buffer

**Source and rationale:** CRASAR-U-DROIDs contains 7,880 building-polygon adjustment annotations, and later operational work identifies spatial misalignment as a major deployment problem. Small offsets may remove relevant roof pixels or include the wrong background.

Before training, audit at least 200 buildings stratified by orthomosaic and class.

Variants:

- original polygon;
- corrected polygon where available;
- 5% exterior buffer;
- 10% exterior buffer;
- best deterministic variant plus ±5% train-time translation jitter.

**Image:** `docs/assets/uav-preprocessing/UAV-P05_visual-comparison.png` including well-, mildly-, and strongly-misaligned cases.

| Variant | Coverage statistic | 10% macro F1 | 20% macro F1 | 30% macro F1 | Full-data macro F1 |
|---|---:|---:|---:|---:|---:|
| Original | — | — | — | — | — |
| Corrected | — | — | — | — | — |
| 5% buffer | — | — | — | — | — |
| 10% buffer | — | — | — | — | — |
| Best + jitter | — | — | — | — | — |

**Decision:** `PENDING`.

## UAV-P06 — Input resolution, aspect ratio, padding, interpolation

**Source and rationale:** Corley et al. (2024) show that resizing and normalization materially affect remote-sensing transfer learning. Small inputs may erase subtle damage; very large inputs may increase cost and overfitting.

Variants:

- 256×256;
- 384×384;
- 512×512;
- 768×768;
- 1024×1024 if feasible;
- direct stretch to square as a negative/control comparison.

Preserve aspect ratio and pad. Compare area interpolation against the backbone default for RGB downsampling. Use nearest-neighbor for masks.

**Image:** `docs/assets/uav-preprocessing/UAV-P06_visual-comparison.png`, including zoomed roof detail.

| Size/configuration | Macro F1 | Balanced accuracy | Time/epoch | Peak memory |
|---|---:|---:|---:|---:|
| 256 | — | — | — | — |
| 384 | — | — | — | — |
| 512 | — | — | — | — |
| 768 | — | — | — | — |
| 1024 | — | — | — | — |
| Stretch control | — | — | — | — |

**Decision:** `PENDING`.

## UAV-P07 — GSD normalization and scale robustness

**Source and rationale:** Manzini et al. identify input spatial-resolution variation as a major operational challenge. Models may incorrectly associate pixel scale with damage.

Variants:

- direct resize from original crops;
- resample to 3 cm/pixel before model resize;
- resample to 2 cm/pixel before model resize;
- train-time GSD/scale augmentation approximating 2–6 cm/pixel;
- multi-scale training at 384, 512, and 768.

**Image:** `docs/assets/uav-preprocessing/UAV-P07_visual-comparison.png`.

| Variant | In-distribution macro F1 | Unseen-event macro F1 | Performance by GSD bin |
|---|---:|---:|---|
| Direct resize | — | — | — |
| 3 cm/pixel | — | — | — |
| 2 cm/pixel | — | — | — |
| GSD augmentation | — | — | — |
| Multi-scale | — | — | — |

**Decision:** `PENDING`.

## UAV-P08 — RGB normalization matched to pretraining

**Source and rationale:** Corley et al. (2024) show that matching input size and normalization to pretraining can substantially improve remote-sensing transfer learning. LADI v2 uses model-specific preprocessing for low-altitude disaster imagery.

Variants:

- RGB divided by 255 only;
- ImageNet mean/std;
- UAV training-set mean/std;
- exact processor distributed with the selected pretrained backbone.

**Image:** `docs/assets/uav-preprocessing/UAV-P08_visual-comparison.png` or a channel-distribution plot.

| Variant | Macro F1 | Convergence speed | Calibration |
|---|---:|---:|---:|
| [0,1] only | — | — | — |
| ImageNet | — | — | — |
| UAV statistics | — | — | — |
| Backbone processor | — | — | — |

**Decision:** `PENDING`.

## UAV-P09 — Exterior treatment for masked buildings

**Hypothesis:** black mask exteriors may create artificial edges or remove useful context.

Variants:

- black;
- training-set mean RGB;
- blurred context;
- attenuated context;
- unchanged context.

**Image:** `docs/assets/uav-preprocessing/UAV-P09_visual-comparison.png`.

| Variant | Macro F1 | Minority-class F1 | Mask-edge shortcut evidence |
|---|---:|---:|---|
| Black | — | — | — |
| Mean RGB | — | — | — |
| Blurred context | — | — | — |
| Attenuated context | — | — | — |
| Unchanged context | — | — | — |

**Decision:** `PENDING`.

# Train-only augmentation experiments

## UAV-A01 — Flips and rotations

**Source and rationale:** Fan (2026) supports augmentation in high-resolution UAV classification. Overhead building damage should normally be invariant to cardinal orientation. Exact right-angle rotations preserve detail better than large interpolated rotations.

Variants:

- none;
- horizontal flip;
- horizontal and vertical flips;
- flips plus 90°/180°/270° rotations;
- previous best plus continuous ±15°;
- previous best plus continuous ±40°.

**Image:** `docs/assets/uav-preprocessing/UAV-A01_visual-comparison.png`.

| Variant | Macro F1 | Ordinal MAE | Train–validation gap |
|---|---:|---:|---:|
| None | — | — | — |
| H flip | — | — | — |
| H/V flips | — | — | — |
| Right-angle rotations | — | — | — |
| ±15° | — | — | — |
| ±40° | — | — | — |

**Decision:** `PENDING`.

## UAV-A02 — Brightness, contrast, gamma, saturation

**Source and rationale:** Fan (2026) evaluates illumination-oriented augmentation for high-resolution UAV imagery. Mild changes may reduce dependence on one flight's sunlight and exposure; strong changes may remove real damage cues.

Variants:

- none;
- brightness ±10%;
- brightness and contrast ±10%;
- previous plus gamma 0.9–1.1;
- previous plus saturation ±10%.

**Image:** `docs/assets/uav-preprocessing/UAV-A02_visual-comparison.png`.

| Variant | Macro F1 | Per-event stability | Per-class effect |
|---|---:|---:|---|
| None | — | — | — |
| Brightness | — | — | — |
| Brightness + contrast | — | — | — |
| + gamma | — | — | — |
| + saturation | — | — | — |

**Decision:** `PENDING`.

## UAV-A03 — Translation and crop jitter

**Source and rationale:** Fan (2026) uses spatial shifts, while CRASAR work shows alignment errors are operationally relevant. Small jitter may improve robustness; aggressive shifts may remove the target building.

Variants:

- none;
- ±2% translation;
- ±5% translation;
- ±10% translation;
- random crop retaining at least 90% of the ROI;
- random crop retaining at least 80% of the ROI.

**Image:** `docs/assets/uav-preprocessing/UAV-A03_visual-comparison.png`.

| Variant | Macro F1 | Robustness to synthetic offset | Missed-building rate |
|---|---:|---:|---:|
| None | — | — | — |
| ±2% | — | — | — |
| ±5% | — | — | — |
| ±10% | — | — | — |
| 90% ROI crop | — | — | — |
| 80% ROI crop | — | — | — |

**Decision:** `PENDING`.

## UAV-A04 — Zoom and resolution degradation

**Source and rationale:** Fan (2026) uses zoom augmentation, and operational studies identify spatial-resolution variation as a primary challenge.

Variants:

- none;
- zoom 0.95–1.05;
- zoom 0.90–1.10;
- zoom 0.80–1.20;
- realistic downsample–upsample augmentation;
- best zoom plus resolution degradation.

**Image:** `docs/assets/uav-preprocessing/UAV-A04_visual-comparison.png`.

| Variant | Macro F1 | Performance by GSD bin | Small-building F1 |
|---|---:|---|---:|
| None | — | — | — |
| 0.95–1.05 | — | — | — |
| 0.90–1.10 | — | — | — |
| 0.80–1.20 | — | — | — |
| Resolution degradation | — | — | — |
| Combined | — | — | — |

**Decision:** `PENDING`.

## UAV-A05 — Synthetic shadows

**Source and rationale:** Fan (2026) motivates illumination and shadow robustness. Moderate synthetic shadows may improve robustness when roofs or debris are partly obscured; large opaque shadows may remove target signal.

Variants:

- none;
- soft shadow over 10–25% of crop;
- soft/moderate shadow over 25–50%;
- direction and opacity constrained to observed UAV conditions.

**Image:** `docs/assets/uav-preprocessing/UAV-A05_visual-comparison.png`.

| Variant | Macro F1 | Naturally shadowed-crop performance | Non-shadowed performance |
|---|---:|---:|---:|
| None | — | — | — |
| 10–25% shadow | — | — | — |
| 25–50% shadow | — | — | — |
| Observed-condition shadow | — | — | — |

**Decision:** `PENDING`.

# Additional modality

## UAV-M01 — DSM height and slope channels

**Hypothesis:** local height and slope may expose roof collapse or major structural deformation unavailable from RGB alone.

Because DSM exists for only part of the UAV data, use a separately documented matched subset.

Variants:

- RGB-only matched-subset baseline;
- RGB plus normalized height;
- RGB plus slope;
- RGB plus height and slope;
- separate RGB and DSM encoders with feature fusion.

**Image:** `docs/assets/uav-preprocessing/UAV-M01_visual-comparison.png`.

| Variant | Matched-subset macro F1 | Major/destroyed F1 | Coverage limitation |
|---|---:|---:|---|
| RGB | — | — | — |
| RGB + height | — | — | — |
| RGB + slope | — | — | — |
| RGB + height + slope | — | — | — |
| Two encoders | — | — | — |

**Decision:** `PENDING`.

# Experiment run log

Add one row for each executed run or grouped experiment and link the configuration/result artifact when available.

| Date | Run ID | Step | Dataset subset | Model/config | Seeds | Result artifact | Outcome |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |

# Final frozen UAV pipeline

Complete only after all priority-1 candidates have been evaluated.

| Stage | Selected configuration | Supporting experiment | Decision date |
|---|---|---|---|
| Invalid-pixel handling | Pending | UAV-P01 | — |
| Building ROI/context | Pending | UAV-P04 | — |
| Polygon/mask handling | Pending | UAV-P05 | — |
| Input size/interpolation | Pending | UAV-P06 | — |
| GSD/scale handling | Pending | UAV-P07 | — |
| RGB normalization | Pending | UAV-P08 | — |
| Geometric augmentation | Pending | UAV-A01 | — |
| Photometric augmentation | Pending | UAV-A02 | — |
| Translation/crop augmentation | Pending | UAV-A03 | — |
| Zoom/resolution augmentation | Pending | UAV-A04 | — |
| Shadow augmentation | Pending | UAV-A05 | — |
| DSM usage | Pending | UAV-M01 | — |

# References

1. Hasan, F., Yeum, C. M., Lesani, A., and Costa, R. (2026). [Graph-Attention Network for Spatially-Aware Post-Hurricane Building Damage Assessment from UAV Imagery](https://doi.org/10.5194/isprs-annals-XI-3-2026-101-2026).
2. Manzini, T., Perali, P., Karnik, R., and Murphy, R. (2024). [CRASAR-U-DROIDs: A Large Scale Benchmark Dataset for Building Alignment and Damage Assessment in Georectified sUAS Imagery](https://arxiv.org/abs/2407.17673).
3. Manzini, T., Perali, P., Murphy, R. R., and Merrick, D. (2025). [Challenges and Research Directions from the Operational Use of a Machine Learning Damage Assessment System via Small Uncrewed Aerial Systems at Hurricanes Debby and Helene](https://arxiv.org/abs/2506.15890).
4. Manzini, T., Perali, P., and Murphy, R. R. (2026). [Deploying Rapid Damage Assessments from sUAS Imagery for Disaster Response](https://doi.org/10.1609/aaai.v40i47.41474).
5. Corley, I., Robinson, C., Dodhia, R., Lavista Ferres, J. M., and Najafirad, P. (2024). [Revisiting Pre-trained Remote Sensing Model Benchmarks: Resizing and Normalization Matters](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/html/Corley_Revisiting_Pre-trained_Remote_Sensing_Model_Benchmarks_Resizing_and_Normalization_Matters_CVPRW_2024_paper.html).
6. Fan, C.-L. (2026). [High-Resolution UAV Image Classification of Land Use and Land Cover Based on CNN Architecture Optimization](https://doi.org/10.32604/cmc.2026.077260).
7. Scheele, S., Picchione, K., and Liu, J. (2025). [LADI v2: Multi-label Dataset and Classifiers for Low-Altitude Disaster Imagery](https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/html/Scheele_LADI_v2_Multi-label_Dataset_and_Classifiers_for_Low-Altitude_Disaster_Imagery_CVPRW_2025_paper.html).

# Update checklist for every completed step

- [ ] Change status to `Complete` in the candidate summary.
- [ ] Add exact code/configuration used.
- [ ] Add dataset subset and manifest identifier.
- [ ] Add seed count.
- [ ] Fill the results table.
- [ ] Add at least one representative visual comparison.
- [ ] Add metric plots or confusion matrices where useful.
- [ ] Document relevant failure cases.
- [ ] Record computational cost if materially changed.
- [ ] Set the decision to `KEEP`, `REJECT`, or `NEEDS FULL DATA`.
- [ ] Explain the final decision in one concise paragraph.
- [ ] Add the run to the experiment log.
- [ ] If retained, update the final frozen UAV pipeline table.
