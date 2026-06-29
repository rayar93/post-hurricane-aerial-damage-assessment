# CRASAR-U-DROIDs Cross-Source Redundancy Summary

## Purpose

The purpose of this analysis was to determine whether CRASAR-U-DROIDs contains repeated building-level annotations that could affect model training, evaluation, label consistency, or representative crop selection.

The analysis compared building-level annotation records using `building_id`, `view_id`, `filename`, and `label`.

## Main finding

CRASAR-U-DROIDs does not appear to have the same type of temporal frame redundancy observed in UAV video. Instead, the redundancy pattern depends strongly on imagery source.

The UAS annotations appear to contain one annotation per `building_id`, while satellite and crewed aerial imagery contain many repeated `building_id` entries across multiple filenames, often with inconsistent labels.

## UAS analysis

All available UAS building damage assessment JSON files were analyzed.

| Metric                                     |  Value |
| ------------------------------------------ | -----: |
| UAS JSON files found                       |     52 |
| UAS JSON files loaded                      |     52 |
| Annotation records                         | 21,642 |
| Unique buildings                           | 21,642 |
| Repeated building IDs                      |      0 |
| Repeated buildings with multiple views     |      0 |
| Repeated buildings with multiple labels    |      0 |
| Repeated buildings with multiple filenames |      0 |

### UAS label distribution

| Label         |  Count |
| ------------- | -----: |
| no damage     | 11,012 |
| minor damage  |  6,063 |
| major damage  |  2,349 |
| destroyed     |  1,361 |
| un-classified |    838 |
| obscured      |     19 |

### UAS interpretation

The UAS subset does not show repeated `building_id` entries. This suggests that, within CRASAR-U-DROIDs UAS annotations, each building is represented once in the current metadata structure.

Therefore, for CRASAR-U-DROIDs UAS, pHash may not be needed to remove repeated views of the same `building_id`. Instead, pHash may be more useful for auditing visual similarity between nearby crops or for selecting representative crops if additional candidate views are generated.

## Satellite analysis

A sample of 100 satellite building damage assessment JSON files was analyzed out of 174 available files.

| Metric                                     |  Value |
| ------------------------------------------ | -----: |
| Satellite JSON files found                 |    174 |
| Satellite JSON files loaded                |    100 |
| Annotation records                         | 61,794 |
| Unique buildings                           | 14,268 |
| Repeated building IDs                      | 14,249 |
| Repeated buildings with multiple views     |      0 |
| Repeated buildings with multiple labels    |  5,086 |
| Repeated buildings with multiple filenames | 14,249 |

### Satellite label distribution

| Label         |  Count |
| ------------- | -----: |
| no damage     | 49,337 |
| minor damage  |  5,506 |
| obscured      |  1,803 |
| un-classified |  1,929 |
| major damage  |  1,664 |
| destroyed     |  1,555 |

### Satellite interpretation

The satellite subset shows extensive repeated `building_id` entries across multiple filenames. However, these repeated buildings do not appear to have multiple `view_id` values in the inspected sample.

A major concern is that 5,086 repeated buildings had multiple labels. This suggests that some buildings may have inconsistent annotations across different satellite image files, dates, products, or contexts. This is important for model training because the same building should not appear across train, validation, and test splits with conflicting or repeated information.

## Crewed aerial analysis

All available crewed aerial building damage assessment JSON files were analyzed.

| Metric                                     |  Value |
| ------------------------------------------ | -----: |
| CREWED JSON files found                    |     59 |
| CREWED JSON files loaded                   |     59 |
| Annotation records                         | 25,690 |
| Unique buildings                           | 18,505 |
| Repeated building IDs                      |  4,402 |
| Repeated buildings with multiple views     |      6 |
| Repeated buildings with multiple labels    |    726 |
| Repeated buildings with multiple filenames |  4,396 |

### Crewed aerial label distribution

| Label         |  Count |
| ------------- | -----: |
| no damage     | 12,736 |
| minor damage  |  6,776 |
| major damage  |  3,416 |
| destroyed     |  1,405 |
| un-classified |  1,240 |
| obscured      |    117 |

### Crewed aerial interpretation

The crewed aerial subset also contains repeated `building_id` entries across multiple filenames. A smaller number of repeated buildings had multiple `view_id` values, and 726 repeated buildings had multiple labels.

This suggests that crewed aerial data may contain repeated building annotations across different image products, and some of these repeated annotations may be label-inconsistent.

## All-source sample analysis

A sample of 100 building damage assessment JSON files was analyzed across all available imagery sources.

| Metric                                     |  Value |
| ------------------------------------------ | -----: |
| All-source JSON files found                |    285 |
| All-source JSON files loaded               |    100 |
| Annotation records                         | 45,332 |
| Unique buildings                           | 20,677 |
| Repeated building IDs                      |  8,496 |
| Repeated buildings with multiple views     |      6 |
| Repeated buildings with multiple labels    |  3,842 |
| Repeated buildings with multiple filenames |  8,490 |

### All-source label distribution

| Label         |  Count |
| ------------- | -----: |
| no damage     | 24,988 |
| minor damage  |  9,155 |
| major damage  |  4,921 |
| destroyed     |  2,720 |
| un-classified |  1,905 |
| obscured      |  1,643 |

### All-source interpretation

The all-source sample confirms that repeated `building_id` entries are common when multiple CRASAR-U-DROIDs annotation files are considered together.

Most repeated buildings appear across multiple filenames rather than multiple `view_id` values. This suggests that the main redundancy issue in CRASAR-U-DROIDs is not video-frame-like temporal duplication, but repeated building annotations across different files, image products, or sources.

The presence of repeated buildings with multiple labels is especially important. These cases should be reviewed or handled carefully before training because they may create label ambiguity or leakage if the same building appears in different dataset splits.

## Implications for pHash

For DoriaNET-style UAV video, pHash is useful for detecting temporal redundancy between consecutive frames.

For CRASAR-U-DROIDs, pHash should be reframed as a building-level or crop-level tool:

1. Extract building crops using the `pixels` or polygon annotations.
2. Group records by `building_id`.
3. Compare crops across filenames or sources using pHash.
4. Identify visually redundant building crops.
5. Check whether repeated building IDs have consistent or conflicting labels.
6. Select one representative crop or a small set of diverse high-quality crops per building.
7. Use `building_id` as the grouping unit for leakage-resistant train/validation/test splits.

## Methodological implications

The current evidence suggests three different cases:

1. **DoriaNET UAV video:** high full-frame temporal redundancy, where pHash can detect near-duplicate consecutive frames.
2. **CRASAR-U-DROIDs UAS:** no repeated `building_id` entries in the inspected full UAS subset, so pHash may be less relevant for duplicate removal but still useful for crop-quality auditing.
3. **CRASAR-U-DROIDs satellite and crewed imagery:** substantial repeated `building_id` entries across filenames, often with label inconsistencies, making grouped splitting and label-consistency checks essential.

## Open questions for discussion

1. Should train/validation/test splits be grouped strictly by `building_id` across all sources?
2. Should repeated building IDs with multiple labels be excluded, manually reviewed, or assigned a consensus label?
3. Should `un-classified` and `obscured` be removed from the main 4-class classification setup?
4. Should UAS, satellite, and crewed aerial imagery be analyzed separately or combined?
5. Should pHash be used for representative crop selection, redundancy auditing, or both?
6. Should the project keep one crop per building or a small diverse set of high-quality crops per building?

