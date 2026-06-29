# CRASAR-U-DROIDs Initial Inspection Summary

## Purpose

The purpose of this inspection was to understand whether CRASAR-U-DROIDs can support building-level redundancy analysis and representative crop selection.

Unlike the initial UAV video experiment, CRASAR-U-DROIDs is not organized as consecutive video frames. Instead, it is organized around annotated orthomosaic imagery, building polygons, building identifiers, and view identifiers.

## File inventory

The Hugging Face dataset repository contains:

| Item | Count |
|---|---:|
| Total files listed | 956 |
| JSON files | 658 |
| GeoTIFF imagery files | 294 |
| CSV files | 1 |
| Building damage assessment JSON files | 287 |
| Building alignment adjustment JSON files | 265 |

The dataset includes multiple imagery sources:

| Source | File count |
|---|---:|
| CREWED | 165 |
| SATELLITE | 494 |
| UAS | 260 |
| UAS_DSM | 29 |

## Sample annotation inspection

A sample of building damage assessment JSON files was inspected without downloading the full imagery dataset.

| Metric | Value |
|---|---:|
| Sample annotation entries | 3,676 |
| Unique building IDs | 1,926 |
| Unique view IDs | 1,878 |
| Repeated building IDs | 1,244 |

The annotation entries consistently include the following fields:

```text
EPSG:4326
boundary
building_id
filename
id
jds_version
label
payload_version
pixels
source
view_id

| Label         | Count |
| ------------- | ----: |
| no damage     | 1,961 |
| minor damage  |   680 |
| major damage  |   327 |
| destroyed     |   468 |
| un-classified |   177 |
| obscured      |    63 |

Interpretation

This initial inspection confirms that CRASAR-U-DROIDs supports building-level analysis. The presence of building_id, view_id, pixels, boundary, and label means that redundancy should be studied at the building/view level rather than at the raw frame level.

The key methodological question is therefore not whether there are duplicate video frames, but whether the same building appears in multiple redundant or near-redundant views.

This structure is useful for the next stage of the pHash work because building crops can be extracted using the pixels or polygon information, grouped by building_id, and then compared using pHash and image-quality metrics.

Next methodological direction

The next step is to adapt the pHash pipeline from full-frame duplicate detection to building-level crop selection:

Extract or identify building-level crops using polygon or pixel annotations.
Group crops by building_id.
Compute pHash for each building crop or view.
Measure redundancy between views of the same building.
Select either one representative crop or a small diverse set of high-quality crops per building.
Use building_id to support leakage-resistant train/validation/test splits.
