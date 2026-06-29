# Building-Level pHash and Representative Crop Selection

## Motivation

The first pHash experiment showed very high temporal redundancy in consecutive UAV frames. However, the model's target is building-level damage classification, not full-frame classification. Therefore, the pHash approach should move from full-frame duplicate detection to building-level crop selection.

The main question becomes:

```text
For each building, which crop or view should be kept as the most informative representative image?

Why CRASAR-U-DROIDs is useful

The initial CRASAR-U-DROIDs inspection showed that the dataset includes the following building-level fields:

building_id
view_id
label
pixels
boundary
source
filename

This means that CRASAR supports building-level and view-level redundancy analysis. Instead of comparing complete frames, we can compare crops or polygons belonging to the same building.

Proposed workflow

The proposed building-level workflow is:

Use building polygon or pixel annotations to extract building crops.
Group crops by building_id.
Compute pHash for each building crop.
Compare crops/views belonging to the same building.
Estimate redundancy using Hamming distances.
Compute image-quality metrics such as sharpness, contrast, brightness, and crop size.
Select either:
one best crop per building, or
a small diverse set of high-quality crops per building.
Selection criteria

A good representative crop should ideally have:

High sharpness.
Sufficient contrast.
Good brightness balance.
Large enough building area.
Minimal occlusion.
Low redundancy relative to other selected crops.
A consistent damage label.
Methodological value

Building-level pHash can support three goals:

Detect redundant views of the same building.
Select clearer and more informative building crops.
Reduce leakage risk by ensuring that train, validation, and test splits are grouped by building_id.
Open questions for discussion
Should we keep one best crop per building or multiple diverse views?
Should pHash be used for hard duplicate removal or only for representative selection?
Should CRASAR-U-DROIDs be the main dataset and DoriaNET the Dorian UAV case study?
How should Alan's label-harmonization work be connected with the crop-selection pipeline?
How should un-classified and obscured labels be handled?
