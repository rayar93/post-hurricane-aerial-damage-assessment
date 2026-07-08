# Segmentation Dataset Initial Run Summary

## Purpose

The purpose of this step was to create the first model-ready segmentation dataset from a class-diverse CRASAR hurricane-related GeoTIFF and its corresponding building damage assessment annotation JSON.

The selected candidate was:

```text
train/imagery/SATELLITE/1002-Ft-Myers-Beach-TFD.geo.tif_10300100DB06A700-visual.tif.geo.tif

This file was selected because it contains multiple useful damage classes.

Pipeline order

The pipeline follows this order:

1. Read GeoTIFF.
2. Convert GeoTIFF to RGB.
3. Read building_damage_assessment annotation JSON.
4. Remove unusable labels.
5. Create stratified and grouped train/validation/test split.
6. Generate segmentation masks from building polygon pixels.
7. Extract image/mask crops.
8. Apply deterministic preprocessing.
9. Apply augmentation only to train.
10. Save model-ready dataset.
Split strategy

The dataset uses:

Split	Ratio
Train	70%
Validation	15%
Test	15%

The split is stratified by damage label and grouped by building_id to reduce leakage risk.

Augmentation is applied only to the training split.

Labels

The first clean setup uses:

Class ID	Label
0	background
1	no damage
2	minor damage
3	major damage
4	destroyed
255	ignore

The labels un-classified and obscured are excluded from this first clean setup.

GeoTIFF metadata

The selected GeoTIFF has:

Field	Value
CRS	EPSG:32617
Width	7,180
Height	5,732
Bands	3
Resolution X	0.30517578125
Resolution Y	0.30517578125
Dataset output

The generated dataset is saved locally in:

data/segmentation_dataset/ian_tfd_satellite_10300100DB06A700/

The output structure is:

train/images
train/masks
val/images
val/masks
test/images
test/masks
metadata/manifest.csv
metadata/split_summary.json
metadata/label_mapping.json
metadata/validation_summary.json
Initial dataset size

Original saved samples:

Split	Samples
Train	534
Validation	115
Test	114

After training-only augmentation:

Split	Samples
Train	1,602
Validation	115
Test	114
Original label distribution by split
Train
Label	Count
no damage	231
minor damage	172
major damage	28
destroyed	103
Validation
Label	Count
no damage	50
minor damage	37
major damage	6
destroyed	22
Test
Label	Count
no damage	49
minor damage	37
major damage	6
destroyed	22
Validation result

The generated dataset was validated using:

scripts/validate_segmentation_dataset.py

Validation results:

Check	Result
Total manifest rows	1,831
Errors	0
Warnings	0
Mask values seen	0, 1, 2, 3, 4
Train-val building_id overlap	0
Train-test building_id overlap	0
Val-test building_id overlap	0
Augmented samples outside train	0
Interpretation

The dataset is ready for baseline segmentation training.

The split is approximately 70/15/15 and preserves the label distribution across train, validation, and test.

Training augmentation is applied only to train. Validation and test remain unaugmented. This is important to avoid data leakage.

Current status

This is the first model-ready segmentation dataset produced by the preprocessing pipeline.

Next step

The next step is to run a baseline segmentation model using this prepared dataset.

Suggested baseline:

Architecture: U-Net
Input size: 256 x 256
Initial loss: cross entropy
Alternative losses: weighted cross entropy, focal loss
Metrics: mean IoU, per-class IoU, foreground mean IoU, pixel accuracy
