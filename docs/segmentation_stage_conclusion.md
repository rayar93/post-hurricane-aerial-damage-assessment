# Segmentation Pipeline Stage Conclusion

## Current stage

This stage moved the project from GeoTIFF inspection and crop extraction into a first complete segmentation training pipeline.

The goal was to prepare CRASAR hurricane-related GeoTIFF data for segmentation model training while avoiding data leakage.

## Dataset used

The first segmentation dataset was created from the Hurricane Ian SATELLITE GeoTIFF:

```text
train/imagery/SATELLITE/1002-Ft-Myers-Beach-TFD.geo.tif_10300100DB06A700-visual.tif.geo.tif

The corresponding annotation file was:

train/annotations/SATELLITE/building_damage_assessment/1002-Ft-Myers-Beach-TFD.geo.tif_10300100DB06A700-visual.tif.geo.tif.json

This file was selected because it had multiple useful damage classes:

no damage
minor damage
major damage
destroyed
Pipeline created

The pipeline now supports:

GeoTIFF RGB conversion
building polygon mask generation
image/mask crop extraction
stratified grouped train/validation/test split
train-only augmentation
dataset validation
baseline U-Net training
loss function comparison
balanced sampling experiment
Dataset split

The dataset was split using a 70/15/15 split:

Split	Original samples	With augmentation
Train	534	1,602
Validation	115	115
Test	114	114

Augmentation was applied only to the training split.

Label distribution

Original label distribution by split:

Split	no damage	minor damage	major damage	destroyed
Train	231	172	28	103
Validation	50	37	6	22
Test	49	37	6	22
Dataset validation

The dataset validation script confirmed:

Check	Result
Errors	0
Warnings	0
Valid mask values	0, 1, 2, 3, 4
Train-val building_id overlap	0
Train-test building_id overlap	0
Val-test building_id overlap	0

This confirms that the generated dataset is structurally valid and leakage-safe at the building_id level.

Models trained

A baseline U-Net was trained under several settings:

Run	Pixel accuracy	Mean IoU all	Foreground mean IoU	no damage IoU	minor damage IoU	major damage IoU	destroyed IoU
Cross entropy, 1 epoch	0.8326	0.2539	0.0895	0.3561	0.0019	0.0000	0.0000
Weighted cross entropy, 5 epochs	0.8417	0.3523	0.2133	0.3225	0.1105	0.0000	0.4200
Focal loss, 5 epochs	0.4734	0.1988	0.1350	0.2949	0.1424	0.0009	0.1018
Weighted cross entropy, 15 epochs	0.8580	0.4171	0.2922	0.3059	0.2875	0.0000	0.5752
Weighted cross entropy + balanced sampler, 15 epochs	0.8234	0.3845	0.2516	0.1110	0.2449	0.0895	0.5610
Main result

The best overall run so far is:

Weighted cross entropy, 15 epochs

It achieved:

Metric	Value
Pixel accuracy	0.8580
Mean IoU all	0.4171
Foreground mean IoU	0.2922
Background IoU	0.9169
no damage IoU	0.3059
minor damage IoU	0.2875
major damage IoU	0.0000
destroyed IoU	0.5752
Balanced sampler result

The balanced sampler helped the model learn some major damage signal:

major damage IoU: 0.0000 → 0.0895

However, it reduced the overall foreground mean IoU:

foreground mean IoU: 0.2922 → 0.2516

and strongly reduced no damage IoU:

no damage IoU: 0.3059 → 0.1110
Interpretation

The preprocessing and training pipeline works.

The first cross entropy smoke test mostly collapsed to background/no damage, which was expected because of class imbalance and the short training run.

Weighted cross entropy improved the foreground segmentation substantially.

The remaining bottleneck is the major damage class. This class has very few examples:

Train original major damage samples: 28
Validation major damage samples: 6
Test major damage samples: 6

The balanced sampler shows that major damage is learnable, but the current single-file dataset is too limited for stable performance on that class.

Current conclusion

This stage is complete.

The project now has:

1. A leakage-safe segmentation preprocessing pipeline.
2. A validated image/mask dataset.
3. A baseline U-Net training script.
4. Loss function comparison results.
5. Evidence that class imbalance is the next modelling bottleneck.
Recommended next step

The next step should be decided with the professor.

The most likely next direction is to expand from a single-file dataset to a multi-file dataset with more examples of major damage.

The next technical phase would be:

1. Select additional class-diverse GeoTIFF/annotation pairs.
2. Build a multi-file manifest.
3. Create a leakage-safe train/validation/test split across files.
4. Generate a larger segmentation dataset.
5. Retrain weighted cross entropy and balanced-sampling baselines.

The priority should be more class-diverse data before heavier model tuning.
