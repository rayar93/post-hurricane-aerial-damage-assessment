# Segmentation Preprocessing Pipeline Plan

## Goal

The goal of this phase is to prepare CRASAR hurricane-related GeoTIFF data for segmentation model training.

The pipeline should convert raw georeferenced GeoTIFF imagery and building annotation JSON files into clean RGB image crops and segmentation masks.

## Correct order

The correct order is:

```text
1. Build original data manifest.
2. Remove unusable labels.
3. Create stratified and leakage-safe train/validation/test split.
4. Convert GeoTIFF to RGB.
5. Generate segmentation masks from building polygons.
6. Extract image/mask crops.
7. Apply deterministic preprocessing.
8. Apply data augmentation only to training data.
9. Save final model-ready dataset.
Split strategy

The preferred split is:

train: 70%
validation: 15%
test: 15%

The split must be stratified by damage label and grouped to avoid data leakage.

The grouping variables should include:

building_id
original GeoTIFF file
source
hurricane/site when available

No augmented version of an image should ever appear in a different split from the original.

Labels

The first clean segmentation setup should use:

Class ID	Label
0	background
1	no damage
2	minor damage
3	major damage
4	destroyed
255	ignore

The labels un-classified and obscured should be excluded or mapped to ignore for the first clean setup.

GeoTIFF preprocessing

For each selected GeoTIFF:

1. Read bands using rasterio.
2. Convert bands 1-3 to RGB.
3. Preserve CRS, bounds, resolution, and transform in metadata.
4. Use annotation JSON `pixels` field to generate polygon masks.
5. Extract image crops and matching mask crops.
Deterministic preprocessing

Apply to train, validation, and test:

RGB conversion
mask generation
crop extraction
black/no-data filtering
resize image and mask
contrast normalization
normalization to model input range

Masks must be resized with nearest-neighbor interpolation.

Images can be resized with bilinear interpolation.

Training-only augmentation

Apply only to train:

horizontal flip
vertical flip
90-degree rotations
brightness/contrast jitter
mild denoising

Do not apply augmentation before splitting.

Notes on object detection

Object detection is not required in the first segmentation pipeline because CRASAR already provides building polygon annotations. These polygons can be used directly to generate segmentation masks.

Notes on deblurring and rectification

Deblurring should not be applied aggressively by default because it can introduce artifacts.

GeoTIFF orthomosaics are already georeferenced and usually orthorectified. Additional distortion correction should only be used if a clear distortion problem is observed.

Model-ready output

The final dataset should be saved as:

data/segmentation_dataset/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
  metadata/
    manifest.csv
    split_summary.json
    label_mapping.json
Future modelling

Initial loss functions to compare:

cross entropy
weighted cross entropy
focal loss
dice loss / combined cross entropy + dice

Focal loss may be useful if the dataset is class-imbalanced.
