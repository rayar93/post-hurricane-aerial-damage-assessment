# U-Net Baseline Smoke Test Summary

## Purpose

The purpose of this step was to verify that the generated segmentation dataset can be used for model training.

This was a smoke test, not a final model training run.

## Dataset

The dataset used was:

```text
data/segmentation_dataset/ian_tfd_satellite_10300100DB06A700/

This dataset was generated from the Hurricane Ian SATELLITE GeoTIFF:

train/imagery/SATELLITE/1002-Ft-Myers-Beach-TFD.geo.tif_10300100DB06A700-visual.tif.geo.tif

The dataset had already passed validation:

Check	Result
Errors	0
Warnings	0
Train-val building_id overlap	0
Train-test building_id overlap	0
Val-test building_id overlap	0
Valid mask values	0, 1, 2, 3, 4
Model

The smoke test used a small baseline U-Net:

Parameter	Value
Architecture	U-Net
Base channels	16
Input size	256 x 256
Batch size	4
Epochs	1
Loss	Cross entropy
Optimizer	AdamW
Learning rate	0.001
Device	MPS
Dataset size used by the model
Split	Samples
Train	1,602
Validation	115
Test	114

The training split includes augmentation. Validation and test are not augmented.

Smoke test result

Validation after epoch 1:

Metric	Value
Train loss	0.6971
Validation loss	0.4268
Validation foreground mean IoU	0.0900
Validation pixel accuracy	0.8363

Final test metrics:

Metric	Value
Test loss	0.4382
Pixel accuracy	0.8326
Mean IoU, all classes	0.2539
Mean IoU, foreground classes	0.0895
IoU background	0.9115
IoU no damage	0.3561
IoU minor damage	0.0019
IoU major damage	0.0000
IoU destroyed	0.0000
Interpretation

The smoke test succeeded.

The model can ingest the generated image/mask pairs, train for one epoch, evaluate on validation and test, and save model checkpoints and metrics.

The metrics should not be interpreted as final model performance because this was only a one-epoch smoke test.

The high background IoU and low foreground mean IoU suggest that ordinary cross entropy is initially dominated by background and common classes. This is expected in segmentation datasets where the building mask occupies a smaller fraction of each crop and where class imbalance exists.

Next steps

The next modelling steps are:

1. Visualize predictions from the smoke-test model.
2. Train a longer cross-entropy baseline.
3. Train with weighted cross entropy.
4. Train with focal loss.
5. Compare mean IoU, foreground mean IoU, and per-class IoU.

Weighted cross entropy and focal loss are especially relevant because the first smoke test did not learn minority classes well after one epoch.
