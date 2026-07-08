# Initial Loss Function Comparison for Segmentation

## Purpose

The purpose of this experiment was to compare early segmentation training behavior using:

```text
cross entropy
weighted cross entropy
focal loss

The goal was not to obtain final performance, but to verify whether imbalance-aware losses improve foreground and minority-class segmentation.

Dataset

Dataset used:

data/segmentation_dataset/ian_tfd_satellite_10300100DB06A700/

This dataset contains a 70/15/15 split with no building_id leakage across train, validation, and test.

Original split:

Split	Samples
Train	534
Validation	115
Test	114

Train with augmentation:

Split	Samples
Train	1,602
Validation	115
Test	114

Original label counts:

Split	no damage	minor damage	major damage	destroyed
Train	231	172	28	103
Validation	50	37	6	22
Test	49	37	6	22
Compared runs
Cross entropy smoke test
epochs: 1
base channels: 16
loss: cross entropy
Weighted cross entropy
epochs: 5
base channels: 16
loss: weighted cross entropy
Focal loss
epochs: 5
base channels: 16
loss: focal loss
Test metrics
Metric	CE 1 epoch	Weighted CE 5 epochs	Focal 5 epochs
Loss	0.4382	0.7073	0.1432
Pixel accuracy	0.8326	0.8417	0.4734
Mean IoU, all classes	0.2539	0.3523	0.1988
Mean IoU, foreground	0.0895	0.2133	0.1350
IoU background	0.9115	0.9084	0.4540
IoU no damage	0.3561	0.3225	0.2949
IoU minor damage	0.0019	0.1105	0.1424
IoU major damage	0.0000	0.0000	0.0009
IoU destroyed	0.0000	0.4200	0.1018
Interpretation

Weighted cross entropy is the best early result so far.

It improved foreground mean IoU from 0.0895 to 0.2133 and substantially improved destroyed IoU from 0.0000 to 0.4200.

Focal loss improved minor damage relative to the one-epoch cross-entropy smoke test, but it substantially reduced background IoU and overall pixel accuracy in this early run.

The main remaining weakness is major damage, which still has near-zero IoU. This is likely related to class scarcity:

Train original major damage samples: 28
Validation major damage samples: 6
Test major damage samples: 6
Current conclusion

The preprocessing and training pipeline works.

The next modelling problem is class imbalance, especially for major damage.

The next experiments should be:

1. Train weighted cross entropy for more epochs.
2. Try balanced sampling or class-balanced batches.
3. Add more class-diverse GeoTIFF files to increase major damage examples.
4. Compare models using foreground mean IoU and per-class IoU, not only pixel accuracy.
