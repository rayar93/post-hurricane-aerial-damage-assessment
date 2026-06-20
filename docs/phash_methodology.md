# pHash duplicate-identification methodology

The UAV imagery used in this project may contain temporal redundancy because consecutive frames can show the same building from nearly identical viewpoints. This is especially likely when the drone is static or moving slowly.

This redundancy can create data leakage if near-identical images of the same structure are randomly split across training, validation, and test sets. In that case, model performance may be artificially inflated because the model is tested on images that are visually similar to images seen during training.

To address this issue, we implemented an initial perceptual-hashing pipeline. First, frames are extracted from UAV video files. Then, for each extracted frame, the script computes a perceptual hash, or pHash. Visual similarity between two frames is measured using the Hamming distance between their hash values.

In the first pass, the script compares temporally adjacent frames:

```text
frame_i vs frame_{i+1}
