# Post-Hurricane Aerial Damage Assessment

Possible Research Project on Post-Hurricane Aerial Damage Assessment.

## Current preprocessing objective

The current preprocessing goal is to identify duplicate and near-duplicate UAV/drone frames before model training and evaluation.

This is important because continuous drone videos can contain many visually redundant frames, especially when the UAV is static or moving slowly. If near-identical frames of the same building are randomly split across training, validation, and test sets, model performance may be artificially inflated due to data leakage.

## Current pHash pipeline

The current pipeline has two steps:

```text
video file → extracted frames → pHash duplicate detection report
```

### Step 1: Extract frames from video

Script:

```text
scripts/extract_frames_from_video.py
```

Example:

```bash
python scripts/extract_frames_from_video.py \
  --video-path data/raw/videos/example_video.mp4 \
  --output-dir data/processed/frames/example_video \
  --every-n-frames 1 \
  --image-format jpg
```

This extracts frames from the input video and saves them as image files.

### Step 2: Detect duplicate or near-duplicate frames

Script:

```text
scripts/phash_duplicate_detection.py
```

Example:

```bash
python scripts/phash_duplicate_detection.py \
  --input-dir data/processed/frames/example_video \
  --output-csv outputs/phash_reports/example_video_phash_report.csv \
  --threshold 6
```

The script computes perceptual hashes for consecutive frames, calculates Hamming distances, and flags image pairs whose distance is below or equal to the selected threshold.

## Initial threshold

The initial exploratory threshold is:

```text
threshold = 6
```

This should be validated through manual inspection and sensitivity analysis.

## Expected output

The pHash script generates a CSV file with columns such as:

```text
image_1,image_2,hash_1,hash_2,hamming_distance,duplicate_candidate,threshold
```

A lower Hamming distance means the two frames are more visually similar.

## Data policy

Large video files, raw datasets, and extracted image frames should not be committed to GitHub.

Only code, documentation, and small output reports should be version-controlled.
