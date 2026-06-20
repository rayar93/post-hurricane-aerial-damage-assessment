# Initial pHash Duplicate-Detection Results

## Video segment

An initial test was run on the first 1,800 extracted frames of a Hurricane Dorian UAV video segment.

Consecutive frame pairs were compared using perceptual hashing (pHash) and Hamming distance. The purpose of this initial test was to quantify temporal redundancy in UAV video imagery before model training and evaluation.

## Initial summary

| Metric | Value |
|---|---:|
| Frames analyzed | 1,800 |
| Sequential frame pairs compared | 1,799 |
| Duplicate candidates at threshold 6 | 1,790 |
| Duplicate-candidate rate at threshold 6 | 99.50% |
| Minimum Hamming distance | 0 |
| Median Hamming distance | 0 |
| Maximum Hamming distance | 10 |

## Threshold sensitivity analysis

| Hamming-distance threshold | Duplicate-candidate pairs | Duplicate-candidate rate |
|---:|---:|---:|
| <= 0 | 1,365 | 75.88% |
| <= 2 | 1,712 | 95.16% |
| <= 4 | 1,774 | 98.61% |
| <= 6 | 1,790 | 99.50% |
| <= 8 | 1,796 | 99.83% |
| <= 10 | 1,799 | 100.00% |

## Interpretation

The initial pHash results suggest substantial temporal redundancy in the UAV video segment. Even under a strict threshold of 0, 75.88% of consecutive frame pairs produced identical pHash values. Under the exploratory threshold of 6, 99.50% of consecutive frame pairs were flagged as duplicate or near-duplicate candidates.

These preliminary results support the need for duplicate or near-duplicate screening before model training and evaluation. Treating highly redundant consecutive frames as independent observations could inflate model performance, especially if visually similar frames of the same building are split across training, validation, and test sets.

The next step is to validate the threshold through visual inspection and then apply the pipeline across additional UAV videos or frame sequences.
