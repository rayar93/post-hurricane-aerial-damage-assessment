# Initial pHash Duplicate-Detection Results

## Video segment

An initial test was run on a short UAV video segment associated with Hurricane Dorian aerial imagery.

The first 1,800 extracted frames were analyzed. Consecutive frame pairs were compared using perceptual hashing (pHash) and Hamming distance.

## Initial results

| Metric | Value |
|---|---:|
| Frames analyzed | 1,800 |
| Sequential frame pairs compared | 1,799 |
| Duplicate candidates at threshold 6 | 1,790 |
| Duplicate-candidate rate | 99.50% |
| Minimum Hamming distance | 0 |
| Median Hamming distance | 0 |
| Maximum Hamming distance | 10 |

## Interpretation

The initial pHash results suggest substantial temporal redundancy in the UAV video segment. Most consecutive frames are visually near-identical under the initial pHash threshold, which supports the need for duplicate or near-duplicate screening before model training and evaluation.

This is relevant because treating highly redundant frames as independent observations can inflate model performance, especially if visually similar frames of the same building are split across training, validation, and test sets.

These results are preliminary. The next step is to validate the threshold through visual inspection and sensitivity analysis across multiple Hamming-distance cutoffs.
