# Post-Hurricane Aerial Damage Assessment

An open-source research project for building-level post-hurricane damage assessment from satellite and UAV imagery.

The project is designed around three requirements: evidence-based preprocessing, leakage-safe evaluation, and generalization to disaster events that were not used for model development.

## Research documentation

- [UAV preprocessing experiment registry](UAV_PREPROCESSING_EXPERIMENTS.md): candidate preprocessing steps, literature rationale, experiment results, visual evidence, and final keep/reject decisions.

## Completion target — August 30, 2026

By **August 30, 2026**, the repository should contain a complete, reproducible pipeline covering:

- data inventory and quality validation;
- finalized satellite and UAV preprocessing;
- duplicate and near-duplicate control;
- building detection or segmentation;
- leakage-safe multi-event dataset construction;
- class-imbalance handling;
- baseline model comparison;
- final architecture selection and tuning;
- definitive model training;
- evaluation on an unseen disaster event;
- inference, documentation, and reproducibility assets.

## Target end-to-end pipeline

```text
Raw satellite and UAV imagery
        ↓
Data integrity and annotation audit
        ↓
Valid-pixel masking and quality filtering
        ↓
Duplicate and near-duplicate grouping
        ↓
Building detection or segmentation
        ↓
Building-level image and mask generation
        ↓
Grouped, stratified, multi-event dataset split
        ↓
Train-only augmentation and class balancing
        ↓
Baseline comparison and architecture selection
        ↓
Definitive model training
        ↓
Unseen-event evaluation and inference package
```

## August 2026 roadmap

| Dates | Milestone | Required work | Exit criterion |
|---|---|---|---|
| **July 30 – August 2** | Freeze the research and evaluation protocol | Inventory all satellite, UAV, DSM, and annotation files. Match imagery with annotations. Define the prediction task, label taxonomy, exclusion rules, target image resolution, evaluation metrics, random seeds, and the disaster events reserved for validation and final testing. | A version-controlled experiment protocol and data manifest define exactly what will be trained, validated, and tested. |
| **August 3 – August 6** | Complete the data-quality audit | Quantify missing data, invalid annotations, obscured or unclassified labels, blank regions, alpha-channel coverage, corrupted images, crop dimensions, class frequencies, event frequencies, geographic coverage, and resolution differences. Produce representative visual examples for every relevant problem. | A data-quality report identifies every exclusion, transformation, and unresolved risk with counts and percentages. |
| **August 7 – August 9** | Finalize and freeze preprocessing | Evaluate candidate preprocessing steps on stratified 10%, 20%, and 30% subsets. Test valid-pixel masking, resizing and interpolation, normalization, crop margins, contrast transformations, denoising, and any modality-specific corrections. Retain a step only when it has a clear hypothesis and produces consistent visual or quantitative benefit. | One deterministic preprocessing configuration is selected for satellite imagery and one for UAV imagery. No preprocessing decision remains open after August 9. |
| **August 10 – August 12** | Finalize leakage prevention and dataset splitting | Detect exact and near duplicates. Group temporally adjacent UAV frames, repeated views, and samples sharing the same building identity. Define grouped and stratified train, validation, and test splits. Reserve at least one complete disaster event for final out-of-distribution evaluation. Ensure augmentation is restricted to the training split. | Automated validation reports zero prohibited overlap across splits and document duplicate clusters, building groups, event groups, and class distributions. |
| **August 13 – August 16** | Complete building isolation and generate the final dataset | Compare the viable building-isolation approaches, including annotation-derived masks, semantic segmentation, instance segmentation, or object detection. Select the approach that provides the best combination of building coverage, boundary quality, robustness, and computational cost. Generate the complete multi-file, multi-event dataset and all manifests. | The final training dataset is generated, validated, class-distribution checked, and reproducible from raw data through a documented command sequence. |
| **August 17 – August 20** | Train and compare baseline models | Train the existing U-Net baseline and reproduce the selected published baselines, including DorianNet where applicable and additional competitive architectures. Use the same frozen splits, preprocessing, metrics, and stopping rules for every model. Record parameter counts, training time, inference time, and per-class results. | A baseline comparison table identifies the strongest architectures and the main remaining error modes. |
| **August 21 – August 24** | Select and tune the definitive architecture | Shortlist the two strongest models. Compare class-weighted losses, focal loss, Dice-based losses, balanced sampling, targeted augmentation, learning rates, schedulers, image sizes, pretrained encoders, and regularization. Run controlled ablations and multiple seeds for the most important configurations. Freeze the architecture and all hyperparameters by August 24. | A signed-off final configuration is selected using validation results only. The final test event remains untouched. |
| **August 25 – August 27** | Train the definitive model | Train the frozen final configuration using the complete development dataset. Save checkpoints, training curves, configuration files, random seeds, environment details, and model-selection metadata. Run repeated training where needed to confirm stability. Export the best checkpoint and inference-ready weights. | The definitive model is fully trained, reproducible, and ready for final evaluation without additional tuning. |
| **August 28** | Run final evaluation | Evaluate the frozen model once on the reserved unseen disaster event. Report overall and per-class performance, confusion matrices, segmentation metrics where applicable, calibration, failure cases, and performance by modality, event, resolution, and damage severity. Compare against every selected baseline. | A final results package demonstrates both in-distribution performance and cross-disaster generalization. |
| **August 29** | Complete reproducibility and inference assets | Create a single-image or folder-level inference entry point. Verify the pipeline from a clean environment. Finalize installation instructions, dataset preparation commands, training commands, evaluation commands, model-card information, limitations, and result tables. | Another researcher can reproduce preprocessing, training, evaluation, and inference from the repository documentation. |
| **August 30** | Project freeze and final release | Run the full validation checklist, resolve documentation inconsistencies, verify all referenced paths and commands, archive final configurations and reports, tag the release, and freeze the August research pipeline. | Preprocessing, final dataset construction, definitive model training, unseen-event evaluation, and reproducibility documentation are complete. |

## Evaluation requirements

The final comparison must use metrics that expose class imbalance and generalization rather than relying on overall accuracy alone.

For damage classification, report:

- macro F1 and weighted F1;
- balanced accuracy;
- per-class precision, recall, and F1;
- confusion matrix;
- calibration metrics where practical;
- results by disaster event and imagery modality.

For building or damage segmentation, additionally report:

- mean IoU;
- foreground mean IoU;
- per-class IoU;
- Dice score;
- building coverage and missed-building rate.

The final test event must not influence preprocessing selection, model selection, hyperparameter tuning, early stopping, or threshold selection.

## Definition of done

The August milestone is complete only when all of the following are true:

- [ ] Raw imagery can be converted into a validated model-ready dataset through documented commands.
- [ ] Every retained preprocessing step has visual and quantitative justification.
- [ ] Duplicate, building-level, and event-level leakage checks pass automatically.
- [ ] Validation and test data contain no training augmentations.
- [ ] The final dataset contains multiple files and disaster events with documented class distributions.
- [ ] Published and internal baselines are evaluated under the same protocol.
- [ ] The definitive architecture and hyperparameters are frozen before final testing.
- [ ] The definitive model is trained and its checkpoint is saved with full configuration metadata.
- [ ] The final model is evaluated on at least one completely unseen disaster event.
- [ ] Per-class results and representative failure cases are documented.
- [ ] Inference can be run from a documented command or entry point.
- [ ] A clean environment can reproduce the principal preprocessing, training, and evaluation outputs.

## Research rules

1. No preprocessing step is included without a dataset-specific hypothesis and supporting evidence.
2. Train, validation, and test splits are grouped to prevent the same building, duplicate image, or near-identical UAV sequence from crossing split boundaries.
3. Augmentation is applied only to training data.
4. Model selection is based on validation data, never on the final test event.
5. The final test evaluation is performed only after preprocessing, architecture, and hyperparameters are frozen.
6. All reported models use the same data split and evaluation protocol unless an exception is explicitly documented.

## Data policy

Large videos, raw datasets, extracted frames, generated datasets, model checkpoints, and other heavy artifacts should not be committed to GitHub.

Only source code, configuration files, documentation, manifests without restricted data, and small result reports should be version-controlled.



