# Experimental protocol

## Primary analysis

1. Preserve official Derm7pt train/validation/test indexes.
2. Audit files and categorical vocabularies before training.
3. Estimate class weights from training only.
4. Train concept perception model; select checkpoint using validation mean macro-F1.
5. Train independent black-box model; select checkpoint using validation AUROC.
6. Fit learned CBM baselines using training predictions/concepts.
7. Fit temperature scaling using validation logits only.
8. Build exact concept-profile reliability map from training annotations only.
9. Select metacognitive action thresholds on validation only.
10. Evaluate the frozen pipeline exactly once on the test split.
11. Preserve row-level test predictions for paired statistical analyses.

## Primary endpoints

- diagnosis: AUROC, AUPRC, sensitivity, specificity, balanced accuracy, F1;
- concepts: per-head macro-F1, accuracy, one-vs-rest AUROC where estimable;
- calibration: Brier score and ECE;
- reasoning: neural-symbolic disagreement and concept-profile reliability;
- selective prediction: risk–coverage curve and coverage at prespecified target risk;
- intervention: error-recovery rate and probability change after one uncertainty×impact-selected concept correction.

## Seeds

The publication experiment should use at least five independently trained seeds. Report each seed and the aggregate mean/SD. Statistical comparisons that operate on predictions should remain paired at the patient/case level.

## No hidden test tuning

Do not alter preprocessing, decision thresholds, reliability weights, or calibration after inspecting test results. If a post-hoc exploratory change is made, label it explicitly and confirm it on a separate dataset.


## Secondary analyses

- PCA/t-SNE and optional UMAP on frozen visual embeddings;
- concept-probability corruption at prespecified rates;
- uncertainty/reliability response to corrupted representations;
- Grad-CAM visual explanation baseline for the EfficientNet-B3 black-box model;
- frozen external validation after all Derm7pt decisions are locked.
