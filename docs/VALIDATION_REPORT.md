# Codebase validation report

Validation performed before packaging:

1. **Unit tests:** entropy, metacognitive reliability, concept-profile inconsistency, soft seven-point reasoning, and Grad-CAM output shape/bounds.
2. **Python compilation:** all files under `src/`, `scripts/`, and `tests/` compiled successfully.
3. **Synthetic Derm7pt audit:** official-style `images/`, `meta/meta.csv`, and train/valid/test index files were generated; the audit detected no overlap or corrupt images.
4. **Model forward pass:** all seven heads returned the configured dimensions: 3, 3, 5, 4, 3, 2, and 8 classes (28 categorical states total).
5. **Training smoke test:** one epoch of concept-model and black-box training completed on a tiny synthetic dataset using the development-only `tiny_cnn` backbone.
6. **Pipeline smoke test:** learned-CBM/oracle-CBM fitting, temperature calibration, final metacognitive evaluation, one-step intervention, ablations, bootstrap export, publication table generation, external-manifest evaluation, representation analysis, and concept-noise robustness all executed end-to-end on synthetic data.

The synthetic outputs are deliberately **not** included in the release package and must not be interpreted as research results. The first operation on the real Derm7pt copy must be `scripts/00_audit_data.py`.
