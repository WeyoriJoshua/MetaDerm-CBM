# MetaDerm-CBM

**MetaDerm-CBM** is an inconsistency-aware metacognitive concept bottleneck framework for selective and interpretable skin-lesion decision support.

The framework combines calibrated visual diagnosis, seven probabilistic dermoscopic concept heads, soft seven-point clinical reasoning, concept-profile reliability assessment, metacognitive **PREDICT / QUERY / ABSTAIN** actions, reasoning-impact-guided concept verification, simulated expert concept intervention, external PH² validation, backbone sensitivity, robustness analysis, and an interactive research workstation.

> **Research use only.** MetaDerm-CBM is not a medical device and is not intended for autonomous clinical diagnosis.

## Authors

- Joshua Akowuje Weyori
- Emmanuel Ayerko Herson
- Richmond Ansah-Sarpong
- Isaac Baiden

## Manuscript

**MetaDerm-CBM: An Inconsistency-Aware Metacognitive Concept Bottleneck Framework for Selective and Interpretable Skin Lesion Diagnosis**

## Validated experimental scope

The accompanying study reports:

- Derm7pt internal development/evaluation across five seeds (2026–2030);
- black-box, soft reasoner, predicted-CBM, and oracle-CBM comparisons;
- concept calibration and concept-level evaluation;
- selective risk/coverage and class-conditional coverage;
- targeted oracle-simulated concept intervention;
- semantic corruption robustness;
- EfficientNet-B3 vs Swin-Tiny sensitivity;
- locked PH² external validation without retraining, recalibration, or threshold adjustment;
- external concept-transfer analysis;
- PCA/t-SNE representation visualization and Grad-CAM.

The later input-suitability/OOD safeguard in the workstation is a **prototype safety extension** and should not be described as clinically validated unless a dedicated OOD evaluation is reported.

## Repository structure

```text
MetaDerm-CBM/
├── configs/
├── scripts/
├── src/
│   └── metaderm/
├── system/
├── tests/
├── docs/
├── results/
│   └── publication/
├── requirements.txt
├── run_system.ps1
├── README.md
├── CITATION.cff
└── .gitignore
```

## Data

The project uses the Derm7pt and PH² research datasets subject to their original access and redistribution conditions. Dataset files and restricted medical images must remain outside version control.

## Reproducibility

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Research workstation

See [docs/SYSTEM_WORKSTATION.md](docs/SYSTEM_WORKSTATION.md).

## Citation

Machine-readable citation metadata are provided in `CITATION.cff`. A permanent DOI can be added after archiving a tagged software release (for example with Zenodo).

## License

A software license has not yet been selected. Choose an appropriate open-source license before the code release is treated as fully reusable.
