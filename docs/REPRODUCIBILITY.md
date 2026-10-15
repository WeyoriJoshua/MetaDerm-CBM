# Reproducibility Guide

## Scope

This guide describes how to reproduce the **validated manuscript analyses**. The workstation is a demonstrator and should not be used as a substitute for the quantitative experiments.

## Environment

The development environment used Python 3.11.x with PyTorch/CUDA on an NVIDIA GPU.

Create a clean environment and install the repository requirements. Record exact package versions for the tagged release used in the manuscript.

## Data

Obtain Derm7pt and PH² from their official research sources and respect all access/licensing conditions.

Do not commit dataset images to this repository.

## Experimental design

- Primary visual backbone: EfficientNet-B3.
- Sensitivity backbone: Swin-Tiny.
- Seeds: 2026, 2027, 2028, 2029, 2030.
- Five-seed reporting is used for internal diagnostic, selective, and intervention analyses.
- PH² is a locked external validation cohort and must not be used for retraining, recalibration, or threshold optimization.

## Statistical analysis

The manuscript uses:
- mean ± standard deviation across seeds;
- 10,000-resample bootstrap analyses;
- paired patient-level comparisons where applicable;
- McNemar tests with Holm correction for the specified PH² comparisons.

## Important interpretation rules

- Derm7pt test performance is internal evaluation because Derm7pt influenced development.
- PH² is the external confirmatory cohort.
- Expert intervention is oracle-simulated rather than a prospective clinician study.
- Selective `PREDICT` performance must be reported together with balanced accuracy, sensitivity, specificity, and class-conditional coverage.
- The later input-suitability gate is a prototype extension unless separately evaluated in a formal OOD study.
