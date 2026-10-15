# MetaDerm-CBM Research Workstation

The Streamlit workstation operationalizes the frozen seed-2026 research pipeline.

## Main workflow

```text
Dermoscopic image
    -> visual perception
    -> seven semantic concept distributions
    -> direct neural diagnosis + soft clinical reasoner
    -> profile/reliability metacognition
    -> PREDICT / QUERY / ABSTAIN
    -> optional expert concept verification
    -> updated reasoning
```

## Interface components

- calibrated black-box melanoma probability;
- concept-based reasoner probability;
- seven concept predictions with confidence/uncertainty;
- reliability and profile-reliability indicators;
- PREDICT / QUERY / ABSTAIN action;
- decision-critical concept ranking;
- simulated expert concept correction;
- Grad-CAM localization;
- structured case audit/export.

## Safety statement

This is a research prototype, not a medical device.

The later v1.4 upstream input-suitability gate is a defensive software feature added after the primary diagnostic experiments were frozen. It must not be represented as clinically validated OOD detection without dedicated evaluation.
