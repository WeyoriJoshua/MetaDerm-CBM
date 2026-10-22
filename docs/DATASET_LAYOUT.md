# Derm7pt data contract

The code expects the official release structure. `meta.csv` is treated as the source of truth; split CSV values are row indexes into that file.

Seven categorical concept heads are retained:

| Concept | Configured states |
|---|---|
| pigment_network | absent, typical, atypical |
| streaks | absent, regular, irregular |
| pigmentation | absent, diffuse_regular, localized_regular, localized_irregular, diffuse_irregular |
| regression_structures | absent, blue_areas, white_areas, combinations |
| dots_and_globules | absent, regular, irregular |
| blue_whitish_veil | absent, present |
| vascular_structures | absent, comma, wreath, arborizing, hairpin, dotted, within_regression, linear_irregular |

If your downloaded release uses different spellings, **do not silently recode them**. First inspect `results/audit/concept_summary.csv`, then document and version any mapping in the configuration/code.
