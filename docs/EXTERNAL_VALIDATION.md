# External validation protocol

External validation should be performed after the Derm7pt pipeline is frozen.

Create a manifest CSV with:

```text
image_path,label
image_001.bmp,1
image_002.bmp,0
```

Add concept columns only when the external dataset genuinely provides compatible expert annotations. Any mapping between external concept terminology and Derm7pt states must be specified in the manuscript and code; unsupported concepts must be marked unavailable rather than imputed from the Derm7pt labels.

Recommended reporting:

- external AUROC/AUPRC/sensitivity/specificity;
- calibration degradation;
- change in selective risk at matched coverage;
- concept metrics only for genuinely overlapping annotated concepts;
- subgroup/domain-shift analysis where sample size supports it.
