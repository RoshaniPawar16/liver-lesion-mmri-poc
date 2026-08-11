# Failure analysis

Threshold: 0.50 (fixed for this report).
Misclassified cases: 29 of 105 test patients.

## Grad-CAM observations (misclassified cases)

- **MR113033** (true=benign, pred=malignant, p=0.80): Grad-CAM attention concentrated at lesion periphery or adjacent tissue.
- **MR130096** (true=malignant, pred=benign, p=0.24): Grad-CAM attention concentrated at lesion periphery or adjacent tissue.


*Observations are drawn from the Grad-CAM outputs above, not from priors about lesion biology.*
