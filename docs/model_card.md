# ICU Mortality Insight Model Card

## Intended Use

This model is intended for educational and portfolio demonstration of ICU mortality risk modeling from structured clinical data.

It is not intended for real clinical care.

## Target Outcome

The target variable is `mortality_event`, a synthetic in-hospital mortality label.

In a real implementation, the target definition should be reviewed by clinical, data governance and ethics stakeholders. Possible outcomes include:

- ICU mortality
- In-hospital mortality
- 30-day mortality
- Composite critical deterioration outcome

## Inputs

The synthetic dataset includes age, comorbidity burden, vital signs, oxygenation, mental status, ventilation status, vasopressor use and laboratory values.

## Evaluation

Recommended evaluation metrics:

- ROC-AUC
- Average precision
- Recall
- Precision
- F1 score
- Confusion matrix
- Calibration curve
- Subgroup analysis

## Governance Requirements

Before production use:

- Validate on de-identified local ICU data
- Calibrate probabilities by hospital and ICU population
- Review false negatives clinically
- Evaluate bias by age, sex, diagnosis group and service line
- Monitor model drift and alert burden
- Keep human review in the workflow

