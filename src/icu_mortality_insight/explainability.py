"""Explainability exports for ICU mortality models."""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from icu_mortality_insight.config import FEATURE_COLUMNS, RANDOM_STATE


def explain_model(model, x_validation: pd.DataFrame, y_validation: pd.Series) -> pd.DataFrame:
    """Return feature importance with SHAP when available, otherwise permutation importance."""
    shap_result = _try_shap_explain(model, x_validation)
    if shap_result is not None:
        return shap_result

    result = permutation_importance(
        model,
        x_validation[FEATURE_COLUMNS],
        y_validation,
        scoring="roc_auc",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
                "method": "permutation_importance",
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _try_shap_explain(model, x_validation: pd.DataFrame) -> pd.DataFrame | None:
    try:
        import numpy as np
        import shap
    except ImportError:
        return None

    try:
        sample = x_validation[FEATURE_COLUMNS].head(300)
        classifier = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocessor"]
        transformed = preprocessor.transform(sample)
        explainer = shap.Explainer(classifier, transformed)
        values = explainer(transformed)
        raw_values = values.values
        if raw_values.ndim == 3:
            raw_values = raw_values[:, :, 1]
        importance = np.abs(raw_values).mean(axis=0)
        return (
            pd.DataFrame(
                {
                    "feature": FEATURE_COLUMNS,
                    "importance": importance,
                    "importance_std": 0.0,
                    "method": "shap_mean_abs",
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        return None

