"""Model evaluation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    roc_auc: float
    average_precision: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_predictions(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float = 0.35,
) -> BinaryMetrics:
    y_pred = (y_probability >= threshold).astype(int)
    return BinaryMetrics(
        roc_auc=float(roc_auc_score(y_true, y_probability)),
        average_precision=float(average_precision_score(y_true, y_probability)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        threshold=threshold,
    )


def confusion_matrix_table(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float = 0.35,
) -> pd.DataFrame:
    y_pred = (y_probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["actual_survived", "actual_mortality"],
        columns=["predicted_survived", "predicted_mortality"],
    )

