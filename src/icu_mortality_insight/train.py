"""Training workflow for ICU mortality models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from icu_mortality_insight.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPORT_DIR,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from icu_mortality_insight.data import save_synthetic_dataset
from icu_mortality_insight.evaluation import confusion_matrix_table, evaluate_predictions
from icu_mortality_insight.explainability import explain_model
from icu_mortality_insight.features import add_clinical_features
from icu_mortality_insight.pipeline import build_candidate_models


def train_models(
    data_path: Path,
    model_out: Path,
    report_dir: Path,
    threshold: float = 0.35,
) -> dict[str, float | str]:
    if not data_path.exists():
        save_synthetic_dataset(data_path, rows=6000)

    df = add_clinical_features(pd.read_csv(data_path))
    missing_columns = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    comparison_rows = []
    trained_models = {}
    for model_name, model in build_candidate_models().items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics = evaluate_predictions(y_test.to_numpy(), probabilities, threshold).to_dict()
        comparison_rows.append({"model_name": model_name, **metrics})
        trained_models[model_name] = model

    comparison = pd.DataFrame(comparison_rows).sort_values("roc_auc", ascending=False)
    best_name = str(comparison.iloc[0]["model_name"])
    best_model = trained_models[best_name]
    best_probabilities = best_model.predict_proba(x_test)[:, 1]
    best_metrics = evaluate_predictions(y_test.to_numpy(), best_probabilities, threshold).to_dict()
    best_metrics["model_name"] = best_name
    best_metrics["train_rows"] = int(len(x_train))
    best_metrics["test_rows"] = int(len(x_test))
    best_metrics["positive_rate"] = float(y.mean())

    report_dir.mkdir(parents=True, exist_ok=True)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    (report_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    confusion_matrix_table(y_test.to_numpy(), best_probabilities, threshold).to_csv(
        report_dir / "confusion_matrix.csv"
    )
    explain_model(best_model, x_test, y_test).to_csv(report_dir / "feature_importance.csv", index=False)

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "threshold": threshold,
            "features": FEATURE_COLUMNS,
        },
        model_out,
    )
    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ICU mortality prediction models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()

    metrics = train_models(args.data, args.model_out, args.report_dir, threshold=args.threshold)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

