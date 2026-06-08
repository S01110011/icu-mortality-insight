"""Synthetic ICU cohort generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from icu_mortality_insight.config import DEFAULT_DATA_PATH, RANDOM_STATE
from icu_mortality_insight.features import add_clinical_features


def generate_synthetic_icu_data(rows: int = 6000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate synthetic structured ICU observations with a mortality label."""
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(64, 17, rows), 18, 98).round().astype(int)
    comorbidity_score = np.clip(rng.poisson(2.2 + (age > 70) * 1.1, rows), 0, 10)
    admission_source_ed = rng.binomial(1, 0.56, rows)

    severity = rng.gamma(shape=2.0, scale=1.0, size=rows) + (age > 75) * 0.8 + comorbidity_score * 0.22

    heart_rate = np.clip(rng.normal(88, 18, rows) + severity * 4, 42, 180)
    respiratory_rate = np.clip(rng.normal(20, 5, rows) + severity * 1.2, 8, 46)
    systolic_bp = np.clip(rng.normal(118, 22, rows) - severity * 4.5, 62, 220)
    spo2 = np.clip(rng.normal(95, 3.2, rows) - severity * 0.9, 70, 100)
    temperature_c = np.clip(rng.normal(37.0, 0.8, rows), 33.5, 41.8)
    gcs = np.clip(np.round(rng.normal(14.4, 1.4, rows) - severity * 0.35), 3, 15).astype(int)
    fio2 = np.clip(rng.normal(0.32, 0.12, rows) + severity * 0.025, 0.21, 1.0)
    lactate = np.clip(rng.lognormal(mean=0.45, sigma=0.55, size=rows) + severity * 0.25, 0.4, 14)
    creatinine = np.clip(rng.lognormal(mean=0.08, sigma=0.45, size=rows) + severity * 0.12, 0.3, 8)
    bilirubin = np.clip(rng.lognormal(mean=-0.2, sigma=0.7, size=rows) + severity * 0.08, 0.1, 18)
    platelets = np.clip(rng.normal(230, 75, rows) - severity * 17, 18, 650)
    wbc = np.clip(rng.normal(10.5, 4.0, rows) + severity * 0.7, 1.0, 45)
    ventilated = rng.binomial(1, np.clip(0.10 + severity * 0.075 + (spo2 < 90) * 0.25, 0, 0.90))
    vasopressor = rng.binomial(1, np.clip(0.06 + severity * 0.065 + (systolic_bp < 90) * 0.35, 0, 0.85))

    df = pd.DataFrame(
        {
            "age": age,
            "comorbidity_score": comorbidity_score,
            "heart_rate": heart_rate.round(1),
            "respiratory_rate": respiratory_rate.round(1),
            "systolic_bp": systolic_bp.round(1),
            "spo2": spo2.round(1),
            "temperature_c": temperature_c.round(1),
            "gcs": gcs,
            "fio2": fio2.round(2),
            "lactate": lactate.round(2),
            "creatinine": creatinine.round(2),
            "bilirubin": bilirubin.round(2),
            "platelets": platelets.round(1),
            "wbc": wbc.round(1),
            "ventilated": ventilated,
            "vasopressor": vasopressor,
            "admission_source_ed": admission_source_ed,
        }
    )
    df = add_clinical_features(df)

    logit = (
        -7.0
        + 0.030 * df["age"]
        + 0.24 * df["comorbidity_score"]
        + 0.038 * (df["heart_rate"] - 85)
        + 0.080 * (df["respiratory_rate"] - 20)
        - 0.042 * (df["systolic_bp"] - 115)
        - 0.105 * (df["spo2"] - 94)
        + 0.48 * df["lactate"]
        + 0.36 * df["creatinine"]
        + 0.075 * df["bilirubin"]
        - 0.006 * (df["platelets"] - 180)
        + 0.85 * df["ventilated"]
        + 0.95 * df["vasopressor"]
        + 0.60 * df["altered_mental_status_flag"]
        + rng.normal(0, 0.75, rows)
    )
    probability = 1 / (1 + np.exp(-logit))
    df["mortality_event"] = rng.binomial(1, np.clip(probability, 0.01, 0.98))
    return df


def save_synthetic_dataset(output: Path, rows: int, seed: int = RANDOM_STATE) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_icu_data(rows=rows, seed=seed).to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ICU mortality data.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    path = save_synthetic_dataset(args.output, rows=args.rows, seed=args.seed)
    print(f"Saved synthetic ICU dataset to {path}")


if __name__ == "__main__":
    main()

