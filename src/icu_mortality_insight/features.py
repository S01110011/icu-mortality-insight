"""Clinical feature engineering for ICU mortality modeling."""

from __future__ import annotations

import pandas as pd


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["shock_index"] = enriched["heart_rate"] / enriched["systolic_bp"].clip(lower=1)
    enriched["pao2_fio2_proxy"] = enriched["spo2"] / enriched["fio2"].clip(lower=0.21)
    enriched["hypoxemia_flag"] = (enriched["spo2"] < 90).astype(int)
    enriched["renal_dysfunction_flag"] = (enriched["creatinine"] >= 2.0).astype(int)
    enriched["thrombocytopenia_flag"] = (enriched["platelets"] < 100).astype(int)
    enriched["altered_mental_status_flag"] = (enriched["gcs"] < 15).astype(int)
    return enriched

