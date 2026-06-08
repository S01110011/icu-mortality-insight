"""Project configuration."""

from pathlib import Path

RANDOM_STATE = 42
TARGET_COLUMN = "mortality_event"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "icu_synthetic.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "icu_mortality_model.joblib"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

BASE_FEATURE_COLUMNS = [
    "age",
    "comorbidity_score",
    "heart_rate",
    "respiratory_rate",
    "systolic_bp",
    "spo2",
    "temperature_c",
    "gcs",
    "fio2",
    "lactate",
    "creatinine",
    "bilirubin",
    "platelets",
    "wbc",
    "ventilated",
    "vasopressor",
    "admission_source_ed",
]

DERIVED_FEATURE_COLUMNS = [
    "shock_index",
    "pao2_fio2_proxy",
    "hypoxemia_flag",
    "renal_dysfunction_flag",
    "thrombocytopenia_flag",
    "altered_mental_status_flag",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS

