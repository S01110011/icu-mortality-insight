from icu_mortality_insight.config import TARGET_COLUMN
from icu_mortality_insight.data import generate_synthetic_icu_data


def test_generate_synthetic_icu_data_has_expected_columns_and_target():
    df = generate_synthetic_icu_data(rows=300, seed=123)

    assert len(df) == 300
    assert TARGET_COLUMN in df.columns
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})
    assert df["spo2"].between(70, 100).all()
    assert df["gcs"].between(3, 15).all()

