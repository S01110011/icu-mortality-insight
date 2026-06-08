from icu_mortality_insight.data import save_synthetic_dataset
from icu_mortality_insight.inference import load_model, predict_mortality_risk
from icu_mortality_insight.train import train_models


def test_training_pipeline_exports_model_and_reports(tmp_path):
    data_path = tmp_path / "icu.csv"
    model_path = tmp_path / "model.joblib"
    report_dir = tmp_path / "reports"

    save_synthetic_dataset(data_path, rows=700, seed=456)
    metrics = train_models(data_path, model_path, report_dir)

    assert model_path.exists()
    assert (report_dir / "metrics.json").exists()
    assert (report_dir / "confusion_matrix.csv").exists()
    assert (report_dir / "feature_importance.csv").exists()
    assert 0 <= metrics["roc_auc"] <= 1

    model = load_model(model_path)
    result = predict_mortality_risk(
        {
            "age": 78,
            "comorbidity_score": 5,
            "heart_rate": 124,
            "respiratory_rate": 31,
            "systolic_bp": 84,
            "spo2": 86,
            "temperature_c": 38.6,
            "gcs": 11,
            "fio2": 0.72,
            "lactate": 5.8,
            "creatinine": 2.4,
            "bilirubin": 2.1,
            "platelets": 82,
            "wbc": 18.5,
            "ventilated": 1,
            "vasopressor": 1,
            "admission_source_ed": 1,
        },
        model,
    )

    assert 0 <= result["mortality_probability"] <= 1
    assert result["risk_band"] in {"low", "moderate", "high"}

