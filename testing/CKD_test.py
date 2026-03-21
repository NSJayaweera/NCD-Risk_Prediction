import json
import sys
from pathlib import Path
from unittest.mock import mock_open, patch

import joblib
import numpy as np
import pandas as pd
import pytest

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import CKD  # noqa: E402

SAVE_DIR = PROJECT_ROOT / "ckdModel"

EXPECTED_FEATURE_ORDER = [
    "age",
    "gender",
    "bmi",
    "bp_systolic",
    "bp_diastolic",
    "serum_creatinine",
    "blood_urea_nitrogen",
    "urine_albumin",
    "urine_creatinine",
    "albumin_creatinine_ratio",
    "albumin_serum",
    "uric_acid",
    "diabetes_diagnosed",
    "bun_creatinine_ratio",
]


class DummyContext:
    def __init__(self, collectors=None):
        self.collectors = collectors

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def metric(self, *args, **kwargs):
        if self.collectors is not None:
            self.collectors["metric"].append((args, kwargs))


class FakeModel:
    def __init__(self, pred=0, prob=0.2):
        self.pred = pred
        self.prob = prob
        self.last_input = None

    def predict_proba(self, x):
        self.last_input = x.copy()
        return np.array([[1.0 - self.prob, self.prob]])

    def predict(self, x):
        self.last_input = x.copy()
        return np.array([self.pred])

def identity_cache_resource(func=None, **kwargs):
    if func is None:
        return lambda f: f
    return func


def patch_streamlit(
    monkeypatch,
    *,
    model_choice="Both",
    gender="Female",
    diabetes="No",
    button_value=True,
    number_overrides=None,
):
    if number_overrides is None:
        number_overrides = {}

    collectors = {
        "error": [],
        "success": [],
        "warning": [],
        "progress": [],
        "metric": [],
    }

    def fake_radio(label, *args, **kwargs):
        if label == "Choose Prediction Model":
            return model_choice
        if label == "Gender":
            return gender
        if label == "Diabetes Diagnosed":
            return diabetes
        options = kwargs.get("options", [])
        return options[0] if options else None

    def fake_number_input(label, *args, **kwargs):
        return number_overrides.get(label, kwargs.get("value"))

    def fake_columns(spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [DummyContext(collectors) for _ in range(count)]

    monkeypatch.setattr(CKD.st, "cache_resource", identity_cache_resource)
    monkeypatch.setattr(CKD.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(CKD.st, "title", lambda *args, **kwargs: None)
    monkeypatch.setattr(CKD.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(CKD.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(CKD.st, "radio", fake_radio)
    monkeypatch.setattr(CKD.st, "number_input", fake_number_input)
    monkeypatch.setattr(CKD.st, "columns", fake_columns)
    monkeypatch.setattr(CKD.st, "button", lambda *args, **kwargs: button_value)
    monkeypatch.setattr(
        CKD.st,
        "metric",
        lambda *args, **kwargs: collectors["metric"].append((args, kwargs)),
    )
    monkeypatch.setattr(
        CKD.st,
        "progress",
        lambda value, *args, **kwargs: collectors["progress"].append(value),
    )
    monkeypatch.setattr(
        CKD.st,
        "error",
        lambda message, *args, **kwargs: collectors["error"].append(str(message)),
    )
    monkeypatch.setattr(
        CKD.st,
        "success",
        lambda message, *args, **kwargs: collectors["success"].append(str(message)),
    )
    monkeypatch.setattr(
        CKD.st,
        "warning",
        lambda message, *args, **kwargs: collectors["warning"].append(str(message)),
    )

    return collectors


def test_feature_engineering_logic():
    weight = 60.0
    height_cm = 165.0
    bmi = round(weight / ((height_cm / 100) ** 2), 1)
    assert bmi == 22.0

    urine_albumin = 1.0
    urine_creatinine = 200.0
    acr = round((urine_albumin / urine_creatinine) * 1000, 2)
    assert acr == 5.0

    blood_urea_nitrogen = 10.0
    serum_creatinine = 0.7
    bun_cr_ratio = round(blood_urea_nitrogen / serum_creatinine, 2)
    assert bun_cr_ratio == 14.29


def test_zero_guards_and_probability_clamping():
    urine_albumin = 10.0
    urine_creatinine = 0
    acr = round((urine_albumin / urine_creatinine) * 1000, 2) if urine_creatinine > 0 else 0
    assert acr == 0

    blood_urea_nitrogen = 15.0
    serum_creatinine = 0
    bun_cr_ratio = round(blood_urea_nitrogen / serum_creatinine, 2) if serum_creatinine > 0 else 0
    assert bun_cr_ratio == 0

    assert max(0.0, min(1.0, float(-0.5))) == 0.0
    assert max(0.0, min(1.0, float(1.5))) == 1.0
    assert max(0.0, min(1.0, float(0.72))) == 0.72


def test_feature_order_json_is_valid():
    feature_order_path = SAVE_DIR / "feature_order.json"
    assert feature_order_path.exists(), f"Missing file: {feature_order_path}"

    with open(feature_order_path, "r", encoding="utf-8") as f:
        feature_order = json.load(f)

    assert feature_order == EXPECTED_FEATURE_ORDER
    assert len(feature_order) == 14
    assert len(feature_order) == len(set(feature_order)), "Duplicate features found in feature_order.json"

    leakage_cols = {"egfr", "ckd_stage", "ckd_present", "participant_id"}
    assert leakage_cols.isdisjoint(feature_order), "Leakage column found in feature_order.json"


def test_dataframe_alignment_matches_feature_order():
    input_data = {
        "age": 30,
        "gender": 0,
        "bmi": 22.0,
        "bp_systolic": 110,
        "bp_diastolic": 70,
        "serum_creatinine": 0.7,
        "blood_urea_nitrogen": 10.0,
        "urine_albumin": 1.0,
        "urine_creatinine": 200.0,
        "albumin_creatinine_ratio": 5.0,
        "albumin_serum": 4.5,
        "uric_acid": 4.5,
        "diabetes_diagnosed": 0,
        # bun_creatinine_ratio intentionally missing
    }

    df = pd.DataFrame([input_data])

    for col in EXPECTED_FEATURE_ORDER:
        if col not in df.columns:
            df[col] = np.nan

    final_df = df[EXPECTED_FEATURE_ORDER]

    assert list(final_df.columns) == EXPECTED_FEATURE_ORDER
    assert final_df.shape == (1, 14)
    assert pd.isna(final_df.loc[0, "bun_creatinine_ratio"])


def test_run_ckd_analysis_handles_model_loading_failure(monkeypatch):
    collectors = patch_streamlit(monkeypatch, button_value=False)

    def fake_joblib_load(_):
        raise Exception("File not found")

    monkeypatch.setattr(CKD.joblib, "load", fake_joblib_load)

    CKD.run_ckd_analysis()

    assert collectors["error"], "Expected streamlit.error to be called"
    assert "Error loading models" in collectors["error"][0]


def test_run_ckd_analysis_when_button_not_clicked(monkeypatch):
    collectors = patch_streamlit(monkeypatch, button_value=False)

    stacking = FakeModel(pred=0, prob=0.12)
    bagging = FakeModel(pred=0, prob=0.18)

    def fake_joblib_load(path):
        path = str(path)
        if path.endswith("stacking_model.pkl"):
            return stacking
        if path.endswith("bagging_model.pkl"):
            return bagging
        raise FileNotFoundError(path)

    monkeypatch.setattr(CKD.joblib, "load", fake_joblib_load)

    with patch("builtins.open", mock_open(read_data=json.dumps(EXPECTED_FEATURE_ORDER))):
        CKD.run_ckd_analysis()

    assert stacking.last_input is None
    assert bagging.last_input is None
    assert len(collectors["progress"]) == 0


def test_run_ckd_analysis_builds_expected_feature_frame(monkeypatch):
    collectors = patch_streamlit(
        monkeypatch,
        model_choice="Both",
        gender="Female",
        diabetes="No",
        button_value=True,
        number_overrides={
            "Age (years)": 30,
            "Weight (kg)": 60.0,
            "Height (cm)": 165.0,
            "Systolic BP (mmHg)": 110,
            "Diastolic BP (mmHg)": 70,
            "Blood Urea Nitrogen (mg/dL)": 10.0,
            "Urine Albumin (mg/L)": 1.0,
            "Urine Creatinine (mg/dL)": 200.0,
            "Serum Albumin (g/dL)": 4.5,
            "Serum Creatinine (mg/dL)": 0.7,
            "Uric Acid (mg/dL)": 4.5,
        },
    )

    stacking = FakeModel(pred=0, prob=0.12)
    bagging = FakeModel(pred=0, prob=0.18)

    def fake_joblib_load(path):
        path = str(path)
        if path.endswith("stacking_model.pkl"):
            return stacking
        if path.endswith("bagging_model.pkl"):
            return bagging
        raise FileNotFoundError(path)

    monkeypatch.setattr(CKD.joblib, "load", fake_joblib_load)

    mocked_feature_order = json.dumps(EXPECTED_FEATURE_ORDER)
    with patch("builtins.open", mock_open(read_data=mocked_feature_order)):
        CKD.run_ckd_analysis()

    expected_df = pd.DataFrame(
        [
            {
                "age": 30,
                "gender": 0,
                "bmi": 22.0,
                "bp_systolic": 110,
                "bp_diastolic": 70,
                "serum_creatinine": 0.7,
                "blood_urea_nitrogen": 10.0,
                "urine_albumin": 1.0,
                "urine_creatinine": 200.0,
                "albumin_creatinine_ratio": 5.0,
                "albumin_serum": 4.5,
                "uric_acid": 4.5,
                "diabetes_diagnosed": 0,
                "bun_creatinine_ratio": 14.29,
            }
        ]
    )[EXPECTED_FEATURE_ORDER]

    pd.testing.assert_frame_equal(
        stacking.last_input.reset_index(drop=True),
        expected_df.reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        bagging.last_input.reset_index(drop=True),
        expected_df.reset_index(drop=True),
        check_dtype=False,
    )

    assert len(collectors["progress"]) == 2


def test_stacking_only_renders_single_progress_bar(monkeypatch):
    collectors = patch_streamlit(
        monkeypatch,
        model_choice="Stacking Model",
        gender="Female",
        diabetes="No",
        button_value=True,
    )

    stacking = FakeModel(pred=0, prob=0.11)
    bagging = FakeModel(pred=1, prob=0.91)

    def fake_joblib_load(path):
        path = str(path)
        if path.endswith("stacking_model.pkl"):
            return stacking
        if path.endswith("bagging_model.pkl"):
            return bagging
        raise FileNotFoundError(path)

    monkeypatch.setattr(CKD.joblib, "load", fake_joblib_load)

    with patch("builtins.open", mock_open(read_data=json.dumps(EXPECTED_FEATURE_ORDER))):
        CKD.run_ckd_analysis()

    assert len(collectors["progress"]) == 1
    assert 0.0 <= collectors["progress"][0] <= 1.0


def test_bagging_only_renders_single_progress_bar(monkeypatch):
    collectors = patch_streamlit(
        monkeypatch,
        model_choice="Bagging Model",
        gender="Female",
        diabetes="No",
        button_value=True,
    )

    stacking = FakeModel(pred=1, prob=0.88)
    bagging = FakeModel(pred=0, prob=0.21)

    def fake_joblib_load(path):
        path = str(path)
        if path.endswith("stacking_model.pkl"):
            return stacking
        if path.endswith("bagging_model.pkl"):
            return bagging
        raise FileNotFoundError(path)

    monkeypatch.setattr(CKD.joblib, "load", fake_joblib_load)

    with patch("builtins.open", mock_open(read_data=json.dumps(EXPECTED_FEATURE_ORDER))):
        CKD.run_ckd_analysis()

    assert len(collectors["progress"]) == 1
    assert 0.0 <= collectors["progress"][0] <= 1.0


def test_both_models_agree_no_ckd_shows_success(monkeypatch):
    collectors = patch_streamlit(
        monkeypatch,
        model_choice="Both",
        gender="Female",
        diabetes="No",
        button_value=True,
    )

    stacking = FakeModel(pred=0, prob=0.18)
    bagging = FakeModel(pred=0, prob=0.24)

    def fake_joblib_load(path):
        path = str(path)
        if path.endswith("stacking_model.pkl"):
            return stacking
        if path.endswith("bagging_model.pkl"):
            return bagging
        raise FileNotFoundError(path)

    monkeypatch.setattr(CKD.joblib, "load", fake_joblib_load)

    with patch("builtins.open", mock_open(read_data=json.dumps(EXPECTED_FEATURE_ORDER))):
        CKD.run_ckd_analysis()

    assert any("No CKD Detected" in msg for msg in collectors["success"])


def test_both_models_agree_ckd_shows_error(monkeypatch):
    collectors = patch_streamlit(
        monkeypatch,
        model_choice="Both",
        gender="Female",
        diabetes="No",
        button_value=True,
    )

    stacking = FakeModel(pred=1, prob=0.91)
    bagging = FakeModel(pred=1, prob=0.87)

    def fake_joblib_load(path):
        path = str(path)
        if path.endswith("stacking_model.pkl"):
            return stacking
        if path.endswith("bagging_model.pkl"):
            return bagging
        raise FileNotFoundError(path)

    monkeypatch.setattr(CKD.joblib, "load", fake_joblib_load)

    with patch("builtins.open", mock_open(read_data=json.dumps(EXPECTED_FEATURE_ORDER))):
        CKD.run_ckd_analysis()

    assert any("CKD Detected" in msg for msg in collectors["error"])


def test_models_disagree_shows_warning(monkeypatch):
    collectors = patch_streamlit(
        monkeypatch,
        model_choice="Both",
        gender="Female",
        diabetes="No",
        button_value=True,
    )

    stacking = FakeModel(pred=1, prob=0.91)
    bagging = FakeModel(pred=0, prob=0.24)

    def fake_joblib_load(path):
        path = str(path)
        if path.endswith("stacking_model.pkl"):
            return stacking
        if path.endswith("bagging_model.pkl"):
            return bagging
        raise FileNotFoundError(path)

    monkeypatch.setattr(CKD.joblib, "load", fake_joblib_load)

    with patch("builtins.open", mock_open(read_data=json.dumps(EXPECTED_FEATURE_ORDER))):
        CKD.run_ckd_analysis()

    assert any("Models disagree" in msg for msg in collectors["warning"])


@pytest.mark.skipif(
    not (
        (SAVE_DIR / "stacking_model.pkl").exists()
        and (SAVE_DIR / "bagging_model.pkl").exists()
        and (SAVE_DIR / "feature_order.json").exists()
    ),
    reason="Model artifacts not found in ckdModel/",
)
def test_model_artifacts_can_load_and_predict_binary_output():
    stacking = joblib.load(SAVE_DIR / "stacking_model.pkl")
    bagging = joblib.load(SAVE_DIR / "bagging_model.pkl")

    with open(SAVE_DIR / "feature_order.json", "r", encoding="utf-8") as f:
        feature_order = json.load(f)

    patient_df = pd.DataFrame(
        [
            {
                "age": 43.0,
                "gender": 1,
                "bmi": 27.0,
                "bp_systolic": 135.0,
                "bp_diastolic": 98.0,
                "serum_creatinine": 0.8,
                "blood_urea_nitrogen": 11.0,
                "urine_albumin": 23.12,
                "urine_creatinine": 136.0,
                "albumin_creatinine_ratio": 17.0,
                "albumin_serum": 4.3,
                "uric_acid": 5.1,
                "diabetes_diagnosed": 0.0,
                "bun_creatinine_ratio": 13.75,
            }
        ]
    )[feature_order]

    for model in (stacking, bagging):
        pred = model.predict(patient_df)
        prob = model.predict_proba(patient_df)

        assert pred.shape == (1,)
        assert prob.shape == (1, 2)
        assert pred[0] in (0, 1)
        assert 0.0 <= prob[0][1] <= 1.0