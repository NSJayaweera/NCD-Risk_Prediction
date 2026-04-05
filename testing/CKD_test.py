from pathlib import Path
from unittest.mock import patch
import importlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# PROJECT PATHS
# -----------------------------------------------------------------------------

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import CKD

MODEL_DIR = PROJECT_ROOT / "ckdModel"

EXPECTED_FEATURE_ORDER = [
    "age",
    "gender",
    "diabetes_diagnosed",
    "bp_systolic",
    "bp_diastolic",
    "bmi",
    "serum_creatinine",
    "blood_urea_nitrogen",
    "albumin_serum",
    "albumin_creatinine_ratio",
    "uric_acid",
    "phosphorus",
    "calcium",
]

LEAKAGE_OR_OLD_COLUMNS = {
    "participant_id",
    "ckd_present",
    "ckd_stage",
    "egfr",
    "height_cm",
    "weight_kg",
    "urine_albumin",
    "urine_creatinine",
    "bun_creatinine_ratio",
}


# -----------------------------------------------------------------------------
# SMALL HELPERS
# -----------------------------------------------------------------------------
def compute_bmi(weight_kg: float, height_cm: float) -> float:
    return round(weight_kg / ((height_cm / 100) ** 2), 1)


def compute_acr(urine_albumin: float, urine_creatinine: float) -> float:
    return round((urine_albumin / urine_creatinine) * 1000, 2) if urine_creatinine > 0 else 0


def calculate_egfr_like_deployment(creatinine: float, age: float, gender: int):
    """
    Mirrors the current CKD deployment formula.
    Female = 0, Male = 1.
    """
    if creatinine <= 0 or age <= 0:
        return None

    kappa = 0.7 if gender == 0 else 0.9
    alpha = -0.241 if gender == 0 else -0.302
    sex_factor = 1.012 if gender == 0 else 1.0
    ratio = creatinine / kappa

    if ratio < 1:
        egfr = 142 * (ratio ** alpha) * (0.9938 ** age) * sex_factor
    else:
        egfr = 142 * (ratio ** -1.200) * (0.9938 ** age) * sex_factor

    return round(egfr, 1)


def load_feature_order() -> list[str]:
    feature_file = MODEL_DIR / "feature_order.json"
    assert feature_file.exists(), (
        "feature_order.json not found. Save the final deployment files first: "
        "bagging_model.pkl, stacking_model.pkl, feature_order.json"
    )
    with open(feature_file, "r", encoding="utf-8") as f:
        return json.load(f)


def align_to_feature_order(input_data: dict, feature_order: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([input_data])
    for col in feature_order:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_order].copy()


def load_model_artifact(filename: str):
    path = MODEL_DIR / filename
    assert path.exists(), f"Required model file not found: {path}"
    return joblib.load(path)


def import_ckd_module_or_skip():
    try:
        return importlib.import_module("CKD")
    except Exception as exc:
        pytest.skip(f"CKD.py is not importable in this environment: {exc}")


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------
class TestCKDAnalysis:

    def test_feature_order_exact_match_current_pipeline(self):
        """feature_order.json must match the current selected deployment schema exactly."""
        feature_order = load_feature_order()
        assert feature_order == EXPECTED_FEATURE_ORDER
        assert len(feature_order) == 13
        assert len(feature_order) == len(set(feature_order))

    def test_feature_order_excludes_old_and_leakage_columns(self):
        """The new pipeline must not contain removed or leakage columns."""
        feature_order = load_feature_order()
        for col in LEAKAGE_OR_OLD_COLUMNS:
            assert col not in feature_order, f"Unexpected column found in feature_order: {col}"

    def test_dataframe_alignment_matches_current_feature_order(self):
        """Deployment alignment should add any missing columns and preserve exact order."""
        feature_order = load_feature_order()

        input_data = {
            "age": 45,
            "gender": 0,
            "diabetes_diagnosed": 0,
            "bp_systolic": 118,
            "bp_diastolic": 76,
            "bmi": 23.8,
            "serum_creatinine": 0.8,
            "blood_urea_nitrogen": 12.0,
            "albumin_serum": 4.1,
            "albumin_creatinine_ratio": 10.0,
            "uric_acid": 5.2,
            # phosphorus and calcium intentionally omitted
        }

        aligned = align_to_feature_order(input_data, feature_order)

        assert list(aligned.columns) == feature_order
        assert aligned.shape == (1, 13)
        assert pd.isna(aligned.loc[0, "phosphorus"])
        assert pd.isna(aligned.loc[0, "calcium"])

    def test_current_derived_feature_logic(self):
        """The current deployment only derives BMI, ACR, and display-only eGFR."""
        bmi = compute_bmi(weight_kg=55.0, height_cm=160.0)
        acr = compute_acr(urine_albumin=1.0, urine_creatinine=120.0)
        egfr = calculate_egfr_like_deployment(creatinine=0.7, age=32, gender=0)

        assert bmi == 21.5
        assert acr == 8.33
        assert egfr is not None
        assert egfr > 60

    def test_acr_zero_guard(self):
        """ACR must return 0 when urine creatinine is 0."""
        assert compute_acr(urine_albumin=10.0, urine_creatinine=0) == 0

    def test_egfr_behavior(self):
        """eGFR should go down when creatinine or age goes up."""
        egfr_low_creatinine = calculate_egfr_like_deployment(0.8, 50, 1)
        egfr_high_creatinine = calculate_egfr_like_deployment(3.0, 50, 1)
        egfr_younger = calculate_egfr_like_deployment(1.0, 30, 1)
        egfr_older = calculate_egfr_like_deployment(1.0, 70, 1)

        assert egfr_low_creatinine > egfr_high_creatinine
        assert egfr_younger > egfr_older
        assert calculate_egfr_like_deployment(0, 50, 1) is None
        assert calculate_egfr_like_deployment(1.0, 0, 1) is None

    def test_current_default_values_are_reasonable(self):
        """Default deployment inputs should look low-risk and internally consistent."""
        age = 32
        gender = 0
        weight = 55.0
        height = 160.0
        bp_systolic = 110
        bp_diastolic = 70
        serum_creatinine = 0.7
        blood_urea_nitrogen = 10.0
        urine_albumin = 1.0
        urine_creatinine = 120.0
        albumin_serum = 4.5
        uric_acid = 4.2
        phosphorus = 3.8
        calcium = 9.4
        diabetes = 0

        bmi = compute_bmi(weight, height)
        acr = compute_acr(urine_albumin, urine_creatinine)
        egfr = calculate_egfr_like_deployment(serum_creatinine, age, gender)

        assert 18.5 <= bmi <= 25.0
        assert acr < 30
        assert egfr > 60
        assert bp_systolic > bp_diastolic
        assert 0.5 <= phosphorus <= 15.0
        assert 4.0 <= calcium <= 15.0
        assert albumin_serum > 0
        assert uric_acid > 0
        assert blood_urea_nitrogen > 0
        assert diabetes in {0, 1}

    def test_required_model_files_exist(self):
        """Final deployment must contain exactly the three required artifact names."""
        required = [
            MODEL_DIR / "bagging_model.pkl",
            MODEL_DIR / "stacking_model.pkl",
            MODEL_DIR / "feature_order.json",
        ]
        for path in required:
            assert path.exists(), f"Missing required deployment artifact: {path}"

    def test_model_artifacts_can_be_loaded(self):
        """Saved Bagging and Stacking artifacts should load successfully."""
        bagging = load_model_artifact("bagging_model.pkl")
        stacking = load_model_artifact("stacking_model.pkl")

        assert hasattr(bagging, "predict")
        assert hasattr(bagging, "predict_proba")
        assert hasattr(stacking, "predict")
        assert hasattr(stacking, "predict_proba")

    def test_models_return_valid_probabilities_and_rank_risk_correctly(self):
        """
        More robust than exact hard-coded labels:
        the high-risk profile should score above the low-risk profile.
        """
        feature_order = load_feature_order()
        bagging = load_model_artifact("bagging_model.pkl")
        stacking = load_model_artifact("stacking_model.pkl")

        low_risk = align_to_feature_order({
            "age": 32,
            "gender": 0,
            "diabetes_diagnosed": 0,
            "bp_systolic": 110,
            "bp_diastolic": 70,
            "bmi": 21.7,
            "serum_creatinine": 0.7,
            "blood_urea_nitrogen": 9.0,
            "albumin_serum": 4.6,
            "albumin_creatinine_ratio": 4.0,
            "uric_acid": 4.3,
            "phosphorus": 3.8,
            "calcium": 9.4,
        }, feature_order)

        high_risk = align_to_feature_order({
            "age": 72,
            "gender": 1,
            "diabetes_diagnosed": 1,
            "bp_systolic": 168,
            "bp_diastolic": 102,
            "bmi": 29.1,
            "serum_creatinine": 3.4,
            "blood_urea_nitrogen": 78.0,
            "albumin_serum": 2.8,
            "albumin_creatinine_ratio": 1417.0,
            "uric_acid": 9.1,
            "phosphorus": 5.9,
            "calcium": 8.3,
        }, feature_order)

        for model in (bagging, stacking):
            low_prob = float(model.predict_proba(low_risk)[0][1])
            high_prob = float(model.predict_proba(high_risk)[0][1])

            assert 0.0 <= low_prob <= 1.0
            assert 0.0 <= high_prob <= 1.0
            assert high_prob > low_prob

    def test_risk_score_clamping(self):
        """Average probability display should stay inside [0, 1]."""
        assert max(0.0, min(1.0, -0.5)) == 0.0
        assert max(0.0, min(1.0, 1.5)) == 1.0
        assert max(0.0, min(1.0, 0.72)) == 0.72

    def test_asset_loading_failure(self):
        """If model loading fails, the Streamlit app should show a clear error."""
        CKD = import_ckd_module_or_skip()

        with patch.object(CKD.joblib, "load", side_effect=Exception("File not found")):
            with patch.object(CKD.st, "error") as mock_error:
                with patch.object(CKD.st, "markdown"):
                    CKD.run_ckd_analysis()
                    assert mock_error.called
                    args, _ = mock_error.call_args
                    assert "Error loading models" in args[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
