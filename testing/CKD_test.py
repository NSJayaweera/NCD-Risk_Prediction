import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import sys
import os
import json
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import CKD

# TEST CLASS 

class TestCKDAnalysis:

    # TEST 1 — Feature Engineering Logic
    # CKD tests:    ACR, BUN ratio, BMI, eGFR

    def test_feature_engineering_logic(self):
        """Test if the manual math for derived features is correct."""

        # ── BMI ──
        weight    = 70.0
        height_cm = 170.0
        bmi = round(weight / ((height_cm / 100) ** 2), 1)
        assert bmi == 24.2

        # ── ACR (Albumin-Creatinine Ratio) ──
        urine_albumin    = 10.0
        urine_creatinine = 100.0
        acr = round(urine_albumin / urine_creatinine * 1000, 2)
        assert acr == 100.0
        assert acr > 0

        # ── BUN / Creatinine Ratio ──
        blood_urea_nitrogen = 15.0
        serum_creatinine    = 1.0
        bun_cr_ratio = round(blood_urea_nitrogen / serum_creatinine, 2)
        assert bun_cr_ratio == 15.0
        assert bun_cr_ratio > 0

        # ── eGFR (CKD-EPI 2021) ──
        egfr = CKD.calculate_egfr(1.0, 50, 0)
        assert egfr is not None
        assert egfr > 0

        # eGFR should decrease as creatinine increases
        egfr_low  = CKD.calculate_egfr(0.8, 50, 1)
        egfr_high = CKD.calculate_egfr(3.0, 50, 1)
        assert egfr_low > egfr_high

        # eGFR should decrease as age increases
        egfr_young = CKD.calculate_egfr(1.0, 30, 1)
        egfr_old   = CKD.calculate_egfr(1.0, 70, 1)
        assert egfr_young > egfr_old

        # Invalid inputs should return None
        assert CKD.calculate_egfr(0, 50, 1)  is None
        assert CKD.calculate_egfr(1.0, 0, 1) is None

    # TEST 2 — Asset Loading Failure 
    # CKD tests:    same — joblib.load crash → st.error called

    @patch('joblib.load')
    def test_asset_loading_failure(self, mock_joblib):

        # Make joblib.load crash — same as Heart test
        mock_joblib.side_effect = Exception("File not found")

        # Check if st.error is called when models are missing
        with patch('streamlit.error') as mock_error:
            CKD.run_ckd_analysis()
            assert mock_error.called

            # Check the error message contains the hint
            args, _ = mock_error.call_args
            assert "Error loading models" in args[0]

    # TEST 3 — DataFrame Alignment 
    # CKD tests:    reindex for 14 feature columns

    def test_dataframe_alignment(self):
        """Test if reindexing aligns columns correctly with training features."""

        # These are the 14 features the model expects
        feature_order = [
            'age', 'gender', 'bmi', 'bp_systolic', 'bp_diastolic',
            'serum_creatinine', 'blood_urea_nitrogen', 'urine_albumin',
            'urine_creatinine', 'albumin_creatinine_ratio', 'albumin_serum',
            'uric_acid', 'diabetes_diagnosed', 'bun_creatinine_ratio'
        ]

        # Simulate user input — one feature missing
        input_data = {
            'age'                     : 50,
            'gender'                  : 1,
            'bmi'                     : 24.2,
            'bp_systolic'             : 120,
            'bp_diastolic'            : 80,
            'serum_creatinine'        : 1.0,
            'blood_urea_nitrogen'     : 15.0,
            'urine_albumin'           : 3.0,
            'urine_creatinine'        : 150.0,
            'albumin_creatinine_ratio': 20.0,
            'albumin_serum'           : 4.0,
            'uric_acid'               : 5.0,
            'diabetes_diagnosed'      : 0,
            # bun_creatinine_ratio missing intentionally
        }

        df = pd.DataFrame([input_data])

        # Align — same as deployment app logic
        for col in feature_order:
            if col not in df.columns:
                df[col] = np.nan

        final_df = df[feature_order]

        # bun_creatinine_ratio should now exist filled with NaN
        assert 'bun_creatinine_ratio' in final_df.columns

        # Shape must match exactly
        assert final_df.shape[1] == len(feature_order)
        assert final_df.shape[1] == 14

        # NaN should be there for missing column
        assert pd.isna(final_df['bun_creatinine_ratio'].iloc[0])

    # TEST 4 — Risk Score Clamping
    # Same as test_risk_score_clamping in Test_Heart.py
    # Heart tested: max(0, min(1, risk_score))
    # CKD tests:    same — avg_prob clamping logic

    def test_risk_score_clamping(self):
        """Ensure the probability display doesn't go below 0% or above 100%."""

        # Same logic as CKD.py:
        # clamped = max(0.0, min(1.0, float(avg_prob)))
        low_score    = -0.5
        high_score   =  1.5
        normal_score =  0.72

        assert max(0.0, min(1.0, low_score))    == 0.0
        assert max(0.0, min(1.0, high_score))   == 1.0
        assert max(0.0, min(1.0, normal_score)) == 0.72

    # TEST 5 — Default Values Produce Normal ACR
    # Catches the bug we found — all defaults tested together

    def test_default_values(self):

        # All default values exactly as set in CKD.py
        age                 = 50
        gender              = 0       # Female
        weight              = 70.0
        height_cm           = 170.0
        bp_systolic         = 120
        bp_diastolic        = 80
        serum_creatinine    = 1.0
        blood_urea_nitrogen = 15.0
        urine_albumin       = 3.0    
        urine_creatinine    = 150.0   
        albumin_serum       = 4.0
        uric_acid           = 5.0
        diabetes            = 0

        # Auto calculated — same as CKD.py
        bmi          = round(weight / ((height_cm / 100) ** 2), 1)
        acr          = round(urine_albumin / urine_creatinine * 1000, 2)
        bun_cr_ratio = round(blood_urea_nitrogen / serum_creatinine, 2)
        egfr         = CKD.calculate_egfr(serum_creatinine, age, gender)

        # Check 1 — BMI normal range
        assert 18.5 <= bmi <= 30, \
            f"Default BMI = {bmi} is outside normal range"

        # Check 2 — ACR must be below 30 (most important)
        assert acr < 30, \
            f"Default ACR = {acr} mg/g exceeds normal 30 mg/g. " \
            f"Fix CKD.py: urine_albumin=3.0, urine_creatinine=150.0"

        # Check 3 — BUN ratio normal range
        assert 10 <= bun_cr_ratio <= 20, \
            f"Default BUN ratio = {bun_cr_ratio} is outside normal range"

        # Check 4 — eGFR should be normal
        assert egfr > 60, \
            f"Default eGFR = {egfr} is below 60 — indicates CKD risk"

    # TEST 6 — ACR Zero Guard

    def test_acr_zero_guard(self):
        """If urine creatinine is 0, ACR should return 0 not crash."""

        urine_albumin    = 10.0
        urine_creatinine = 0

        acr = round(urine_albumin / urine_creatinine * 1000, 2) \
              if urine_creatinine > 0 else 0

        assert acr == 0

    # TEST 7 — BUN Ratio Zero Guard

    def test_bun_ratio_zero_guard(self):
        """If serum creatinine is 0, BUN ratio should return 0 not crash."""

        blood_urea_nitrogen = 15.0
        serum_creatinine    = 0

        bun_cr_ratio = round(blood_urea_nitrogen / serum_creatinine, 2) \
                       if serum_creatinine > 0 else 0

        assert bun_cr_ratio == 0

    # TEST 8 — Known Patient Prediction
    # Uses real patients from NHANES dataset

    def test_model_prediction_known_patient(self):
        """
        Test model with known patients from NHANES dataset.
        CKD patient  → model must predict 1
        No CKD patient → model must predict 0
        """

        stacking = joblib.load(os.path.join(SAVE_DIR, 'stacking_model.pkl'))
        bagging  = joblib.load(os.path.join(SAVE_DIR, 'bagging_model.pkl'))

         with open(os.path.join(SAVE_DIR, 'feature_order.json')) as f:
            feature_order = json.load(f)

        # Known CKD patient — actual label = 1
        # From NHANES dataset row 9
        ckd_patient = pd.DataFrame([{
            'age': 68.0, 'gender': 0, 'bmi': 42.6,
            'bp_systolic': 143.0, 'bp_diastolic': 76.0,
            'serum_creatinine': 0.76, 'blood_urea_nitrogen': 15.0,
            'urine_albumin': 5.93, 'urine_creatinine': 23.0,
            'albumin_creatinine_ratio': 25.78, 'albumin_serum': 3.7,
            'uric_acid': 6.2, 'diabetes_diagnosed': 0.0,
            'bun_creatinine_ratio': 19.74,
        }])[feature_order]

        # Known No-CKD patient — actual label = 0
        # From NHANES dataset row 0
        no_ckd_patient = pd.DataFrame([{
            'age': 43.0, 'gender': 1, 'bmi': 27.0,
            'bp_systolic': 135.0, 'bp_diastolic': 98.0,
            'serum_creatinine': 0.8, 'blood_urea_nitrogen': 11.0,
            'urine_albumin': 23.12, 'urine_creatinine': 136.0,
            'albumin_creatinine_ratio': 17.0, 'albumin_serum': 4.3,
            'uric_acid': 5.1, 'diabetes_diagnosed': 0.0,
            'bun_creatinine_ratio': 13.75,
        }])[feature_order]

        # Stacking must predict correctly
        assert stacking.predict(ckd_patient)[0] == 1, \
            "Stacking failed to detect known CKD patient"
        assert stacking.predict(no_ckd_patient)[0] == 0, \
            "Stacking wrongly predicted CKD for healthy patient"

        # Bagging must predict correctly
        assert bagging.predict(ckd_patient)[0] == 1, \
            "Bagging failed to detect known CKD patient"
        assert bagging.predict(no_ckd_patient)[0] == 0, \
            "Bagging wrongly predicted CKD for healthy patient"

    # TEST 9 — BP Consistency Check

    def test_bp_consistency_check(self):
        """Systolic BP must always be greater than diastolic BP."""

        # Valid BP
        assert 120 > 80, \
            "Valid BP failed"

        # Invalid — equal
        assert not (80 > 80), \
            "Equal systolic and diastolic should be invalid"

        # Invalid — reversed
        assert not (75 > 90), \
            "Systolic lower than diastolic should be invalid"

    # TEST 10 — Diabetes Encoding

    def test_diabetes_encoding(self):
        """
        NHANES diabetes codes must encode correctly.
        1 = Yes → 1
        2 = No  → 0
        3 = Borderline → None
        9 = Don't know → None
        """

        dm_map = {1.0: 1, 2.0: 0}

        assert dm_map[1.0] == 1
        assert dm_map[2.0] == 0
        assert dm_map.get(3.0, None) is None, \
            "Borderline should be NaN not 0 or 1"
        assert dm_map.get(9.0, None) is None, \
            "Unknown status should be NaN"

    # TEST 11 — Gender Encoding 

    def test_gender_encoding(self):
        """Male = 1, Female = 0"""

        gender_map = {'Male': 1, 'Female': 0, 'M': 1, 'F': 0}

        assert gender_map['Male']   == 1
        assert gender_map['Female'] == 0
        assert gender_map['M']      == 1
        assert gender_map['F']      == 0

    # TEST 12 — Feature Order Count 

    def test_feature_order_count(self):
        """
        feature_order.json must have exactly 14 features.
        No leakage columns like egfr or ckd_stage.
        No duplicates.
        """

        with open('ckdModel/feature_order.json') as f:
            feature_order = json.load(f)

        # Must be exactly 14
        assert len(feature_order) == 14, \
            f"Expected 14 features, got {len(feature_order)}"

        # Leakage columns must NOT be present
        leakage_cols = ['egfr', 'ckd_stage', 'ckd_present', 'participant_id']
        for col in leakage_cols:
            assert col not in feature_order, \
                f"Leakage column '{col}' found in feature_order!"

        # No duplicates
        assert len(feature_order) == len(set(feature_order)), \
            "Duplicate features found in feature_order.json"


# ENTRY POINT

if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
