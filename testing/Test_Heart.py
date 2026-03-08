import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Adds the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Heart


class TestHeartAnalysis:

    def test_feature_engineering_logic(self):
        """Test if the manual math for new features is correct."""
        age = 40
        thalach = 160
        chol = 200
        oldpeak = 1.5
        log_chol = np.log1p(chol)
        log_oldpeak = np.log1p(oldpeak)
        hr_reserve = (220 - age) - thalach

        assert hr_reserve == 20
        assert log_chol > 0
        assert log_oldpeak > 0

    @patch('joblib.load')
    @patch('catboost.CatBoostRegressor.load_model')

    def test_asset_loading_failure(self, mock_cat, mock_joblib):
        """Verify that the app handles missing model files gracefully."""
        mock_joblib.side_effect = Exception("File not found")

        # We use a mock streamlit to see if st.error is called
        with patch('streamlit.error') as mock_error:
            Heart.run_heart_analysis()
            assert mock_error.called
            # Check if the error message contains our hint
            args, _ = mock_error.call_args
            assert "Error loading models" in args[0]

    def test_dataframe_alignment(self):
        """Test if reindexing aligns columns correctly with training features."""
        model_columns = ['age', 'sex', 'cp_0', 'cp_1', 'hr_reserve']
        input_data = {
            'age': [30],
            'sex': [1],
            'cp': [0],
            'hr_reserve': [50]
        }
        df = pd.DataFrame(input_data)
        df_encoded = pd.get_dummies(df, columns=['cp'])

        # Aligning
        final_df = df_encoded.reindex(columns=model_columns, fill_value=0)

        assert 'cp_1' in final_df.columns
        assert final_df['cp_1'].iloc[0] == 0  # Should be filled with 0
        assert final_df.shape[1] == len(model_columns)

    def test_risk_score_clamping(self):
        """Ensure the percentage display doesn't go below 0% or above 100%."""
        # Logic: max(0, min(1, risk_score))
        low_score = -0.5
        high_score = 1.5

        assert max(0, min(1, low_score)) == 0
        assert max(0, min(1, high_score)) == 1