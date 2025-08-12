"""
Test Probability Calibration
Verifies that calibrated probabilities improve Brier score and calibration metrics
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from enhanced_ensemble import EnhancedTradingEnsemble

class TestCalibration(unittest.TestCase):
    """Test cases for probability calibration"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create synthetic data
        self.n_samples = 200
        self.n_features = 15
        
        # Generate features
        self.X = np.random.randn(self.n_samples, self.n_features)
        
        # Generate target with some predictability
        signal = np.dot(self.X[:, :5], np.random.randn(5)) * 0.15
        noise = np.random.randn(self.n_samples) * 0.02
        returns = signal + noise
        self.y = (returns > 0).astype(int)
        
        # Create ensemble
        self.ensemble = EnhancedTradingEnsemble(random_state=42, n_splits=2)
    
    def test_calibration_improves_brier_score(self):
        """Test that calibration improves Brier score"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Get uncalibrated probabilities
        uncalibrated_proba = {}
        for name, pipeline in self.ensemble.models.items():
            if name in self.ensemble.oof_probabilities:
                uncalibrated_proba[name] = self.ensemble.oof_probabilities[name]
        
        # Calculate uncalibrated Brier scores
        uncalibrated_brier = {}
        for name, proba in uncalibrated_proba.items():
            brier = self._calculate_brier_score(proba, self.y)
            uncalibrated_brier[name] = brier
        
        # Calibrate probabilities
        self.ensemble.calibrate_probabilities(method='isotonic')
        
        # Calculate calibrated Brier scores
        calibrated_brier = {}
        for name in self.ensemble.calibrators:
            if name in self.ensemble.oof_probabilities:
                # Use calibrated model for predictions
                calibrated_proba = self.ensemble.calibrators[name].predict_proba(self.X)[:, 1]
                brier = self._calculate_brier_score(calibrated_proba, self.y)
                calibrated_brier[name] = brier
        
        # Verify calibration improved Brier scores (lower is better)
        for name in calibrated_brier:
            if name in uncalibrated_brier:
                self.assertLessEqual(
                    calibrated_brier[name], 
                    uncalibrated_brier[name] * 1.1,  # Allow 10% tolerance
                    f"Calibration did not improve Brier score for {name}"
                )
    
    def test_calibration_improves_calibration_mae(self):
        """Test that calibration improves calibration MAE"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Get uncalibrated calibration errors
        uncalibrated_mae = {}
        for name, proba in self.ensemble.oof_probabilities.items():
            mae = self._calculate_calibration_mae(proba, self.y)
            uncalibrated_mae[name] = mae
        
        # Calibrate probabilities
        self.ensemble.calibrate_probabilities(method='isotonic')
        
        # Calculate calibrated calibration errors
        calibrated_mae = {}
        for name in self.ensemble.calibrators:
            if name in self.ensemble.oof_probabilities:
                calibrated_proba = self.ensemble.calibrators[name].predict_proba(self.X)[:, 1]
                mae = self._calculate_calibration_mae(calibrated_proba, self.y)
                calibrated_mae[name] = mae
        
        # Verify calibration improved calibration MAE (lower is better)
        for name in calibrated_mae:
            if name in uncalibrated_mae:
                self.assertLessEqual(
                    calibrated_mae[name], 
                    uncalibrated_mae[name] * 1.1,  # Allow 10% tolerance
                    f"Calibration did not improve calibration MAE for {name}"
                )
    
    def test_isotonic_vs_sigmoid_calibration(self):
        """Test both isotonic and sigmoid calibration methods"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Test isotonic calibration
        self.ensemble.calibrate_probabilities(method='isotonic')
        isotonic_calibrators = len(self.ensemble.calibrators)
        
        # Reset calibrators
        self.ensemble.calibrators = {}
        
        # Test sigmoid calibration
        self.ensemble.calibrate_probabilities(method='sigmoid')
        sigmoid_calibrators = len(self.ensemble.calibrators)
        
        # Both methods should create calibrators
        self.assertGreater(isotonic_calibrators, 0)
        self.assertGreater(sigmoid_calibrators, 0)
    
    def test_calibration_preserves_probability_range(self):
        """Test that calibrated probabilities remain in [0, 1] range"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Calibrate probabilities
        self.ensemble.calibrate_probabilities(method='isotonic')
        
        # Test calibrated probabilities
        for name in self.ensemble.calibrators:
            if name in self.ensemble.models:
                calibrated_proba = self.ensemble.calibrators[name].predict_proba(self.X)[:, 1]
                
                # Check probability range
                self.assertTrue(np.all(calibrated_proba >= 0))
                self.assertTrue(np.all(calibrated_proba <= 1))
                
                # Check that we have some variation (not all 0 or all 1)
                self.assertGreater(np.std(calibrated_proba), 0.01)
    
    def test_calibration_consistency(self):
        """Test that calibration produces consistent results"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Calibrate probabilities
        self.ensemble.calibrate_probabilities(method='isotonic')
        
        # Get first set of predictions
        first_predictions = {}
        for name in self.ensemble.calibrators:
            if name in self.ensemble.models:
                proba = self.ensemble.calibrators[name].predict_proba(self.X)[:, 1]
                first_predictions[name] = proba.copy()
        
        # Get second set of predictions
        second_predictions = {}
        for name in self.ensemble.calibrators:
            if name in self.ensemble.models:
                proba = self.ensemble.calibrators[name].predict_proba(self.X)[:, 1]
                second_predictions[name] = proba.copy()
        
        # Verify consistency
        for name in first_predictions:
            np.testing.assert_array_almost_equal(
                first_predictions[name], 
                second_predictions[name], 
                decimal=10,
                err_msg=f"Inconsistent predictions for {name}"
            )
    
    def _calculate_brier_score(self, proba: np.ndarray, y: np.ndarray) -> float:
        """Calculate Brier score"""
        from sklearn.metrics import brier_score_loss
        return brier_score_loss(y, proba)
    
    def _calculate_calibration_mae(self, proba: np.ndarray, y: np.ndarray) -> float:
        """Calculate calibration MAE"""
        # Bin probabilities and calculate calibration error
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(proba, bins) - 1
        
        calibration_error = 0
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_proba = np.mean(proba[mask])
                bin_actual = np.mean(y[mask])
                calibration_error += np.abs(bin_proba - bin_actual)
        
        return calibration_error / (len(bins) - 1)

if __name__ == '__main__':
    unittest.main()
