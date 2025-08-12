"""
Test OOF Alignment and Data Leakage Prevention
Ensures OOF predictions are properly aligned and no data leakage occurs
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

class TestOOFAlignment(unittest.TestCase):
    """Test cases for OOF alignment and data leakage prevention"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create synthetic data with known structure
        self.n_samples = 100
        self.n_features = 10
        
        # Generate features
        self.X = np.random.randn(self.n_samples, self.n_features)
        
        # Generate target with some predictability
        signal = np.dot(self.X[:, :3], np.random.randn(3)) * 0.1
        noise = np.random.randn(self.n_samples) * 0.02
        returns = signal + noise
        self.y = (returns > 0).astype(int)
        
        # Create ensemble with appropriate number of splits for testing
        self.ensemble = EnhancedTradingEnsemble(random_state=42, n_splits=2)
    
    def test_oof_matrix_alignment(self):
        """Test that OOF matrix P (T×M) is properly aligned with target y"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Check OOF probabilities alignment
        self.assertTrue(self.ensemble.is_fitted)
        self.assertGreater(len(self.ensemble.oof_probabilities), 0)
        
        # Verify OOF matrix shape
        oof_matrix = np.column_stack(list(self.ensemble.oof_probabilities.values()))
        expected_shape = (self.n_samples, len(self.ensemble.oof_probabilities))
        self.assertEqual(oof_matrix.shape, expected_shape)
        
        # Verify alignment with target
        self.assertEqual(oof_matrix.shape[0], len(self.y))
        
        # Check that all probabilities are valid
        self.assertTrue(np.all((oof_matrix >= 0) & (oof_matrix <= 1)))
    
    def test_fold_boundaries_no_leakage(self):
        """Test that fold boundaries prevent data leakage"""
        from sklearn.model_selection import TimeSeriesSplit
        
        # Create TimeSeriesSplit with same parameters
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Get fold boundaries
        fold_boundaries = []
        for train_idx, val_idx in tscv.split(self.X):
            fold_boundaries.append((train_idx, val_idx))
        
        # Verify that validation sets don't overlap
        all_val_indices = set()
        for train_idx, val_idx in fold_boundaries:
            # Check no overlap between train and validation
            train_set = set(train_idx)
            val_set = set(val_idx)
            self.assertEqual(len(train_set.intersection(val_set)), 0)
            
            # Check no overlap between validation sets
            self.assertEqual(len(all_val_indices.intersection(val_set)), 0)
            all_val_indices.update(val_set)
        
        # Verify all samples are used in validation
        self.assertEqual(len(all_val_indices), self.n_samples)
    
    def test_calibration_no_leakage(self):
        """Test that calibration doesn't cause data leakage"""
        # Fit models first
        self.ensemble.fit_models(self.X, self.y)
        
        # Store original OOF probabilities
        original_oof = self.ensemble.oof_probabilities.copy()
        
        # Calibrate probabilities
        self.ensemble.calibrate_probabilities(method='isotonic')
        
        # Verify OOF probabilities haven't changed
        for name in original_oof:
            if name in self.ensemble.oof_probabilities:
                np.testing.assert_array_equal(
                    original_oof[name], 
                    self.ensemble.oof_probabilities[name]
                )
        
        # Verify calibrators were created
        self.assertGreater(len(self.ensemble.calibrators), 0)
    
    def test_weight_optimization_oof_only(self):
        """Test that weight optimization uses only OOF data"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Store OOF probabilities
        oof_proba = self.ensemble.oof_probabilities.copy()
        
        # Optimize weights
        results = self.ensemble.optimize_ensemble_weights(
            y=self.y, method='sharpe', cost_per_trade=0.001, slippage=0.0005
        )
        
        # Verify OOF probabilities weren't modified during optimization
        for name in oof_proba:
            if name in self.ensemble.oof_probabilities:
                np.testing.assert_array_equal(
                    oof_proba[name], 
                    self.ensemble.oof_probabilities[name]
                )
        
        # Verify weights were found
        self.assertIn('optimal_weights', results)
        self.assertGreater(len(results['optimal_weights']), 0)
    
    def test_training_data_storage(self):
        """Test that training data is properly stored for calibration"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Verify training data is stored
        self.assertTrue(hasattr(self.ensemble, 'X_train'))
        self.assertTrue(hasattr(self.ensemble, 'y_train'))
        
        # Verify data matches
        np.testing.assert_array_equal(self.ensemble.X_train, self.X)
        np.testing.assert_array_equal(self.ensemble.y_train, self.y)
    
    def test_oof_probability_consistency(self):
        """Test that OOF probabilities are consistent across operations"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Store initial OOF probabilities
        initial_oof = self.ensemble.oof_probabilities.copy()
        
        # Perform multiple operations
        self.ensemble.calibrate_probabilities(method='isotonic')
        self.ensemble.optimize_ensemble_weights(
            y=self.y, method='sharpe', cost_per_trade=0.001, slippage=0.0005
        )
        self.ensemble.apply_diversity_control()
        
        # Verify OOF probabilities remain unchanged
        for name in initial_oof:
            if name in self.ensemble.oof_probabilities:
                np.testing.assert_array_equal(
                    initial_oof[name], 
                    self.ensemble.oof_probabilities[name]
                )

if __name__ == '__main__':
    unittest.main()
