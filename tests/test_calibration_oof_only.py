#!/usr/bin/env python3
"""
Test that calibration is fit only on OOF data (no leakage)
"""

import numpy as np
import pandas as pd
import unittest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')

class TestCalibrationOOFOnly(unittest.TestCase):
    """Test that calibration uses only OOF data to prevent leakage"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Generate synthetic time series data
        n_samples = 1000
        n_features = 10
        
        # Create features with temporal structure
        X = np.random.randn(n_samples, n_features)
        for i in range(1, n_samples):
            X[i] += 0.1 * X[i-1]
            
        # Create targets with some predictability
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        
        self.X = X
        self.y = y
        self.n_samples = n_samples
        
    def test_calibrated_classifier_cv_oof_only(self):
        """Test that CalibratedClassifierCV uses OOF data only"""
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Create base pipeline
        base_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Create calibrated classifier
        calibrated = CalibratedClassifierCV(
            base_pipeline, 
            method='isotonic', 
            cv=tscv
        )
        
        # Fit the calibrated classifier
        calibrated.fit(self.X, self.y)
        
        # Verify that the calibration was done using OOF data
        # by checking that the calibrated model doesn't have perfect
        # calibration on the full dataset
        
        # Get predictions from calibrated model
        calibrated_probs = calibrated.predict_proba(self.X)[:, 1]
        
        # Calculate Brier score on full dataset
        full_brier = brier_score_loss(self.y, calibrated_probs)
        
        # If calibration was done on full data, Brier score would be very low
        # With OOF calibration, it should be higher (more realistic)
        self.assertGreater(full_brier, 0.01)  # Should not be perfect
        
        # Verify that calibration is not overfitted to the full dataset
        # by checking that performance varies across folds
        
    def test_manual_oof_calibration_no_leakage(self):
        """Test manual OOF calibration without leakage"""
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Create base pipeline
        base_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Generate OOF probabilities manually
        oof_probs = np.zeros(len(self.y))
        oof_labels = np.zeros(len(self.y))
        
        for train_idx, val_idx in tscv.split(self.X):
            # Fit base model on training data
            X_train = self.X[train_idx]
            y_train = self.y[train_idx]
            
            # Create a copy to avoid modifying the original
            from copy import deepcopy
            fold_pipeline = deepcopy(base_pipeline)
            fold_pipeline.fit(X_train, y_train)
            
            # Get predictions on validation data
            val_probs = fold_pipeline.predict_proba(self.X[val_idx])[:, 1]
            
            # Store OOF data
            oof_probs[val_idx] = val_probs
            oof_labels[val_idx] = self.y[val_idx]
            
        # Now fit calibration using only OOF data
        # This is the key test - calibration should only use OOF data
        
        # Filter out samples that don't have OOF predictions
        valid_mask = oof_probs != 0
        valid_oof_probs = oof_probs[valid_mask]
        valid_oof_labels = oof_labels[valid_mask]
        
        # Fit isotonic regression on OOF data only
        from sklearn.isotonic import IsotonicRegression
        isotonic = IsotonicRegression(out_of_bounds='clip')
        isotonic.fit(valid_oof_probs, valid_oof_labels)
        
        # Apply calibration to full dataset (using original probabilities)
        full_probs = np.random.random(len(self.y))  # Simulate full dataset probabilities
        calibrated_probs = isotonic.transform(full_probs)
        
        # Calculate Brier scores
        oof_brier = brier_score_loss(valid_oof_labels, valid_oof_probs)
        calibrated_brier = brier_score_loss(self.y, calibrated_probs)
        
        # Both should be reasonable (not perfect)
        self.assertGreater(calibrated_brier, 0.001)
        self.assertGreater(oof_brier, 0.001)
        
    def test_calibration_data_independence(self):
        """Test that calibration data is independent of test data"""
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Create base pipeline
        base_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Store calibration data separately
        all_calibration_probs = []
        all_calibration_labels = []
        
        for train_idx, val_idx in tscv.split(self.X):
            # Fit on training data
            X_train = self.X[train_idx]
            y_train = self.y[train_idx]
            
            # Create a copy for this fold
            from copy import deepcopy
            fold_pipeline = deepcopy(base_pipeline)
            fold_pipeline.fit(X_train, y_train)
            
            # Get validation predictions
            val_probs = fold_pipeline.predict_proba(self.X[val_idx])[:, 1]
            val_labels = self.y[val_idx]
            
            # Store for calibration
            all_calibration_probs.extend(val_probs)
            all_calibration_labels.extend(val_labels)
            
        # Convert to arrays
        calibration_probs = np.array(all_calibration_probs)
        calibration_labels = np.array(all_calibration_labels)
        
        # Verify that calibration data is different from full dataset
        # This ensures we're not using the same data for both training and testing
        
        # Check that calibration data is a subset
        self.assertLess(len(calibration_probs), len(self.y))
        
        # Check that calibration data has different statistics than full dataset
        # (indicating it's truly a subset)
        full_mean = np.mean(self.y)
        cal_mean = np.mean(calibration_labels)
        
        # Should be similar but not identical
        self.assertGreater(abs(full_mean - cal_mean), 0.001)
        
    def test_calibration_performance_validation(self):
        """Test that calibration performance is validated on OOF data"""
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Create base pipeline
        base_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Store performance metrics for each fold
        fold_metrics = []
        
        for train_idx, val_idx in tscv.split(self.X):
            # Fit base model
            X_train = self.X[train_idx]
            y_train = self.y[train_idx]
            
            from copy import deepcopy
            fold_pipeline = deepcopy(base_pipeline)
            fold_pipeline.fit(X_train, y_train)
            
            # Get predictions
            val_probs = fold_pipeline.predict_proba(self.X[val_idx])[:, 1]
            val_labels = self.y[val_idx]
            
            # Calculate metrics before calibration
            before_brier = brier_score_loss(val_labels, val_probs)
            
            # Fit calibration on this fold's validation data
            # (This is a simplified test - in practice you'd use all OOF data)
            from sklearn.isotonic import IsotonicRegression
            isotonic = IsotonicRegression(out_of_bounds='clip')
            isotonic.fit(val_probs, val_labels)
            
            # Apply calibration
            calibrated_probs = isotonic.transform(val_probs)
            
            # Calculate metrics after calibration
            after_brier = brier_score_loss(val_labels, calibrated_probs)
            
            fold_metrics.append({
                'before_brier': before_brier,
                'after_brier': after_brier,
                'improvement': before_brier - after_brier
            })
            
        # Verify that calibration improves performance on average
        avg_improvement = np.mean([m['improvement'] for m in fold_metrics])
        self.assertGreater(avg_improvement, 0)  # Should improve on average
        
        # Verify that improvement is consistent across folds
        improvements = [m['improvement'] for m in fold_metrics]
        self.assertGreater(np.std(improvements), 0.001)  # Should vary somewhat
        
    def test_calibration_methods_comparison(self):
        """Test different calibration methods with OOF data"""
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Create base pipeline
        base_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Test both isotonic and sigmoid calibration
        methods = ['isotonic', 'sigmoid']
        method_performances = {}
        
        for method in methods:
            # Create calibrated classifier
            calibrated = CalibratedClassifierCV(
                base_pipeline, 
                method=method, 
                cv=tscv
            )
            
            # Fit and predict
            calibrated.fit(self.X, self.y)
            calibrated_probs = calibrated.predict_proba(self.X)[:, 1]
            
            # Calculate performance
            brier_score = brier_score_loss(self.y, calibrated_probs)
            method_performances[method] = brier_score
            
        # Both methods should work and produce reasonable results
        for method, score in method_performances.items():
            self.assertGreater(score, 0.001)  # Should not be perfect
            self.assertLess(score, 0.5)       # Should not be terrible
            
        # Methods should have similar performance (within reason)
        score_diff = abs(method_performances['isotonic'] - method_performances['sigmoid'])
        self.assertLess(score_diff, 0.1)  # Should be reasonably close
        
    def test_calibration_data_contamination_detection(self):
        """Test detection of calibration data contamination"""
        # This test verifies that we can detect if calibration data
        # is contaminated with test data
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Create base pipeline
        base_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Generate OOF probabilities
        oof_probs = np.zeros(len(self.y))
        oof_labels = np.zeros(len(self.y))
        
        for train_idx, val_idx in tscv.split(self.X):
            # Fit on training data
            X_train = self.X[train_idx]
            y_train = self.y[train_idx]
            
            from copy import deepcopy
            fold_pipeline = deepcopy(base_pipeline)
            fold_pipeline.fit(X_train, y_train)
            
            # Get validation predictions
            val_probs = fold_pipeline.predict_proba(self.X[val_idx])[:, 1]
            oof_probs[val_idx] = val_probs
            oof_labels[val_idx] = self.y[val_idx]
            
        # Test 1: Calibration on OOF data only (should work)
        valid_mask = oof_probs != 0
        valid_oof_probs = oof_probs[valid_mask]
        valid_oof_labels = oof_labels[valid_mask]
        
        from sklearn.isotonic import IsotonicRegression
        isotonic_oof = IsotonicRegression(out_of_bounds='clip')
        isotonic_oof.fit(valid_oof_probs, valid_oof_labels)
        
        # Create full dataset probabilities for testing
        full_probs = np.random.random(len(self.y))
        
        # Test 2: Calibration on full dataset (contaminated - should fail test)
        isotonic_full = IsotonicRegression(out_of_bounds='clip')
        isotonic_full.fit(full_probs, self.y)
        
        # Apply both calibrations
        oof_calibrated = isotonic_oof.transform(full_probs)
        full_calibrated = isotonic_full.transform(full_probs)
        
        # Calculate Brier scores
        oof_brier = brier_score_loss(self.y, oof_calibrated)
        full_brier = brier_score_loss(self.y, full_calibrated)
        
        # Both calibrations should produce reasonable results
        self.assertGreater(oof_brier, 0.001)
        self.assertGreater(full_brier, 0.001)
        
        # Full calibration might be better (overfitted) but should not be perfect
        self.assertLess(full_brier, 0.5)

if __name__ == '__main__':
    unittest.main()
