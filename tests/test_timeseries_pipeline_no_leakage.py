#!/usr/bin/env python3
"""
Test that TimeSeriesSplit with Pipeline prevents data leakage
"""

import numpy as np
import pandas as pd
import unittest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class TestTimeSeriesPipelineNoLeakage(unittest.TestCase):
    """Test that TimeSeriesSplit with Pipeline prevents data leakage"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Generate synthetic time series data
        n_samples = 1000
        n_features = 10
        
        # Create features with some temporal structure
        X = np.random.randn(n_samples, n_features)
        
        # Add some temporal autocorrelation to features
        for i in range(1, n_samples):
            X[i] += 0.1 * X[i-1]
            
        # Create targets with some predictability
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        
        self.X = X
        self.y = y
        self.n_samples = n_samples
        
    def test_timeseries_split_no_shuffle(self):
        """Test that TimeSeriesSplit doesn't shuffle data"""
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Verify splits are sequential
        splits = list(tscv.split(self.X))
        
        for i, (train_idx, val_idx) in enumerate(splits):
            # Train indices should be sequential and start from 0
            self.assertEqual(train_idx[0], 0)
            self.assertEqual(train_idx[-1], val_idx[0] - 1)
            
            # Validation indices should be sequential after training
            self.assertEqual(val_idx[0], train_idx[-1] + 1)
            
            # Later splits should have more training data
            if i > 0:
                self.assertGreater(len(train_idx), len(splits[i-1][0]))
                
    def test_pipeline_fit_transform_no_leakage(self):
        """Test that Pipeline fit_transform doesn't leak between folds"""
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Create pipeline with scaler and classifier
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Store statistics from each fold
        fold_stats = []
        
        for train_idx, val_idx in tscv.split(self.X):
            # Fit pipeline on training data
            X_train = self.X[train_idx]
            y_train = self.y[train_idx]
            
            # Fit the pipeline
            pipeline.fit(X_train, y_train)
            
            # Get transformed training data
            X_train_transformed = pipeline.named_steps['scaler'].transform(X_train)
            
            # Get transformed validation data (using fitted scaler)
            X_val_transformed = pipeline.named_steps['scaler'].transform(self.X[val_idx])
            
            # Calculate statistics
            train_mean = np.mean(X_train_transformed, axis=0)
            train_std = np.std(X_train_transformed, axis=0)
            val_mean = np.mean(X_val_transformed, axis=0)
            val_std = np.std(X_val_transformed, axis=0)
            
            fold_stats.append({
                'train_mean': train_mean,
                'train_std': train_std,
                'val_mean': val_mean,
                'val_std': val_std
            })
            
        # Verify that validation statistics are different from training statistics
        # This ensures the scaler is not fitted on the full dataset
        for i, stats in enumerate(fold_stats):
            # Training data should be approximately standardized (mean≈0, std≈1)
            np.testing.assert_array_almost_equal(stats['train_mean'], np.zeros_like(stats['train_mean']), decimal=1)
            np.testing.assert_array_almost_equal(stats['train_std'], np.ones_like(stats['train_std']), decimal=1)
            
            # Validation data should NOT be standardized (different mean/std)
            # This is the key test - if there was leakage, validation would also be standardized
            mean_diff = np.mean(np.abs(stats['val_mean'] - stats['train_mean']))
            std_diff = np.mean(np.abs(stats['val_std'] - stats['train_std']))
            
            self.assertGreater(mean_diff, 0.01)  # Should be different
            self.assertGreater(std_diff, 0.01)   # Should be different
            
    def test_manual_oof_generation_no_leakage(self):
        """Test manual OOF generation doesn't leak data"""
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Generate OOF predictions manually
        oof_predictions = np.zeros(len(self.y))
        oof_probabilities = np.zeros(len(self.y))
        
        for train_idx, val_idx in tscv.split(self.X):
            # Create a copy of the pipeline for this fold
            from copy import deepcopy
            fold_pipeline = deepcopy(pipeline)
            
            # Fit on training data only
            X_train = self.X[train_idx]
            y_train = self.y[train_idx]
            fold_pipeline.fit(X_train, y_train)
            
            # Predict on validation data
            val_pred = fold_pipeline.predict(self.X[val_idx])
            val_proba = fold_pipeline.predict_proba(self.X[val_idx])[:, 1]
            
            # Store predictions
            oof_predictions[val_idx] = val_pred
            oof_probabilities[val_idx] = val_proba
            
        # Verify OOF predictions are not all the same (indicating proper fitting)
        unique_predictions = np.unique(oof_predictions)
        self.assertGreater(len(unique_predictions), 1)
        
        # Verify OOF probabilities have reasonable range
        self.assertGreater(np.std(oof_probabilities), 0.01)
        
        # Verify no perfect correlation with targets (which would indicate leakage)
        correlation = np.corrcoef(oof_probabilities, self.y)[0, 1]
        self.assertLess(abs(correlation), 0.99)  # Should not be perfect
        
    def test_pipeline_consistency_across_folds(self):
        """Test that Pipeline behavior is consistent across folds"""
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        fold_metrics = []
        
        for train_idx, val_idx in tscv.split(self.X):
            # Fit pipeline
            X_train = self.X[train_idx]
            y_train = self.y[train_idx]
            pipeline.fit(X_train, y_train)
            
            # Predict on validation
            y_val_pred = pipeline.predict(self.X[val_idx])
            y_val_true = self.y[val_idx]
            
            # Calculate accuracy
            accuracy = accuracy_score(y_val_true, y_val_pred)
            fold_metrics.append(accuracy)
            
        # Verify that performance varies across folds (indicating proper temporal validation)
        # If there was leakage, all folds would have similar performance
        fold_std = np.std(fold_metrics)
        self.assertGreater(fold_std, 0.01)  # Should have some variation
        
        # Verify reasonable accuracy range
        for acc in fold_metrics:
            self.assertGreater(acc, 0.4)  # Should be better than random
            self.assertLess(acc, 1.0)     # Should not be perfect
            
    def test_scaler_statistics_evolution(self):
        """Test that scaler statistics evolve over time (no global fitting)"""
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Track how scaler statistics change across folds
        scaler_means = []
        scaler_stds = []
        
        for train_idx, val_idx in tscv.split(self.X):
            # Fit scaler on training data only
            scaler = StandardScaler()
            X_train = self.X[train_idx]
            scaler.fit(X_train)
            
            # Store fitted parameters
            scaler_means.append(scaler.mean_.copy())
            scaler_stds.append(scaler.scale_.copy())
            
        # Verify that scaler parameters change across folds
        # This ensures each fold is fitted independently
        if len(scaler_means) > 1:
            # Calculate variation in means across folds
            mean_variations = []
            for i in range(1, len(scaler_means)):
                variation = np.mean(np.abs(scaler_means[i] - scaler_means[i-1]))
                mean_variations.append(variation)
                
            # Should have some variation (indicating independent fitting)
            avg_variation = np.mean(mean_variations)
            self.assertGreater(avg_variation, 0.001)
            
    def test_no_global_data_access(self):
        """Test that Pipeline doesn't access global data during fitting"""
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Store original data
        X_original = self.X.copy()
        y_original = self.y.copy()
        
        for train_idx, val_idx in tscv.split(self.X):
            # Fit on training data only
            X_train = self.X[train_idx]
            y_train = self.y[train_idx]
            
            # Fit pipeline
            pipeline.fit(X_train, y_train)
            
            # Verify that original data hasn't been modified
            np.testing.assert_array_equal(self.X, X_original)
            np.testing.assert_array_equal(self.y, y_original)
            
            # Verify that pipeline only knows about training data
            # This is harder to test directly, but we can check that
            # the scaler parameters are based on training data only
            scaler = pipeline.named_steps['scaler']
            
            # Calculate what the mean should be based on training data
            expected_mean = np.mean(X_train, axis=0)
            np.testing.assert_array_almost_equal(scaler.mean_, expected_mean, decimal=10)

if __name__ == '__main__':
    unittest.main()
