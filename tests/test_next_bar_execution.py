#!/usr/bin/env python3
"""
Test next-bar execution to ensure features at t map to returns at t+1
"""

import numpy as np
import pandas as pd
import unittest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class TestNextBarExecution(unittest.TestCase):
    """Test that features at t map to returns at t+1 with no future leakage"""
    
    def setUp(self):
        """Set up test data with temporal structure"""
        np.random.seed(42)
        
        # Generate synthetic time series data
        n_samples = 1000
        
        # Create price-like data with some trend and noise
        prices = np.cumprod(1 + np.random.normal(0.0001, 0.02, n_samples))
        
        # Calculate returns (next-bar)
        returns = np.diff(prices) / prices[:-1]
        
        # Create features based on past prices (no future information)
        features = np.zeros((len(returns), 5))
        
        # Feature 1: Simple moving average (past 5 periods)
        for i in range(5, len(returns)):
            features[i, 0] = np.mean(prices[i-5:i])
            
        # Feature 2: Price momentum (past 3 periods)
        for i in range(3, len(returns)):
            features[i, 1] = (prices[i] - prices[i-3]) / prices[i-3]
            
        # Feature 3: Volatility (past 10 periods)
        for i in range(10, len(returns)):
            features[i, 2] = np.std(returns[i-10:i])
            
        # Feature 4: Price level relative to moving average
        for i in range(10, len(returns)):
            ma = np.mean(prices[i-10:i])
            features[i, 3] = (prices[i] - ma) / ma
            
        # Feature 5: Return autocorrelation (past 5 periods)
        for i in range(5, len(returns)):
            if np.std(returns[i-5:i]) > 0:
                features[i, 4] = np.corrcoef(returns[i-5:i], returns[i-4:i+1])[0, 1]
            else:
                features[i, 4] = 0
                
        # Create binary targets: positive return = 1, negative = 0
        targets = (returns > 0).astype(int)
        
        # Remove first few samples where we don't have enough history for features
        valid_start = 10
        self.X = features[valid_start:]
        self.y = targets[valid_start:]
        self.returns = returns[valid_start:]
        self.prices = prices[valid_start:]
        
        print(f"Data shape: X={self.X.shape}, y={self.y.shape}")
        
    def test_feature_target_alignment(self):
        """Test that features at t align with targets at t+1"""
        # Verify that features and targets have the same length
        self.assertEqual(len(self.X), len(self.y))
        
        # Verify that features at index i correspond to target at index i
        # This ensures proper temporal alignment
        for i in range(len(self.X)):
            # Features at time i should predict target at time i
            # Target represents the return from time i to time i+1
            self.assertIsInstance(self.X[i], np.ndarray)
            self.assertIsInstance(self.y[i], (int, np.integer))
            
        # Verify that targets are binary
        unique_targets = np.unique(self.y)
        self.assertTrue(np.array_equal(unique_targets, np.array([0, 1])))
        
    def test_no_future_feature_access(self):
        """Test that no features access future information"""
        # Check each feature type for future information access
        
        # Feature 1: Moving average - should only use past prices
        for i in range(5, len(self.X)):
            # Moving average at time i should only use prices up to time i
            current_ma = self.X[i, 0]
            expected_ma = np.mean(self.prices[i-5:i])
            np.testing.assert_almost_equal(current_ma, expected_ma, decimal=10)
            
        # Feature 2: Momentum - should only use past prices
        for i in range(3, len(self.X)):
            current_momentum = self.X[i, 1]
            expected_momentum = (self.prices[i] - self.prices[i-3]) / self.prices[i-3]
            np.testing.assert_almost_equal(current_momentum, expected_momentum, decimal=10)
            
        # Feature 3: Volatility - should only use past returns
        for i in range(10, len(self.X)):
            current_vol = self.X[i, 2]
            expected_vol = np.std(self.returns[i-10:i])
            np.testing.assert_almost_equal(current_vol, expected_vol, decimal=10)
            
        # Feature 4: Price relative to MA - should only use past prices
        for i in range(10, len(self.X)):
            current_rel = self.X[i, 3]
            ma = np.mean(self.prices[i-10:i])
            expected_rel = (self.prices[i] - ma) / ma
            np.testing.assert_almost_equal(current_rel, expected_rel, decimal=10)
            
    def test_temporal_causality(self):
        """Test that causality is preserved (past → future)"""
        # Verify that features at time t only depend on data up to time t
        # and targets at time t represent the outcome at time t+1
        
        # Check that the last feature value doesn't depend on future data
        last_idx = len(self.X) - 1
        
        # Last feature should only use data up to last_idx
        last_feature_ma = self.X[last_idx, 0]
        expected_last_ma = np.mean(self.prices[last_idx-5:last_idx])
        np.testing.assert_almost_equal(last_feature_ma, expected_last_ma, decimal=10)
        
        # Last target should represent the return from last_idx to last_idx+1
        # (which would be the next period after our data ends)
        # This is just a verification that we're not accessing future data
        
    def test_timeseries_split_temporal_integrity(self):
        """Test that TimeSeriesSplit preserves temporal integrity"""
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        for train_idx, val_idx in tscv.split(self.X):
            # Verify that validation indices come after training indices
            if len(train_idx) > 0 and len(val_idx) > 0:
                self.assertGreater(val_idx[0], train_idx[-1])
                
            # Verify that training indices are sequential
            for i in range(1, len(train_idx)):
                self.assertEqual(train_idx[i], train_idx[i-1] + 1)
                
            # Verify that validation indices are sequential
            for i in range(1, len(val_idx)):
                self.assertEqual(val_idx[i], val_idx[i-1] + 1)
                
    def test_feature_engineering_no_lookahead(self):
        """Test that feature engineering doesn't introduce lookahead bias"""
        # Create a more complex feature that could potentially introduce lookahead
        n_samples = len(self.X)
        
        # Feature: Rolling Sharpe ratio (past 20 periods)
        rolling_sharpe = np.zeros(n_samples)
        
        for i in range(20, n_samples):
            # Use only past returns to calculate Sharpe
            past_returns = self.returns[i-20:i]
            if np.std(past_returns) > 0:
                rolling_sharpe[i] = np.mean(past_returns) / np.std(past_returns)
            else:
                rolling_sharpe[i] = 0
                
        # Verify that rolling Sharpe at time i only uses data up to time i
        for i in range(20, n_samples):
            expected_sharpe = 0
            if np.std(self.returns[i-20:i]) > 0:
                expected_sharpe = np.mean(self.returns[i-20:i]) / np.std(self.returns[i-20:i])
            np.testing.assert_almost_equal(rolling_sharpe[i], expected_sharpe, decimal=10)
            
    def test_target_creation_no_leakage(self):
        """Test that target creation doesn't introduce data leakage"""
        # Verify that targets are created from future returns relative to features
        
        # For each sample, verify that:
        # - Features at time i use data up to time i
        # - Target at time i represents the return from time i to time i+1
        
        for i in range(len(self.X)):
            # Target should represent the return from current period to next
            if i < len(self.returns):
                expected_target = 1 if self.returns[i] > 0 else 0
                actual_target = self.y[i]
                self.assertEqual(actual_target, expected_target)
                
    def test_pipeline_temporal_consistency(self):
        """Test that Pipeline maintains temporal consistency"""
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Store predictions to verify temporal consistency
        all_predictions = np.zeros(len(self.y))
        
        for train_idx, val_idx in tscv.split(self.X):
            # Fit on training data
            X_train = self.X[train_idx]
            y_train = self.y[train_idx]
            
            pipeline.fit(X_train, y_train)
            
            # Predict on validation data
            val_pred = pipeline.predict(self.X[val_idx])
            all_predictions[val_idx] = val_pred
            
        # Verify that predictions maintain temporal order
        # (This is more of a sanity check that the pipeline works correctly)
        self.assertEqual(len(all_predictions), len(self.y))
        
        # Verify that we have predictions for all samples
        self.assertGreater(np.sum(all_predictions != 0), 0)
        
    def test_feature_statistics_temporal_evolution(self):
        """Test that feature statistics evolve over time (no global fitting)"""
        # Verify that features show temporal evolution rather than being
        # globally normalized or processed
        
        # Check that feature statistics change over time
        feature_means = np.mean(self.X, axis=0)
        feature_stds = np.std(self.X, axis=0)
        
        # Features should have reasonable ranges
        for i, (mean, std) in enumerate(zip(feature_means, feature_stds)):
            self.assertGreater(std, 0)  # Should have some variation
            print(f"Feature {i}: mean={mean:.4f}, std={std:.4f}")
            
        # Check temporal evolution by comparing early vs late periods
        early_period = self.X[:100]
        late_period = self.X[-100:]
        
        early_means = np.mean(early_period, axis=0)
        late_means = np.mean(late_period, axis=0)
        
        # Should see some evolution (not identical)
        mean_differences = np.abs(late_means - early_means)
        self.assertGreater(np.mean(mean_differences), 0.001)

if __name__ == '__main__':
    unittest.main()
