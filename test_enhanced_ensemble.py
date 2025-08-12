"""
Comprehensive Test Suite for EnhancedTradingEnsemble
Tests all functionality including model fitting, calibration, weight optimization, and diversity control
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import the enhanced ensemble
from enhanced_ensemble import EnhancedTradingEnsemble

class TestEnhancedTradingEnsemble(unittest.TestCase):
    """Test cases for EnhancedTradingEnsemble class"""
    
    def setUp(self):
        """Set up test data and ensemble"""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create synthetic trading data
        n_samples = 1000
        n_features = 20
        
        # Generate features (technical indicators)
        self.X = np.random.randn(n_samples, n_features)
        
        # Generate synthetic returns (target)
        # Simulate some predictability
        signal = np.dot(self.X[:, :5], np.random.randn(5)) * 0.1
        noise = np.random.randn(n_samples) * 0.02
        returns = signal + noise
        
        # Convert to binary classification (positive/negative returns)
        self.y = (returns > 0).astype(int)
        
        # Create ensemble
        self.ensemble = EnhancedTradingEnsemble(random_state=42, n_splits=3)
        
        # Mock data for testing
        self.mock_X = np.random.randn(100, 10)
        self.mock_y = np.random.randint(0, 2, 100)
    
    def test_initialization(self):
        """Test ensemble initialization"""
        self.assertIsNotNone(self.ensemble)
        self.assertEqual(len(self.ensemble.models), 9)  # Should have 9 models
        self.assertFalse(self.ensemble.is_fitted)
        self.assertEqual(self.ensemble.random_state, 42)
        self.assertEqual(self.ensemble.n_splits, 3)
        
        # Check that all expected models are present
        expected_models = [
            'RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM', 
            'CatBoost', 'LogisticRegression', 'RidgeClassifier', 'SVC', 'MLP'
        ]
        
        for model_name in expected_models:
            if model_name in self.ensemble.models:
                self.assertIsNotNone(self.ensemble.models[model_name])
    
    def test_model_pipelines(self):
        """Test that all models are properly wrapped in sklearn Pipelines"""
        for name, pipeline in self.ensemble.models.items():
            self.assertIsInstance(pipeline, type(self.ensemble.models['RandomForest']))
            
            # Check that pipeline has scaler and classifier steps
            steps = [step[0] for step in pipeline.steps]
            self.assertIn('scaler', steps)
            self.assertIn('classifier', steps)
    
    def test_fit_models(self):
        """Test model fitting with TimeSeriesSplit"""
        # Test with small dataset to avoid long execution
        X_small = self.X[:100]
        y_small = self.y[:100]
        
        self.ensemble.fit_models(X_small, y_small)
        
        self.assertTrue(self.ensemble.is_fitted)
        self.assertGreater(len(self.ensemble.oof_probabilities), 0)
        
        # Check that OOF probabilities have correct shape
        for name, proba in self.ensemble.oof_probabilities.items():
            self.assertEqual(proba.shape[0], len(y_small))
            self.assertTrue(np.all((proba >= 0) & (proba <= 1)))  # Valid probabilities
    
    def test_calibrate_probabilities(self):
        """Test probability calibration"""
        # First fit models
        X_small = self.X[:100]
        y_small = self.y[:100]
        self.ensemble.fit_models(X_small, y_small)
        
        # Test isotonic calibration
        self.ensemble.calibrate_probabilities(method='isotonic')
        self.assertGreater(len(self.ensemble.calibrators), 0)
        
        # Test sigmoid calibration
        self.ensemble.calibrators = {}  # Reset
        self.ensemble.calibrate_probabilities(method='sigmoid')
        self.assertGreater(len(self.ensemble.calibrators), 0)
    
    def test_weight_optimization_sharpe(self):
        """Test weight optimization for Sharpe ratio"""
        # Fit models first
        X_small = self.X[:100]
        y_small = self.y[:100]
        self.ensemble.fit_models(X_small, y_small)
        
        # Test Sharpe optimization
        result = self.ensemble.optimize_ensemble_weights(
            y_small, method='sharpe', cost_per_trade=0.001, slippage=0.0005
        )
        
        self.assertIn('optimal_weights', result)
        self.assertIn('optimal_threshold', result)
        self.assertIn('performance_metrics', result)
        
        # Check that weights sum to 1
        weights_sum = sum(result['optimal_weights'].values())
        self.assertAlmostEqual(weights_sum, 1.0, places=5)
        
        # Check that threshold is in valid range
        self.assertGreaterEqual(result['optimal_threshold'], 0.50)
        self.assertLessEqual(result['optimal_threshold'], 0.70)
    
    def test_weight_optimization_cagr(self):
        """Test weight optimization for CAGR"""
        # Fit models first
        X_small = self.X[:100]
        y_small = self.y[:100]
        self.ensemble.fit_models(X_small, y_small)
        
        # Test CAGR optimization
        result = self.ensemble.optimize_ensemble_weights(
            y_small, method='cagr', cost_per_trade=0.001, slippage=0.0005
        )
        
        self.assertIn('optimal_weights', result)
        self.assertIn('optimal_threshold', result)
        self.assertIn('performance_metrics', result)
    
    def test_weight_optimization_combined(self):
        """Test combined Sharpe and CAGR optimization"""
        # Fit models first
        X_small = self.X[:100]
        y_small = self.y[:100]
        self.ensemble.fit_models(X_small, y_small)
        
        # Test combined optimization
        result = self.ensemble.optimize_ensemble_weights(
            y_small, method='sharpe_cagr', cost_per_trade=0.001, slippage=0.0005
        )
        
        self.assertIn('optimal_weights', result)
        self.assertIn('optimal_threshold', result)
        self.assertIn('performance_metrics', result)
    
    def test_threshold_optimization(self):
        """Test threshold optimization"""
        # Create synthetic ensemble probabilities
        ensemble_proba = np.random.uniform(0, 1, 100)
        y_test = np.random.randint(0, 2, 100)
        
        threshold = self.ensemble._optimize_threshold(
            ensemble_proba, y_test, cost_per_trade=0.001, slippage=0.0005
        )
        
        self.assertGreaterEqual(threshold, 0.50)
        self.assertLessEqual(threshold, 0.70)
    
    def test_trading_returns_calculation(self):
        """Test trading returns calculation"""
        # Create synthetic data
        ensemble_proba = np.array([0.6, 0.7, 0.8, 0.9, 0.4])
        y_test = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
        threshold = 0.5
        cost_per_trade = 0.001
        slippage = 0.0005
        
        returns = self.ensemble._calculate_trading_returns(
            ensemble_proba, y_test, threshold, cost_per_trade, slippage
        )
        
        self.assertEqual(len(returns), len(ensemble_proba))
        self.assertIsInstance(returns, np.ndarray)
    
    def test_trading_metrics_calculation(self):
        """Test trading metrics calculation"""
        # Create synthetic data
        ensemble_proba = np.random.uniform(0, 1, 100)
        y_test = np.random.uniform(-0.02, 0.02, 100)
        threshold = 0.5
        cost_per_trade = 0.001
        slippage = 0.0005
        
        metrics = self.ensemble._calculate_trading_metrics(
            ensemble_proba, y_test, threshold, cost_per_trade, slippage
        )
        
        expected_keys = [
            'total_return', 'annualized_return', 'volatility', 
            'sharpe_ratio', 'max_drawdown', 'trade_count', 'turnover'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))
    
    def test_diversity_control(self):
        """Test diversity control functionality"""
        # Fit models first
        X_small = self.X[:100]
        y_small = self.y[:100]
        self.ensemble.fit_models(X_small, y_small)
        
        # Optimize weights first
        self.ensemble.optimize_ensemble_weights(
            y_small, method='sharpe', cost_per_trade=0.001, slippage=0.0005
        )
        
        # Apply diversity control
        adjusted_weights = self.ensemble.apply_diversity_control(correlation_threshold=0.95)
        
        self.assertIsInstance(adjusted_weights, dict)
        self.assertEqual(len(adjusted_weights), len(self.ensemble.optimal_weights))
        
        # Check that weights still sum to 1
        weights_sum = sum(adjusted_weights.values())
        self.assertAlmostEqual(weights_sum, 1.0, places=5)
    
    def test_diversity_metrics(self):
        """Test diversity metrics calculation"""
        # Fit models first
        X_small = self.X[:100]
        y_small = self.y[:100]
        self.ensemble.fit_models(X_small, y_small)
        
        # Get diversity metrics
        diversity_metrics = self.ensemble.get_model_diversity_metrics()
        
        self.assertIn('pairwise_correlations', diversity_metrics)
        self.assertIn('average_correlation', diversity_metrics)
        self.assertIn('diversity_score', diversity_metrics)
        self.assertIn('high_correlation_pairs', diversity_metrics)
        
        # Check that diversity score is in valid range
        self.assertGreaterEqual(diversity_metrics['diversity_score'], 0)
        self.assertLessEqual(diversity_metrics['diversity_score'], 1)
    
    def test_probability_prediction(self):
        """Test probability prediction"""
        # Fit models first
        X_small = self.X[:100]
        y_small = self.y[:100]
        self.ensemble.fit_models(X_small, y_small)
        
        # Test prediction without optimal weights (should use equal weights)
        proba = self.ensemble.predict_proba(X_small[:10])
        
        self.assertEqual(len(proba), 10)
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))  # Valid probabilities
        
        # Test prediction with optimal weights
        self.ensemble.optimize_ensemble_weights(
            y_small, method='sharpe', cost_per_trade=0.001, slippage=0.0005
        )
        
        proba_optimal = self.ensemble.predict_proba(X_small[:10])
        
        self.assertEqual(len(proba_optimal), 10)
        self.assertTrue(np.all((proba_optimal >= 0) & (proba_optimal <= 1)))
    
    def test_individual_model_evaluation(self):
        """Test individual model evaluation"""
        # Fit models first
        X_small = self.X[:100]
        y_small = self.y[:100]
        self.ensemble.fit_models(X_small, y_small)
        
        # Evaluate individual models
        results = self.ensemble.evaluate_individual_models(X_small, y_small)
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check that each model has the expected metrics
        for model_name, metrics in results.items():
            expected_keys = ['auc', 'brier_score', 'calibration_mae', 'oof_sharpe']
            for key in expected_keys:
                self.assertIn(key, metrics)
                self.assertIsInstance(metrics[key], (int, float))
    
    def test_calibration_error_calculation(self):
        """Test calibration error calculation"""
        # Create synthetic data
        proba = np.random.uniform(0, 1, 100)
        y_test = np.random.randint(0, 2, 100)
        
        error = self.ensemble._calculate_calibration_error(proba, y_test)
        
        self.assertIsInstance(error, float)
        self.assertGreaterEqual(error, 0)
    
    def test_save_and_load_ensemble(self):
        """Test ensemble saving and loading"""
        # Fit models first
        X_small = self.X[:100]
        y_small = self.y[:100]
        self.ensemble.fit_models(X_small, y_small)
        
        # Optimize weights
        self.ensemble.optimize_ensemble_weights(
            y_small, method='sharpe', cost_per_trade=0.001, slippage=0.0005
        )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            filepath = tmp_file.name
        
        try:
            # Save ensemble
            self.ensemble.save_ensemble(filepath)
            self.assertTrue(os.path.exists(filepath))
            
            # Create new ensemble and load
            new_ensemble = EnhancedTradingEnsemble()
            new_ensemble.load_ensemble(filepath)
            
            # Check that loaded ensemble has same properties
            self.assertEqual(new_ensemble.is_fitted, self.ensemble.is_fitted)
            self.assertEqual(len(new_ensemble.models), len(self.ensemble.models))
            self.assertEqual(new_ensemble.optimal_threshold, self.ensemble.optimal_threshold)
            
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_ensemble_summary(self):
        """Test ensemble summary generation"""
        # Fit models first
        X_small = self.X[:100]
        y_small = self.y[:100]
        self.ensemble.fit_models(X_small, y_small)
        
        # Get summary
        summary = self.ensemble.get_ensemble_summary()
        
        expected_keys = [
            'n_models', 'active_models', 'is_fitted', 'optimal_weights',
            'optimal_threshold', 'n_splits', 'random_state'
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary)
        
        self.assertEqual(summary['n_models'], len(self.ensemble.models))
        self.assertEqual(summary['is_fitted'], self.ensemble.is_fitted)
        self.assertEqual(summary['random_state'], 42)
        self.assertEqual(summary['n_splits'], 3)
    
    def test_error_handling(self):
        """Test error handling for invalid operations"""
        # Test prediction without fitting
        with self.assertRaises(ValueError):
            self.ensemble.predict_proba(self.mock_X)
        
        # Test calibration without fitting
        with self.assertRaises(ValueError):
            self.ensemble.calibrate_probabilities()
        
        # Test weight optimization without fitting
        with self.assertRaises(ValueError):
            self.ensemble.optimize_ensemble_weights(self.mock_y)
        
        # Test diversity control without optimal weights
        with self.assertRaises(ValueError):
            self.ensemble.apply_diversity_control()
        
        # Test individual evaluation without fitting
        with self.assertRaises(ValueError):
            self.ensemble.evaluate_individual_models(self.mock_X, self.mock_y)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test with very small dataset
        X_tiny = self.X[:10]
        y_tiny = self.y[:10]
        
        tiny_ensemble = EnhancedTradingEnsemble(n_splits=2)
        tiny_ensemble.fit_models(X_tiny, y_tiny)
        
        self.assertTrue(tiny_ensemble.is_fitted)
        self.assertGreater(len(tiny_ensemble.oof_probabilities), 0)
        
        # Test with all positive or all negative targets
        y_all_positive = np.ones(50)
        y_all_negative = np.zeros(50)
        
        pos_ensemble = EnhancedTradingEnsemble(n_splits=2)
        pos_ensemble.fit_models(X_tiny, y_all_positive)
        
        neg_ensemble = EnhancedTradingEnsemble(n_splits=2)
        neg_ensemble.fit_models(X_tiny, y_all_negative)
        
        self.assertTrue(pos_ensemble.is_fitted)
        self.assertTrue(neg_ensemble.is_fitted)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

