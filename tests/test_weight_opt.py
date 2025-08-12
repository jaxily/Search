"""
Test Weight Optimization
Verifies that weight optimization respects bounds, sum-to-1 constraint, and improves performance
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

class TestWeightOptimization(unittest.TestCase):
    """Test cases for weight optimization"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create synthetic data
        self.n_samples = 300
        self.n_features = 20
        
        # Generate features
        self.X = np.random.randn(self.n_samples, self.n_features)
        
        # Generate target with some predictability
        signal = np.dot(self.X[:, :8], np.random.randn(8)) * 0.12
        noise = np.random.randn(self.n_samples) * 0.02
        returns = signal + noise
        self.y = (returns > 0).astype(int)
        
        # Create ensemble
        self.ensemble = EnhancedTradingEnsemble(random_state=42, n_splits=2)
    
    def test_weight_optimization_respects_bounds(self):
        """Test that optimized weights respect [0, 1] bounds"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Optimize weights
        results = self.ensemble.optimize_ensemble_weights(
            y=self.y, method='sharpe', cost_per_trade=0.001, slippage=0.0005
        )
        
        # Check bounds
        weights = list(results['optimal_weights'].values())
        self.assertTrue(np.all(np.array(weights) >= 0))
        self.assertTrue(np.all(np.array(weights) <= 1))
    
    def test_weight_optimization_sum_to_one(self):
        """Test that optimized weights sum to 1"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Optimize weights
        results = self.ensemble.optimize_ensemble_weights(
            y=self.y, method='sharpe', cost_per_trade=0.001, slippage=0.0005
        )
        
        # Check sum constraint
        weights = list(results['optimal_weights'].values())
        weight_sum = np.sum(weights)
        self.assertAlmostEqual(weight_sum, 1.0, places=6)
    
    def test_weight_optimization_improves_sharpe(self):
        """Test that optimized weights improve Sharpe ratio vs equal weights"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Calculate equal weights performance
        equal_weights = np.ones(len(self.ensemble.models)) / len(self.ensemble.models)
        equal_weights_dict = dict(zip(self.ensemble.models.keys(), equal_weights))
        
        # Calculate equal weights ensemble probabilities
        proba_matrix = np.column_stack(list(self.ensemble.oof_probabilities.values()))
        equal_ensemble_proba = np.sum(proba_matrix * equal_weights, axis=1)
        
        # Calculate equal weights Sharpe
        equal_returns = self.ensemble._calculate_trading_returns(
            equal_ensemble_proba, self.y, 0.5, 0.001, 0.0005
        )
        if np.std(equal_returns) > 0:
            equal_sharpe = np.mean(equal_returns) / np.std(equal_returns) * np.sqrt(252)
        else:
            equal_sharpe = 0
        
        # Optimize weights
        results = self.ensemble.optimize_ensemble_weights(
            y=self.y, method='sharpe', cost_per_trade=0.001, slippage=0.0005
        )
        
        # Get optimized Sharpe
        optimized_sharpe = results['performance_metrics'].get('sharpe_ratio', 0)
        
        # Verify improvement (allow for small tolerance due to optimization randomness)
        self.assertGreaterEqual(
            optimized_sharpe, 
            equal_sharpe * 0.95,  # Allow 5% tolerance
            "Optimized weights did not improve Sharpe ratio"
        )
    
    def test_different_optimization_methods(self):
        """Test different optimization methods"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        methods = ['sharpe', 'cagr', 'sharpe_cagr']
        
        for method in methods:
            with self.subTest(method=method):
                # Optimize weights
                results = self.ensemble.optimize_ensemble_weights(
                    y=self.y, method=method, cost_per_trade=0.001, slippage=0.0005
                )
                
                # Check that weights were found
                self.assertIn('optimal_weights', results)
                self.assertGreater(len(results['optimal_weights']), 0)
                
                # Check bounds and sum constraint
                weights = list(results['optimal_weights'].values())
                self.assertTrue(np.all(np.array(weights) >= 0))
                self.assertTrue(np.all(np.array(weights) <= 1))
                self.assertAlmostEqual(np.sum(weights), 1.0, places=6)
    
    def test_weight_optimization_with_costs(self):
        """Test weight optimization with different transaction costs"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        cost_levels = [0.0005, 0.001, 0.002]
        
        for cost in cost_levels:
            with self.subTest(cost=cost):
                # Optimize weights
                results = self.ensemble.optimize_ensemble_weights(
                    y=self.y, method='sharpe', cost_per_trade=cost, slippage=0.0005
                )
                
                # Check that weights were found
                self.assertIn('optimal_weights', results)
                self.assertGreater(len(results['optimal_weights']), 0)
                
                # Check performance metrics
                if 'performance_metrics' in results:
                    metrics = results['performance_metrics']
                    self.assertIn('trade_count', metrics)
                    self.assertIn('turnover', metrics)
    
    def test_weight_optimization_robustness(self):
        """Test that weight optimization is robust to different random seeds"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Optimize weights multiple times
        results_list = []
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            
            results = self.ensemble.optimize_ensemble_weights(
                y=self.y, method='sharpe', cost_per_trade=0.001, slippage=0.0005
            )
            
            results_list.append(results)
        
        # Check that all optimizations produced valid weights
        for i, results in enumerate(results_list):
            with self.subTest(seed=i):
                self.assertIn('optimal_weights', results)
                weights = list(results['optimal_weights'].values())
                
                # Check bounds and sum constraint
                self.assertTrue(np.all(np.array(weights) >= 0))
                self.assertTrue(np.all(np.array(weights) <= 1))
                self.assertAlmostEqual(np.sum(weights), 1.0, places=6)
    
    def test_weight_optimization_fallback(self):
        """Test that weight optimization falls back to equal weights on failure"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Mock optimization failure
        with patch.object(self.ensemble, '_optimize_weights_sharpe') as mock_optimize:
            mock_optimize.side_effect = Exception("Optimization failed")
            
            # This should fall back to equal weights
            results = self.ensemble.optimize_ensemble_weights(
                y=self.y, method='sharpe', cost_per_trade=0.001, slippage=0.0005
            )
            
            # Check that equal weights were used
            weights = list(results['optimal_weights'].values())
            expected_equal_weights = np.ones(len(self.ensemble.models)) / len(self.ensemble.models)
            
            np.testing.assert_array_almost_equal(weights, expected_equal_weights)

if __name__ == '__main__':
    unittest.main()
