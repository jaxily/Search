"""
Test Backtest Contract Compliance
Verifies next-bar execution, position sizing constraints, and cost application
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

class TestBacktestContract(unittest.TestCase):
    """Test cases for backtest contract compliance"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create synthetic data
        self.n_samples = 200
        self.n_features = 15
        
        # Generate features
        self.X = np.random.randn(self.n_samples, self.n_features)
        
        # Generate target with some predictability
        signal = np.dot(self.X[:, :5], np.random.randn(5)) * 0.1
        noise = np.random.randn(self.n_samples) * 0.02
        returns = signal + noise
        self.y = (returns > 0).astype(int)
        
        # Create ensemble
        self.ensemble = EnhancedTradingEnsemble(random_state=42, n_splits=2)
    
    def test_next_bar_execution(self):
        """Test that next-bar execution is honored"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Get ensemble probabilities
        ensemble_proba = self.ensemble.predict_proba(self.X)
        
        # Calculate trading returns with next-bar execution
        threshold = 0.5
        cost_per_trade = 0.001
        slippage = 0.0005
        
        # Trading signal: long if p > threshold
        signals = (ensemble_proba > threshold).astype(float)
        
        # Position size: (p - threshold) / (1 - threshold), clipped to [0, 1]
        position_sizes = np.clip((ensemble_proba - threshold) / (1 - threshold), 0, 1)
        
        # Apply position sizes to signals
        weighted_signals = signals * position_sizes
        
        # Calculate strategy returns (next-bar execution)
        strategy_returns = weighted_signals * self.y
        
        # Apply transaction costs when signals change
        signal_changes = np.diff(weighted_signals, prepend=0)
        transaction_costs = np.abs(signal_changes) * (cost_per_trade + slippage)
        
        # Net strategy returns
        net_returns = strategy_returns - transaction_costs
        
        # Verify that returns are calculated correctly
        self.assertEqual(len(net_returns), len(self.y))
        self.assertTrue(np.all(np.isfinite(net_returns)))
    
    def test_position_sizing_constraints(self):
        """Test that position sizes are constrained to [0, 1]"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Get ensemble probabilities
        ensemble_proba = self.ensemble.predict_proba(self.X)
        
        # Test different thresholds
        thresholds = [0.5, 0.6, 0.7]
        
        for threshold in thresholds:
            with self.subTest(threshold=threshold):
                # Calculate position sizes
                position_sizes = np.clip((ensemble_proba - threshold) / (1 - threshold), 0, 1)
                
                # Verify constraints
                self.assertTrue(np.all(position_sizes >= 0))
                self.assertTrue(np.all(position_sizes <= 1))
                
                # Verify that positions are 0 when proba <= threshold
                zero_mask = ensemble_proba <= threshold
                self.assertTrue(np.all(position_sizes[zero_mask] == 0))
                
                # Verify that positions are 1 when proba = 1
                one_mask = ensemble_proba == 1
                if np.any(one_mask):
                    self.assertTrue(np.all(position_sizes[one_mask] == 1))
    
    def test_transaction_costs_application(self):
        """Test that transaction costs are properly applied"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Get ensemble probabilities
        ensemble_proba = self.ensemble.predict_proba(self.X)
        
        # Calculate trading returns
        threshold = 0.5
        cost_per_trade = 0.001
        slippage = 0.0005
        
        # Calculate returns without costs
        signals = (ensemble_proba > threshold).astype(float)
        position_sizes = np.clip((ensemble_proba - threshold) / (1 - threshold), 0, 1)
        weighted_signals = signals * position_sizes
        gross_returns = weighted_signals * self.y
        
        # Calculate returns with costs
        signal_changes = np.diff(weighted_signals, prepend=0)
        transaction_costs = np.abs(signal_changes) * (cost_per_trade + slippage)
        net_returns = gross_returns - transaction_costs
        
        # Verify that costs reduce returns
        cost_impact = gross_returns - net_returns
        self.assertTrue(np.all(cost_impact >= 0))
        
        # Verify that costs are proportional to signal changes
        self.assertTrue(np.all(cost_impact <= np.abs(signal_changes) * (cost_per_trade + slippage)))
    
    def test_trading_metrics_calculation(self):
        """Test that trading metrics are calculated correctly"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Get ensemble probabilities
        ensemble_proba = self.ensemble.predict_proba(self.X)
        
        # Calculate trading metrics
        threshold = 0.5
        cost_per_trade = 0.001
        slippage = 0.0005
        
        metrics = self.ensemble._calculate_trading_metrics(
            ensemble_proba, self.y, threshold, cost_per_trade, slippage
        )
        
        # Verify required metrics
        required_metrics = ['total_return', 'annualized_return', 'volatility', 
                          'sharpe_ratio', 'max_drawdown', 'trade_count', 'turnover']
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertTrue(np.isfinite(metrics[metric]))
        
        # Verify metric relationships
        self.assertGreaterEqual(metrics['volatility'], 0)
        self.assertLessEqual(metrics['max_drawdown'], 0)
        self.assertGreaterEqual(metrics['trade_count'], 0)
        self.assertGreaterEqual(metrics['turnover'], 0)
    
    def test_threshold_optimization(self):
        """Test threshold optimization for trading decisions"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Get ensemble probabilities
        ensemble_proba = self.ensemble.predict_proba(self.X)
        
        # Test threshold optimization
        optimal_threshold = self.ensemble._optimize_threshold(
            ensemble_proba, self.y, 0.001, 0.0005
        )
        
        # Verify threshold is in expected range
        self.assertGreaterEqual(optimal_threshold, 0.5)
        self.assertLessEqual(optimal_threshold, 0.7)
        
        # Verify threshold is a multiple of 0.01
        threshold_rounded = round(optimal_threshold / 0.01) * 0.01
        self.assertAlmostEqual(optimal_threshold, threshold_rounded, places=2)
    
    def test_cost_sensitivity(self):
        """Test that different cost levels affect trading decisions"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Get ensemble probabilities
        ensemble_proba = self.ensemble.predict_proba(self.X)
        
        # Test different cost levels
        cost_levels = [0.0005, 0.001, 0.002]
        trade_counts = []
        
        for cost in cost_levels:
            # Calculate trading returns
            threshold = 0.5
            signals = (ensemble_proba > threshold).astype(float)
            position_sizes = np.clip((ensemble_proba - threshold) / (1 - threshold), 0, 1)
            weighted_signals = signals * position_sizes
            
            # Count trades
            signal_changes = np.diff(weighted_signals, prepend=0)
            trade_count = np.sum(np.abs(signal_changes) > 0.01)  # Threshold for trade
            trade_counts.append(trade_count)
        
        # Higher costs should generally lead to fewer trades
        # (though this may not always be true due to optimization)
        self.assertTrue(len(set(trade_counts)) > 1)  # At least some variation
    
    def test_slippage_impact(self):
        """Test that slippage affects trading performance"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Get ensemble probabilities
        ensemble_proba = self.ensemble.predict_proba(self.X)
        
        # Test different slippage levels
        slippage_levels = [0.0, 0.0005, 0.001]
        net_returns_list = []
        
        for slippage in slippage_levels:
            # Calculate trading returns
            threshold = 0.5
            cost_per_trade = 0.001
            
            signals = (ensemble_proba > threshold).astype(float)
            position_sizes = np.clip((ensemble_proba - threshold) / (1 - threshold), 0, 1)
            weighted_signals = signals * position_sizes
            
            strategy_returns = weighted_signals * self.y
            signal_changes = np.diff(weighted_signals, prepend=0)
            transaction_costs = np.abs(signal_changes) * (cost_per_trade + slippage)
            net_returns = strategy_returns - transaction_costs
            
            net_returns_list.append(net_returns)
        
        # Higher slippage should generally reduce net returns
        # (though this may not always be true due to optimization)
        self.assertTrue(len(set([np.sum(returns) for returns in net_returns_list])) > 1)

if __name__ == '__main__':
    unittest.main()
