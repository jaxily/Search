#!/usr/bin/env python3
"""
Test Sharpe ratio scaling from daily to annual returns
"""

import numpy as np
import unittest
from scipy import stats

class TestSharpeScaling(unittest.TestCase):
    """Test Sharpe ratio annualization scaling"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
    def test_sharpe_scaling_normal_returns(self):
        """Test Sharpe ratio scaling with normal daily returns"""
        # Generate synthetic daily returns (normal distribution)
        n_days = 252 * 5  # 5 years of trading days
        daily_mean = 0.0005  # 0.05% daily return
        daily_std = 0.015    # 1.5% daily volatility
        
        daily_returns = np.random.normal(daily_mean, daily_std, n_days)
        
        # Calculate daily Sharpe
        daily_sharpe = daily_mean / daily_std
        
        # Calculate annualized Sharpe
        annualized_sharpe = daily_sharpe * np.sqrt(252)
        
        # Verify the scaling
        expected_annual_sharpe = daily_mean / daily_std * np.sqrt(252)
        np.testing.assert_almost_equal(annualized_sharpe, expected_annual_sharpe, decimal=10)
        
        # Verify realistic values
        self.assertGreater(annualized_sharpe, 0)
        self.assertLess(annualized_sharpe, 10)  # Reasonable upper bound
        
    def test_sharpe_scaling_zero_volatility(self):
        """Test Sharpe ratio with zero volatility (should be infinite)"""
        daily_returns = np.ones(252) * 0.001  # Constant positive returns
        
        daily_mean = np.mean(daily_returns)
        daily_std = np.std(daily_returns)
        
        # With zero volatility, Sharpe should be infinite
        if daily_std == 0:
            self.assertTrue(np.isinf(daily_mean / daily_std))
        else:
            sharpe = daily_mean / daily_std
            self.assertGreater(sharpe, 0)
            
    def test_sharpe_scaling_negative_returns(self):
        """Test Sharpe ratio with negative expected returns"""
        # Generate synthetic daily returns with negative mean
        n_days = 252 * 3  # 3 years
        daily_mean = -0.0002  # Negative daily return
        daily_std = 0.02      # 2% daily volatility
        
        daily_returns = np.random.normal(daily_mean, daily_std, n_days)
        
        # Calculate Sharpe
        daily_sharpe = daily_mean / daily_std
        annualized_sharpe = daily_sharpe * np.sqrt(252)
        
        # Negative Sharpe is expected
        self.assertLess(annualized_sharpe, 0)
        
        # Verify scaling
        expected_annual_sharpe = daily_mean / daily_std * np.sqrt(252)
        np.testing.assert_almost_equal(annualized_sharpe, expected_annual_sharpe, decimal=10)
        
    def test_sharpe_scaling_empirical_data(self):
        """Test Sharpe ratio scaling with empirical-like data"""
        # Simulate more realistic market data
        np.random.seed(123)
        
        # Generate daily returns with some autocorrelation and fat tails
        n_days = 252 * 10  # 10 years
        
        # Base returns
        base_returns = np.random.normal(0.0003, 0.018, n_days)
        
        # Add some autocorrelation
        for i in range(1, n_days):
            base_returns[i] += 0.1 * base_returns[i-1]
            
        # Add some fat tails (random large moves)
        fat_tail_days = np.random.choice(n_days, size=n_days//100, replace=False)
        base_returns[fat_tail_days] *= 3
        
        # Calculate statistics
        daily_mean = np.mean(base_returns)
        daily_std = np.std(base_returns)
        
        # Calculate Sharpe ratios
        daily_sharpe = daily_mean / daily_std
        annualized_sharpe = daily_sharpe * np.sqrt(252)
        
        # Verify scaling relationship
        expected_annual_sharpe = daily_mean / daily_std * np.sqrt(252)
        np.testing.assert_almost_equal(annualized_sharpe, expected_annual_sharpe, decimal=10)
        
        # Verify realistic bounds
        self.assertGreater(annualized_sharpe, -5)  # Reasonable lower bound
        self.assertLess(annualized_sharpe, 5)      # Reasonable upper bound
        
    def test_sharpe_consistency_different_periods(self):
        """Test Sharpe ratio consistency across different time periods"""
        # Generate 10 years of daily data
        np.random.seed(456)
        n_days = 252 * 10
        daily_returns = np.random.normal(0.0004, 0.016, n_days)
        
        # Calculate Sharpe for different periods
        periods = [252, 252*2, 252*5, 252*10]  # 1, 2, 5, 10 years
        
        sharpe_ratios = []
        for period in periods:
            if period <= n_days:
                period_returns = daily_returns[:period]
                period_mean = np.mean(period_returns)
                period_std = np.std(period_returns)
                
                if period_std > 0:
                    period_sharpe = period_mean / period_std
                    annualized_sharpe = period_sharpe * np.sqrt(252)
                    sharpe_ratios.append(annualized_sharpe)
                    
        # All annualized Sharpe ratios should be similar (within sampling error)
        if len(sharpe_ratios) > 1:
            sharpe_std = np.std(sharpe_ratios)
            self.assertLess(sharpe_std, 0.5)  # Should be relatively stable
            
    def test_sharpe_formula_components(self):
        """Test individual components of Sharpe ratio calculation"""
        # Generate test data
        daily_returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        
        # Calculate components
        daily_mean = np.mean(daily_returns)
        daily_std = np.std(daily_returns, ddof=1)  # Sample standard deviation
        
        # Verify mean calculation
        expected_mean = (0.01 - 0.005 + 0.02 - 0.01 + 0.015) / 5
        np.testing.assert_almost_equal(daily_mean, expected_mean, decimal=10)
        
        # Verify standard deviation calculation
        expected_std = np.sqrt(np.sum((daily_returns - daily_mean)**2) / 4)  # n-1
        np.testing.assert_almost_equal(daily_std, expected_std, decimal=10)
        
        # Verify Sharpe ratio
        sharpe = daily_mean / daily_std
        expected_sharpe = expected_mean / expected_std
        np.testing.assert_almost_equal(sharpe, expected_sharpe, decimal=10)
        
        # Verify annualization
        annualized_sharpe = sharpe * np.sqrt(252)
        expected_annual = expected_sharpe * np.sqrt(252)
        np.testing.assert_almost_equal(annualized_sharpe, expected_annual, decimal=10)

if __name__ == '__main__':
    unittest.main()
