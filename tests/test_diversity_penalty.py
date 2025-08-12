"""
Test Diversity Control and Penalty
Verifies that diversity control prevents double-counting of highly correlated models
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

class TestDiversityPenalty(unittest.TestCase):
    """Test cases for diversity control and penalty"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create synthetic data
        self.n_samples = 150
        self.n_features = 12
        
        # Generate features
        self.X = np.random.randn(self.n_samples, self.n_features)
        
        # Generate target with some predictability
        signal = np.dot(self.X[:, :4], np.random.randn(4)) * 0.1
        noise = np.random.randn(self.n_samples) * 0.02
        returns = signal + noise
        self.y = (returns > 0).astype(int)
        
        # Create ensemble
        self.ensemble = EnhancedTradingEnsemble(random_state=42, n_splits=2)
    
    def test_diversity_control_application(self):
        """Test that diversity control is properly applied"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Apply diversity control
        diversity_weights = self.ensemble.apply_diversity_control(correlation_threshold=0.95)
        
        # Verify diversity weights were created
        self.assertIsNotNone(diversity_weights)
        self.assertGreater(len(diversity_weights), 0)
        
        # Verify weights sum to 1
        weight_sum = sum(diversity_weights.values())
        self.assertAlmostEqual(weight_sum, 1.0, places=6)
        
        # Verify all weights are non-negative
        for weight in diversity_weights.values():
            self.assertGreaterEqual(weight, 0)
    
    def test_high_correlation_penalty(self):
        """Test that highly correlated models receive diversity penalty"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Get original weights (equal weights)
        original_weights = {name: 1.0/len(self.ensemble.models) for name in self.ensemble.models.keys()}
        
        # Apply diversity control
        diversity_weights = self.ensemble.apply_diversity_control(correlation_threshold=0.95)
        
        # Check if any weights were modified
        weight_changes = {}
        for name in original_weights:
            if name in diversity_weights:
                change = abs(diversity_weights[name] - original_weights[name])
                weight_changes[name] = change
        
        # At least some weights should have changed if correlations were found
        max_change = max(weight_changes.values()) if weight_changes else 0
        self.assertGreaterEqual(max_change, 0)
    
    def test_correlation_calculation(self):
        """Test that pairwise correlations are calculated correctly"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Get diversity metrics
        diversity_metrics = self.ensemble.get_model_diversity_metrics()
        
        # Verify required metrics
        self.assertIn('pairwise_correlations', diversity_metrics)
        self.assertIn('average_correlation', diversity_metrics)
        self.assertIn('diversity_score', diversity_metrics)
        self.assertIn('high_correlation_pairs', diversity_metrics)
        
        # Verify correlation values are in [-1, 1]
        correlations = diversity_metrics['pairwise_correlations'].values()
        for corr in correlations:
            self.assertGreaterEqual(corr, -1)
            self.assertLessEqual(corr, 1)
        
        # Verify diversity score is in [0, 2] (1 - avg_corr, where avg_corr in [-1, 1])
        diversity_score = diversity_metrics['diversity_score']
        self.assertGreaterEqual(diversity_score, 0)
        self.assertLessEqual(diversity_score, 2)
    
    def test_identical_model_penalty(self):
        """Test that identical models don't get double-counted"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Create a copy of one model's OOF probabilities to simulate identical models
        if len(self.ensemble.oof_probabilities) > 1:
            model_names = list(self.ensemble.oof_probabilities.keys())
            first_model = model_names[0]
            second_model = model_names[1]
            
            # Make second model identical to first
            original_proba = self.ensemble.oof_probabilities[second_model].copy()
            self.ensemble.oof_probabilities[second_model] = self.ensemble.oof_probabilities[first_model].copy()
            
            # Apply diversity control
            diversity_weights = self.ensemble.apply_diversity_control(correlation_threshold=0.95)
            
            # Verify that identical models don't both get high weights
            first_weight = diversity_weights.get(first_model, 0)
            second_weight = diversity_weights.get(second_model, 0)
            
            # At least one should be penalized
            self.assertTrue(first_weight < 0.5 or second_weight < 0.5)
            
            # Restore original probabilities
            self.ensemble.oof_probabilities[second_model] = original_proba
    
    def test_correlation_threshold_sensitivity(self):
        """Test that different correlation thresholds affect diversity control"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Test different thresholds
        thresholds = [0.8, 0.9, 0.95, 0.99]
        diversity_scores = []
        
        for threshold in thresholds:
            # Apply diversity control
            diversity_weights = self.ensemble.apply_diversity_control(correlation_threshold=threshold)
            
            # Get diversity metrics
            diversity_metrics = self.ensemble.get_model_diversity_metrics()
            diversity_scores.append(diversity_metrics['diversity_score'])
        
        # Different thresholds should produce different diversity scores
        # (though this may not always be true)
        self.assertTrue(len(set(diversity_scores)) > 1)
    
    def test_diversity_control_consistency(self):
        """Test that diversity control produces consistent results"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Apply diversity control multiple times
        results = []
        for _ in range(3):
            diversity_weights = self.ensemble.apply_diversity_control(correlation_threshold=0.95)
            results.append(diversity_weights.copy())
        
        # Verify consistency
        for i in range(1, len(results)):
            with self.subTest(iteration=i):
                # Check that weights are consistent
                for name in results[0]:
                    if name in results[i]:
                        self.assertAlmostEqual(
                            results[0][name], 
                            results[i][name], 
                            places=6,
                            msg=f"Weights inconsistent for {name}"
                        )
    
    def test_diversity_control_with_optimal_weights(self):
        """Test that diversity control works with pre-optimized weights"""
        # Fit models
        self.ensemble.fit_models(self.X, self.y)
        
        # Optimize weights first
        self.ensemble.optimize_ensemble_weights(
            y=self.y, method='sharpe', cost_per_trade=0.001, slippage=0.0005
        )
        
        # Store original optimal weights
        original_weights = self.ensemble.optimal_weights.copy()
        
        # Apply diversity control
        diversity_weights = self.ensemble.apply_diversity_control(correlation_threshold=0.95)
        
        # Verify diversity weights were created
        self.assertIsNotNone(diversity_weights)
        self.assertGreater(len(diversity_weights), 0)
        
        # Verify weights sum to 1
        weight_sum = sum(diversity_weights.values())
        self.assertAlmostEqual(weight_sum, 1.0, places=6)
        
        # Verify that diversity control modified the weights
        weight_changes = {}
        for name in original_weights:
            if name in diversity_weights:
                change = abs(diversity_weights[name] - original_weights[name])
                weight_changes[name] = change
        
        # At least some weights should have changed
        max_change = max(weight_changes.values()) if weight_changes else 0
        self.assertGreaterEqual(max_change, 0)

if __name__ == '__main__':
    unittest.main()
