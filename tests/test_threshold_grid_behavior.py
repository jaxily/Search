#!/usr/bin/env python3
"""
Test threshold grid search behavior to ensure it moves Sharpe ratio
"""

import numpy as np
import pandas as pd
import unittest
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class TestThresholdGridBehavior(unittest.TestCase):
    """Test that threshold grid search produces meaningful Sharpe variations"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Generate synthetic probabilities and returns
        n_samples = 1000
        
        # Create realistic probabilities (not all 0.5)
        base_probs = np.random.beta(2, 2, n_samples)  # Beta distribution centered around 0.5
        
        # Add some signal
        signal_strength = 0.3
        base_probs = base_probs * (1 - signal_strength) + 0.5 * signal_strength
        
        # Create returns with some predictability
        returns = np.random.normal(0.001, 0.02, n_samples)  # Daily returns
        
        # Add signal: higher probabilities should predict positive returns
        signal_component = (base_probs - 0.5) * 0.01  # Small signal
        returns += signal_component
        
        # Add some noise to make it realistic
        returns += np.random.normal(0, 0.005, n_samples)
        
        self.probabilities = base_probs
        self.returns = returns
        self.n_samples = n_samples
        
    def test_threshold_grid_variation(self):
        """Test that threshold grid search produces varying Sharpe ratios"""
        # Define threshold grid
        thresholds = np.arange(0.50, 0.71, 0.01)
        
        # Calculate Sharpe ratio for each threshold
        sharpe_scores = []
        
        for tau in thresholds:
            # Convert probabilities to binary predictions
            predictions = (self.probabilities > tau).astype(int)
            
            # Calculate strategy returns
            strategy_returns = predictions * self.returns
            
            # Calculate Sharpe ratio
            if np.std(strategy_returns) > 0:
                sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
                sharpe_scores.append(sharpe)
            else:
                sharpe_scores.append(0)
                
        # Verify that Sharpe ratios vary across thresholds
        sharpe_std = np.std(sharpe_scores)
        self.assertGreater(sharpe_std, 0.01, "Sharpe ratios should vary across thresholds")
        
        # Verify that we have a range of Sharpe values
        sharpe_range = max(sharpe_scores) - min(sharpe_scores)
        self.assertGreater(sharpe_range, 0.1, "Sharpe ratio range should be meaningful")
        
        # Find best threshold
        best_idx = np.argmax(sharpe_scores)
        best_threshold = thresholds[best_idx]
        best_sharpe = sharpe_scores[best_idx]
        
        print(f"Best threshold: {best_threshold:.2f}, Best Sharpe: {best_sharpe:.3f}")
        print(f"Sharpe variation: {sharpe_std:.3f}, Range: {sharpe_range:.3f}")
        
    def test_threshold_0_50_suspicious_check(self):
        """Test that threshold of exactly 0.50 is flagged as suspicious"""
        # Create a scenario where threshold 0.50 might be suspicious
        # (e.g., poor calibration, class imbalance)
        
        # Test case 1: Well-calibrated probabilities (should work fine)
        well_calibrated_probs = self.probabilities.copy()
        well_calibrated_returns = self.returns.copy()
        
        # Test case 2: Poorly calibrated probabilities (suspicious)
        poorly_calibrated_probs = np.ones(self.n_samples) * 0.5  # All 0.5
        
        # Test both cases
        test_cases = [
            ("well_calibrated", well_calibrated_probs, well_calibrated_returns),
            ("poorly_calibrated", poorly_calibrated_probs, well_calibrated_returns)
        ]
        
        for case_name, probs, returns in test_cases:
            with self.subTest(case=case_name):
                thresholds = np.arange(0.50, 0.71, 0.01)
                sharpe_scores = []
                
                for tau in thresholds:
                    predictions = (probs > tau).astype(int)
                    strategy_returns = predictions * returns
                    
                    if np.std(strategy_returns) > 0:
                        sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
                        sharpe_scores.append(sharpe)
                    else:
                        sharpe_scores.append(0)
                        
                # Check if threshold 0.50 is suspicious
                threshold_50_idx = 0  # First threshold is 0.50
                threshold_50_sharpe = sharpe_scores[threshold_50_idx]
                
                # Calculate variation around threshold 0.50
                nearby_sharpe_std = np.std(sharpe_scores[:5])  # First 5 thresholds
                
                if case_name == "well_calibrated":
                    # Well-calibrated should show variation
                    self.assertGreater(nearby_sharpe_std, 0.01)
                    print(f"{case_name}: Good variation around 0.50 (std={nearby_sharpe_std:.3f})")
                else:
                    # Poorly calibrated should show little variation
                    self.assertLess(nearby_sharpe_std, 0.01)
                    print(f"{case_name}: Suspicious - little variation around 0.50 (std={nearby_sharpe_std:.3f})")
                    
    def test_flat_sharpe_curve_detection(self):
        """Test detection of flat Sharpe ratio curves"""
        # Create different probability distributions to test
        
        # Case 1: Probabilities with clear signal
        clear_signal_probs = np.random.normal(0.6, 0.1, self.n_samples)
        clear_signal_probs = np.clip(clear_signal_probs, 0, 1)
        
        # Case 2: Probabilities clustered around 0.5 (flat curve)
        flat_curve_probs = np.ones(self.n_samples) * 0.5  # Perfectly flat
        
        test_cases = [
            ("clear_signal", clear_signal_probs),
            ("flat_curve", flat_curve_probs)
        ]
        
        for case_name, probs in test_cases:
            with self.subTest(case=case_name):
                thresholds = np.arange(0.50, 0.71, 0.01)
                sharpe_scores = []
                
                for tau in thresholds:
                    predictions = (probs > tau).astype(int)
                    strategy_returns = predictions * self.returns
                    
                    if np.std(strategy_returns) > 0:
                        sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
                        sharpe_scores.append(sharpe)
                    else:
                        sharpe_scores.append(0)
                        
                # Calculate curve characteristics
                sharpe_std = np.std(sharpe_scores)
                sharpe_range = max(sharpe_scores) - min(sharpe_scores)
                
                # Check for flat curve
                is_flat = sharpe_std < 0.001 or sharpe_range < 0.01
                
                if case_name == "clear_signal":
                    self.assertFalse(is_flat, "Clear signal should produce varying Sharpe ratios")
                    print(f"{case_name}: Good variation (std={sharpe_std:.3f}, range={sharpe_range:.3f})")
                else:
                    self.assertTrue(is_flat, "Flat curve should be detected")
                    print(f"{case_name}: Flat curve detected (std={sharpe_std:.3f}, range={sharpe_range:.3f})")
                    
    def test_threshold_optimization_improvement(self):
        """Test that threshold optimization actually improves performance"""
        # Use the original probabilities
        probs = self.probabilities
        returns = self.returns
        
        # Test different thresholds
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
        threshold_performance = {}
        
        for tau in thresholds:
            predictions = (probs > tau).astype(int)
            strategy_returns = predictions * returns
            
            if np.std(strategy_returns) > 0:
                sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
                threshold_performance[tau] = sharpe
            else:
                threshold_performance[tau] = 0
                
        # Find best threshold
        best_threshold = max(threshold_performance, key=threshold_performance.get)
        best_sharpe = threshold_performance[best_threshold]
        
        # Check that best threshold is better than 0.50
        baseline_sharpe = threshold_performance[0.50]
        
        if best_threshold != 0.50:
            improvement = best_sharpe - baseline_sharpe
            self.assertGreater(improvement, 0, "Optimization should improve performance")
            print(f"Threshold optimization improved Sharpe by {improvement:.3f}")
            print(f"Best threshold: {best_threshold}, Best Sharpe: {best_sharpe:.3f}")
        else:
            print("Warning: Threshold 0.50 is optimal - check calibration and class balance")
            
    def test_threshold_grid_resolution(self):
        """Test that threshold grid resolution is appropriate"""
        # Test different grid resolutions
        fine_grid = np.arange(0.50, 0.71, 0.01)    # 0.01 step
        coarse_grid = np.arange(0.50, 0.71, 0.05)  # 0.05 step
        
        probs = self.probabilities
        returns = self.returns
        
        # Calculate Sharpe for both grids
        fine_sharpe = []
        coarse_sharpe = []
        
        for tau in fine_grid:
            predictions = (probs > tau).astype(int)
            strategy_returns = predictions * returns
            if np.std(strategy_returns) > 0:
                sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
                fine_sharpe.append(sharpe)
            else:
                fine_sharpe.append(0)
                
        for tau in coarse_grid:
            predictions = (probs > tau).astype(int)
            strategy_returns = predictions * returns
            if np.std(strategy_returns) > 0:
                sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
                coarse_sharpe.append(sharpe)
            else:
                coarse_sharpe.append(0)
                
        # Fine grid should capture more variation
        fine_std = np.std(fine_sharpe)
        coarse_std = np.std(coarse_sharpe)
        
        # Fine grid should generally show more variation (more granular)
        # But this isn't always true due to noise, so we just check they're reasonable
        self.assertGreater(fine_std, 0.001, "Fine grid should show some variation")
        self.assertGreater(coarse_std, 0.001, "Coarse grid should show some variation")
        
        print(f"Fine grid std: {fine_std:.3f}, Coarse grid std: {coarse_std:.3f}")
        
    def test_threshold_edge_cases(self):
        """Test threshold behavior at edge cases"""
        probs = self.probabilities
        returns = self.returns
        
        # Test very low and very high thresholds
        edge_thresholds = [0.01, 0.10, 0.90, 0.99]
        
        for tau in edge_thresholds:
            with self.subTest(threshold=tau):
                predictions = (probs > tau).astype(int)
                strategy_returns = predictions * returns
                
                # Very low threshold should predict mostly 1s
                # Very high threshold should predict mostly 0s
                prediction_rate = np.mean(predictions)
                
                if tau < 0.1:
                    self.assertGreater(prediction_rate, 0.8, f"Low threshold {tau} should predict mostly 1s")
                elif tau > 0.9:
                    self.assertLess(prediction_rate, 0.2, f"High threshold {tau} should predict mostly 0s")
                    
                # Calculate Sharpe if possible
                if np.std(strategy_returns) > 0:
                    sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
                    print(f"Threshold {tau}: prediction_rate={prediction_rate:.3f}, sharpe={sharpe:.3f}")
                    
    def test_threshold_robustness_to_noise(self):
        """Test that threshold optimization is robust to noise"""
        # Add noise to probabilities and see if optimal threshold changes
        probs = self.probabilities
        returns = self.returns
        
        # Original optimal threshold
        thresholds = np.arange(0.50, 0.71, 0.01)
        original_sharpe = []
        
        for tau in thresholds:
            predictions = (probs > tau).astype(int)
            strategy_returns = predictions * returns
            if np.std(strategy_returns) > 0:
                sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
                original_sharpe.append(sharpe)
            else:
                original_sharpe.append(0)
                
        original_best_idx = np.argmax(original_sharpe)
        original_best_threshold = thresholds[original_best_idx]
        
        # Add noise to probabilities
        noisy_probs = probs + np.random.normal(0, 0.05, len(probs))
        noisy_probs = np.clip(noisy_probs, 0, 1)
        
        # Find optimal threshold with noisy probabilities
        noisy_sharpe = []
        
        for tau in thresholds:
            predictions = (noisy_probs > tau).astype(int)
            strategy_returns = predictions * returns
            if np.std(strategy_returns) > 0:
                sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
                noisy_sharpe.append(sharpe)
            else:
                noisy_sharpe.append(0)
                
        noisy_best_idx = np.argmax(noisy_sharpe)
        noisy_best_threshold = thresholds[noisy_best_idx]
        
        # Threshold should be reasonably stable (within 4-5 steps)
        threshold_difference = abs(noisy_best_idx - original_best_idx)
        self.assertLess(threshold_difference, 6, "Optimal threshold should be stable to noise")
        
        print(f"Original best threshold: {original_best_threshold:.2f}")
        print(f"Noisy best threshold: {noisy_best_threshold:.2f}")
        print(f"Threshold difference: {threshold_difference} steps")

if __name__ == '__main__':
    unittest.main()
