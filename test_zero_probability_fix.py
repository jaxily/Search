#!/usr/bin/env python3
"""
Test script to verify zero probability fix
"""

import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_ensemble import EnhancedTradingEnsemble

def test_zero_probability_fix():
    """Test that the zero probability issue is fixed"""
    print("🧪 Testing Zero Probability Fix...")
    
    # Create test data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    print(f"Data shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Test with different n_splits configurations
    test_configs = [
        (5, "Original problematic config"),
        (3, "Reduced splits config"),
        (2, "Minimal splits config")
    ]
    
    for n_splits, description in test_configs:
        print(f"\n🔧 Testing {description} (n_splits={n_splits})...")
        
        try:
            # Initialize ensemble
            ensemble = EnhancedTradingEnsemble(
                random_state=42,
                n_splits=n_splits,
                cost_per_trade=0.001,
                slippage_bps=5.0
            )
            
            # Fit models
            ensemble.fit_models(X, y)
            
            # Check OOF probabilities
            if ensemble.oof_probabilities:
                # Get first model's probabilities
                first_model = list(ensemble.oof_probabilities.keys())[0]
                probs = ensemble.oof_probabilities[first_model]
                
                # Check for zero probabilities
                zero_count = np.sum(probs == 0)
                nan_count = np.sum(np.isnan(probs))
                valid_count = np.sum((probs > 0) & (probs < 1))
                
                print(f"  Model: {first_model}")
                print(f"  Total samples: {len(probs)}")
                print(f"  Zero probabilities: {zero_count}")
                print(f"  NaN probabilities: {nan_count}")
                print(f"  Valid probabilities: {valid_count}")
                print(f"  Probability range: {probs.min():.6f} to {probs.max():.6f}")
                
                # Check if we have the warm-up period
                if hasattr(ensemble, 'warm_up_samples'):
                    print(f"  Warm-up samples: {ensemble.warm_up_samples}")
                    print(f"  Warm-up percentage: {ensemble.warm_up_samples/len(probs)*100:.1f}%")
                
                # Validate that we don't have all zeros
                if zero_count == len(probs):
                    print("  ❌ CRITICAL: All probabilities are zero!")
                elif zero_count > len(probs) * 0.5:
                    print("  ⚠️  WARNING: More than 50% probabilities are zero!")
                else:
                    print("  ✅ Good: Reasonable probability distribution")
                
                # Check for valid trading signals
                if valid_count > 0:
                    print("  ✅ Good: Valid trading signals available")
                else:
                    print("  ❌ CRITICAL: No valid trading signals!")
                    
            else:
                print("  ❌ No OOF probabilities generated")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("📋 ZERO PROBABILITY FIX TEST SUMMARY")
    print("=" * 50)
    
    print("✅ Expected improvements:")
    print("   • Reduced n_splits for smaller datasets")
    print("   • Warm-up period calculation")
    print("   • NaN handling for insufficient training data")
    print("   • Minimum fold size validation")
    
    print("\n🔧 Key fixes implemented:")
    print("   • _validate_n_splits: Ensures minimum fold size")
    print("   • _calculate_warm_up_period: Prevents zero predictions")
    print("   • _generate_oof_probabilities_manual: Handles warm-up period")
    print("   • Enhanced trading calculator: NaN-aware calculations")
    
    print("\n🚀 Next steps:")
    print("   1. Run this test on your actual QQQ data")
    print("   2. Verify that zero probabilities are eliminated")
    print("   3. Check that warm-up period is reasonable")
    print("   4. Confirm that Sharpe ratios are no longer infinite")

if __name__ == "__main__":
    test_zero_probability_fix()
