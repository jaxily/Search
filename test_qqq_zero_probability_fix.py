#!/usr/bin/env python3
"""
Test script to verify zero probability fix on actual dataset
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_ensemble import EnhancedTradingEnsemble

def load_dataset():
    """Load and prepare dataset"""
    print("📊 Loading dataset...")
    
    # Load the dataset
    df = pd.read_csv('multi_ticker_dataset_20250812_161032.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Total samples: {len(df)}")
    
    return df

def prepare_features_and_targets(df):
    """Prepare features and targets"""
    print("\n🔧 Preparing features and targets...")
    
    # Find feature columns (exclude date, target, and metadata columns)
    exclude_cols = ['Date', 'Daily_Return', 'Ticker', 'Company_Name', 'Sector', 'Industry', 'Country', 'Currency', 'Exchange', 'Website', 'Filter_Pass']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove columns with too many NaN values and non-numeric columns
    nan_threshold = 0.1  # Remove columns with >10% NaN
    valid_feature_cols = []
    
    for col in feature_cols:
        # Check if column is numeric
        if df[col].dtype in ['int64', 'float64']:
            nan_ratio = df[col].isna().sum() / len(df)
            if nan_ratio < nan_threshold:
                valid_feature_cols.append(col)
            else:
                print(f"  Skipping {col}: {nan_ratio*100:.1f}% NaN")
        else:
            print(f"  Skipping {col}: non-numeric type {df[col].dtype}")
    
    print(f"Feature columns: {len(valid_feature_cols)} valid out of {len(feature_cols)} total")
    
    # Prepare data
    X = df[valid_feature_cols].values
    y_raw = df['Daily_Return'].values
    
    # Convert returns to binary targets (positive = 1, negative = 0)
    y = (y_raw > 0).astype(int)
    
    # X is already numeric, no need to convert
    
    # Remove any rows with NaN values
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]
    returns = y_raw[valid_mask]
    
    print(f"Final data shape: X={X.shape}, y={y.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Features range: {X.min():.6f} to {X.max():.6f}")
    print(f"Returns range: {returns.min():.6f} to {returns.max():.6f}")
    print(f"Returns std: {returns.std():.6f}")
    
    return X, y, returns, valid_feature_cols

def test_zero_probability_fix():
    """Test zero probability fix on actual data"""
    print("🧪 Testing Zero Probability Fix on Actual Dataset...")
    print("=" * 70)
    
    # Load data
    df = load_dataset()
    
    # Prepare features and targets
    X, y, returns, feature_cols = prepare_features_and_targets(df)
    
    print(f"\n📈 Dataset Summary:")
    print(f"   Total samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Positive returns: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    print(f"   Negative returns: {np.sum(y==0)} ({np.mean(y==0)*100:.1f}%)")
    print(f"   Returns range: {returns.min():.6f} to {returns.max():.6f}")
    print(f"   Returns std: {returns.std():.6f}")
    print(f"   Years of data: {len(X) / 252:.1f}")
    
    # Test with different configurations
    test_configs = [
        (5, "Original config (5 splits)"),
        (3, "Reduced splits (3 splits)"),
        (2, "Minimal splits (2 splits)")
    ]
    
    for n_splits, description in test_configs:
        print(f"\n🔧 Testing {description}...")
        
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
                
                if valid_count > 0:
                    valid_probs = probs[(probs > 0) & (probs < 1)]
                    print(f"  Valid prob range: {valid_probs.min():.6f} to {valid_probs.max():.6f}")
                    print(f"  Valid prob mean: {valid_probs.mean():.6f}")
                    print(f"  Valid prob std: {valid_probs.std():.6f}")
                
                # Check if we have the warm-up period
                if hasattr(ensemble, 'warm_up_samples'):
                    print(f"  Warm-up samples: {ensemble.warm_up_samples}")
                    print(f"  Warm-up percentage: {ensemble.warm_up_samples/len(probs)*100:.1f}%")
                
                # Validate that we don't have all zeros
                if zero_count == len(probs):
                    print("  ❌ CRITICAL: All probabilities are zero!")
                elif zero_count > len(probs) * 0.1:
                    print(f"  ⚠️  WARNING: {zero_count/len(probs)*100:.1f}% probabilities are zero!")
                else:
                    print(f"  ✅ Good: Only {zero_count/len(probs)*100:.1f}% probabilities are zero")
                
                # Check for valid trading signals
                if valid_count > len(probs) * 0.5:
                    print("  ✅ Good: Sufficient valid trading signals")
                else:
                    print("  ❌ CRITICAL: Insufficient valid trading signals!")
                
                # Test trading calculations
                print(f"\n  🧮 Testing trading calculations...")
                try:
                    # Test with a reasonable threshold
                    threshold = 0.5
                    trading_results = ensemble.trading_calculator.calculate_enhanced_returns(
                        probabilities=probs,
                        returns=returns,
                        threshold=threshold,
                        apply_costs=True,
                        apply_slippage=True,
                        apply_holding_period=True
                    )
                    
                    # Check for zero volatility
                    strategy_std = np.std(trading_results['strategy_returns'])
                    net_std = np.std(trading_results['net_returns'])
                    
                    print(f"    Strategy returns std: {strategy_std:.6f}")
                    print(f"    Net returns std: {net_std:.6f}")
                    
                    if strategy_std == 0:
                        print("    ❌ CRITICAL: Zero volatility in strategy returns!")
                    elif strategy_std < 1e-6:
                        print("    ⚠️  WARNING: Very low volatility in strategy returns")
                    else:
                        print("    ✅ Good: Reasonable volatility in strategy returns")
                    
                    if net_std == 0:
                        print("    ❌ CRITICAL: Zero volatility in net returns!")
                    elif net_std < 1e-6:
                        print("    ⚠️  WARNING: Very low volatility in net returns")
                    else:
                        print("    ✅ Good: Reasonable volatility in net returns")
                    
                    # Test Sharpe calculation
                    print(f"\n    📊 Testing Sharpe calculation...")
                    sharpe_result = ensemble.trading_calculator.calculate_robust_sharpe(trading_results['net_returns'])
                    
                    if 'error' in sharpe_result:
                        print(f"    ❌ Sharpe calculation error: {sharpe_result['error']}")
                    else:
                        sharpe = sharpe_result['sharpe_ratio']
                        print(f"    ✅ Sharpe ratio: {sharpe:.3f}")
                        
                        if np.isinf(sharpe):
                            print("    ❌ CRITICAL: Infinite Sharpe ratio!")
                        elif sharpe > 10:
                            print("    ⚠️  WARNING: Very high Sharpe ratio - may need investigation")
                        else:
                            print("    ✅ Good: Reasonable Sharpe ratio")
                    
                except Exception as e:
                    print(f"    ❌ Trading calculation error: {e}")
                    
            else:
                print("  ❌ No OOF probabilities generated")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("📋 ZERO PROBABILITY FIX TEST SUMMARY")
    print("=" * 70)
    
    print("✅ Expected improvements:")
    print("   • Eliminated zero probability predictions")
    print("   • Proper warm-up period handling")
    print("   • Realistic volatility in returns")
    print("   • Valid trading signals throughout dataset")
    
    print("\n🔧 Key fixes implemented:")
    print("   • Smart n_splits validation")
    print("   • Warm-up period calculation")
    print("   • NaN handling for insufficient training data")
    print("   • Enhanced trading calculator with NaN awareness")
    
    print("\n🚀 Next steps:")
    print("   1. Verify that zero probabilities are eliminated")
    print("   2. Check that warm-up period is reasonable (< 20%)")
    print("   3. Confirm that volatility is realistic")
    print("   4. Run full ensemble optimization")
    print("   5. Verify Sharpe ratios are no longer infinite")

if __name__ == "__main__":
    test_zero_probability_fix()
