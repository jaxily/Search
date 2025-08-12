#!/usr/bin/env python3
"""
Demonstration of Enhanced Ensemble Trading System
Shows all the new features: transaction costs, turnover control, robust Sharpe, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_ensemble import EnhancedTradingEnsemble
from ensemble.trading_utils import EnhancedTradingCalculator

def create_demo_data(n_samples=1000, n_features=20):
    """Create realistic demo data for trading"""
    np.random.seed(42)
    
    # Create features with some signal
    X = np.random.randn(n_samples, n_features)
    
    # Add some trend and momentum features
    X[:, 0] = np.cumsum(np.random.randn(n_samples) * 0.01)  # Trend
    X[:, 1] = np.roll(X[:, 0], 1)  # Lagged trend
    X[:, 2] = np.roll(X[:, 0], 5)  # Momentum
    
    # Create returns with signal
    signal_strength = 0.02
    base_returns = np.random.normal(0.001, 0.02, n_samples)
    
    # Add signal based on features
    signal = np.tanh(X[:, 0] * 0.1 + X[:, 1] * 0.05 + X[:, 2] * 0.03)
    returns = base_returns + signal * signal_strength
    
    # Convert to binary targets (positive return = 1)
    y = (returns > 0).astype(int)
    
    return X, y, returns

def demonstrate_enhanced_features():
    """Demonstrate all enhanced features"""
    print("🚀 Enhanced Ensemble Trading System Demo")
    print("=" * 60)
    
    # Create demo data
    print("📊 Creating demo data...")
    X, y, returns = create_demo_data(n_samples=1000, n_features=20)
    print(f"   Data shape: {X.shape}")
    print(f"   Target distribution: {np.bincount(y)}")
    print(f"   Returns: mean={returns.mean():.6f}, std={returns.std():.6f}")
    
    # Initialize enhanced ensemble with different configurations
    print("\n🔧 Initializing Enhanced Ensemble...")
    
    # Configuration 1: Low costs, aggressive trading
    ensemble_low_cost = EnhancedTradingEnsemble(
        random_state=42,
        n_splits=3,
        cost_per_trade=0.0005,    # 0.05% per trade
        slippage_bps=2.0,         # 2 basis points
        min_hold_days=1,          # 1 day minimum
        hysteresis_buffer=0.01    # 1% buffer
    )
    
    # Configuration 2: High costs, conservative trading
    ensemble_high_cost = EnhancedTradingEnsemble(
        random_state=42,
        n_splits=3,
        cost_per_trade=0.002,     # 0.2% per trade
        slippage_bps=10.0,        # 10 basis points
        min_hold_days=3,          # 3 days minimum
        hysteresis_buffer=0.05    # 5% buffer
    )
    
    print("✅ Two ensemble configurations created:")
    print(f"   Low Cost: {ensemble_low_cost.cost_per_trade:.4f} cost, {ensemble_low_cost.slippage_bps} bps slippage")
    print(f"   High Cost: {ensemble_high_cost.cost_per_trade:.4f} cost, {ensemble_high_cost.slippage_bps} bps slippage")
    
    # Fit models
    print("\n🏗️  Fitting models...")
    ensemble_low_cost.fit_models(X, y)
    ensemble_high_cost.fit_models(X, y)
    
    # Enhanced calibration with auto method
    print("\n🎯 Testing enhanced calibration...")
    ensemble_low_cost.calibrate_probabilities(method='auto', plot_histograms=True)
    ensemble_high_cost.calibrate_probabilities(method='auto', plot_histograms=True)
    
    # Optimize weights and thresholds
    print("\n⚡ Optimizing ensemble weights and thresholds...")
    
    # Low cost ensemble
    results_low_cost = ensemble_low_cost.optimize_ensemble_weights(
        y=returns,
        method='sharpe'
    )
    
    # High cost ensemble
    results_high_cost = ensemble_high_cost.optimize_ensemble_weights(
        y=returns,
        method='sharpe'
    )
    
    # Compare results
    print("\n📈 Performance Comparison:")
    print("=" * 60)
    
    print("Low Cost Configuration:")
    metrics_low = results_low_cost['performance_metrics']
    print(f"   Sharpe Ratio: {metrics_low['sharpe_ratio']:.3f}")
    print(f"   Sharpe without costs: {metrics_low['sharpe_without_costs']:.3f}")
    print(f"   Sharpe degradation: {metrics_low['sharpe_degradation']:.3f}")
    print(f"   Cost drag: {metrics_low['cost_drag_pct']:.2f}%")
    print(f"   Annualized costs: {metrics_low['annualized_costs']:.6f}")
    print(f"   Optimal threshold: {results_low_cost['optimal_threshold']:.3f}")
    
    print("\nHigh Cost Configuration:")
    metrics_high = results_high_cost['performance_metrics']
    print(f"   Sharpe Ratio: {metrics_high['sharpe_ratio']:.3f}")
    print(f"   Sharpe without costs: {metrics_high['sharpe_without_costs']:.3f}")
    print(f"   Sharpe degradation: {metrics_high['sharpe_degradation']:.3f}")
    print(f"   Cost drag: {metrics_high['cost_drag_pct']:.2f}%")
    print(f"   Annualized costs: {metrics_high['annualized_costs']:.6f}")
    print(f"   Optimal threshold: {results_high_cost['optimal_threshold']:.3f}")
    
    # Demonstrate diversity control
    print("\n🌐 Testing diversity control...")
    diversity_low = ensemble_low_cost.get_model_diversity_metrics()
    diversity_high = ensemble_high_cost.get_model_diversity_metrics()
    
    print(f"Low Cost Diversity Score: {diversity_low['diversity_score']:.3f}")
    print(f"High Cost Diversity Score: {diversity_high['diversity_score']:.3f}")
    
    # Apply diversity control
    adjusted_weights_low = ensemble_low_cost.apply_diversity_control(correlation_threshold=0.9)
    adjusted_weights_high = ensemble_high_cost.apply_diversity_control(correlation_threshold=0.9)
    
    print(f"Models after diversity control (Low Cost): {len(adjusted_weights_low)}")
    print(f"Models after diversity control (High Cost): {len(adjusted_weights_high)}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("✅ Enhanced features demonstrated:")
    print("   • Transaction costs and slippage")
    print("   • Turnover control with holding periods")
    print("   • Hysteresis buffer implementation")
    print("   • Robust Sharpe ratio calculation")
    print("   • Enhanced threshold optimization")
    print("   • Automatic calibration method selection")
    print("   • Comprehensive cost impact analysis")
    print("   • Probability distribution visualization")
    print("   • Diversity control and correlation management")
    
    print("\n📊 Key insights from demo:")
    print(f"   • Cost impact: Low cost vs High cost configurations")
    print(f"   • Threshold optimization: {results_low_cost['optimal_threshold']:.3f} vs {results_high_cost['optimal_threshold']:.3f}")
    print(f"   • Sharpe degradation: {metrics_low['sharpe_degradation']:.3f} vs {metrics_high['sharpe_degradation']:.3f}")
    
    print("\n🔧 Next steps:")
    print("   1. Run on your actual QQQ data")
    print("   2. Adjust cost parameters for your broker")
    print("   3. Fine-tune holding periods and hysteresis")
    print("   4. Monitor cost impact and performance")
    
    return {
        'low_cost': results_low_cost,
        'high_cost': results_high_cost,
        'diversity_low': diversity_low,
        'diversity_high': diversity_high
    }

def main():
    """Main demo function"""
    try:
        results = demonstrate_enhanced_features()
        print("\n✅ Demo completed successfully!")
        return results
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
