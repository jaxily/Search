#!/usr/bin/env python3
"""
Test script for enhanced ensemble improvements
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from ensemble package
from ensemble.trading_utils import EnhancedTradingCalculator

# Import from root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from enhanced_ensemble import EnhancedTradingEnsemble

def test_enhanced_trading_calculator():
    """Test the enhanced trading calculator"""
    print("🧪 Testing Enhanced Trading Calculator...")
    
    # Initialize calculator
    calculator = EnhancedTradingCalculator(
        cost_per_trade=0.001,
        slippage_bps=5.0,
        min_hold_days=2,
        hysteresis_buffer=0.02
    )
    
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic probabilities
    probabilities = np.random.beta(2, 2, n_samples)
    
    # Create returns with some signal
    returns = np.random.normal(0.001, 0.02, n_samples)
    signal_component = (probabilities - 0.5) * 0.01
    returns += signal_component
    
    # Test enhanced returns calculation
    results = calculator.calculate_enhanced_returns(
        probabilities=probabilities,
        returns=returns,
        threshold=0.5,
        apply_costs=True,
        apply_slippage=True,
        apply_holding_period=True
    )
    
    print(f"✅ Enhanced returns calculation successful")
    print(f"   Strategy returns std: {np.std(results['strategy_returns']):.6f}")
    print(f"   Net returns std: {np.std(results['net_returns']):.6f}")
    print(f"   Total costs: {results['costs'].sum():.6f}")
    print(f"   Total slippage: {results['slippage_costs'].sum():.6f}")
    print(f"   Annualized turnover: {results['turnover'].sum() * 252 / n_samples:.2f}")
    
    # Test robust Sharpe calculation
    sharpe_result = calculator.calculate_robust_sharpe(results['net_returns'])
    print(f"✅ Robust Sharpe calculation successful")
    print(f"   Sharpe ratio: {sharpe_result['sharpe_ratio']:.3f}")
    print(f"   Daily mean: {sharpe_result['diagnostics']['returns_mean']:.6f}")
    print(f"   Daily std: {sharpe_result['diagnostics']['returns_std']:.6f}")
    
    # Test probability histograms
    calculator.plot_probability_histograms(
        probabilities, 
        threshold=0.5,
        save_path='test_probability_histograms.png'
    )
    print(f"✅ Probability histograms created")
    
    return True

def test_enhanced_ensemble():
    """Test the enhanced ensemble with improvements"""
    print("\n🧪 Testing Enhanced Ensemble...")
    
    try:
        # Initialize enhanced ensemble
        ensemble = EnhancedTradingEnsemble(
            random_state=42,
            n_splits=3,
            cost_per_trade=0.001,
            slippage_bps=5.0,
            min_hold_days=1,
            hysteresis_buffer=0.02
        )
        
        print(f"✅ Enhanced ensemble initialized successfully")
        print(f"   Cost per trade: {ensemble.cost_per_trade:.4f}")
        print(f"   Slippage: {ensemble.slippage_bps} bps")
        print(f"   Min hold days: {ensemble.min_hold_days}")
        print(f"   Hysteresis buffer: {ensemble.hysteresis_buffer:.4f}")
        
        # Generate test data
        np.random.seed(42)
        n_samples = 500
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
        
        # Test fitting (this will test the enhanced calibration)
        print("\n🔧 Testing enhanced calibration...")
        ensemble.fit_models(X, y)
        
        # Test calibration with auto method
        ensemble.calibrate_probabilities(method='auto', plot_histograms=True)
        
        print(f"✅ Enhanced ensemble testing completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Testing Enhanced Ensemble Improvements")
    print("=" * 50)
    
    # Test trading calculator
    success1 = test_enhanced_trading_calculator()
    
    # Test enhanced ensemble
    success2 = test_enhanced_ensemble()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    if success1 and success2:
        print("✅ All tests passed successfully!")
        print("\n🎯 Enhanced features implemented:")
        print("   • Transaction costs and slippage")
        print("   • Turnover control with holding periods")
        print("   • Hysteresis buffer to prevent flip-flopping")
        print("   • Robust Sharpe ratio calculation")
        print("   • Enhanced threshold optimization")
        print("   • Automatic calibration method selection")
        print("   • Comprehensive cost impact analysis")
        print("   • Probability distribution visualization")
    else:
        print("❌ Some tests failed")
        if not success1:
            print("   • Enhanced trading calculator test failed")
        if not success2:
            print("   • Enhanced ensemble test failed")
    
    print("\n🔧 Next steps:")
    print("   1. Run the enhanced ensemble on your actual data")
    print("   2. Check the generated plots and diagnostics")
    print("   3. Verify that transaction costs are properly applied")
    print("   4. Confirm that Sharpe ratios are no longer infinite")

if __name__ == "__main__":
    main()
