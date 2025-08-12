#!/usr/bin/env python3
"""
Simple Ensemble Test
Tests basic ensemble functionality without complex TimeSeriesSplit
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from enhanced_ensemble import EnhancedTradingEnsemble

def main():
    """Test basic ensemble functionality"""
    print("🚀 Testing Enhanced Trading Ensemble...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 500  # Larger dataset for better TimeSeriesSplit
    n_features = 20
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with some predictability
    signal = np.dot(X[:, :8], np.random.randn(8)) * 0.15
    noise = np.random.randn(n_samples) * 0.02
    returns = signal + noise
    y = (returns > 0).astype(int)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Target distribution: {np.mean(y):.3f} positive, {1-np.mean(y):.3f} negative")
    
    # Initialize ensemble with fewer splits for small dataset
    ensemble = EnhancedTradingEnsemble(random_state=42, n_splits=3)
    
    try:
        # Fit models
        print("\n📊 Fitting models...")
        ensemble.fit_models(X, y)
        
        print(f"✅ Models fitted successfully. Active models: {len(ensemble.models)}")
        print(f"Active models: {list(ensemble.models.keys())}")
        
        if len(ensemble.oof_probabilities) > 0:
            print(f"✅ OOF probabilities generated for {len(ensemble.oof_probabilities)} models")
            
            # Test basic prediction
            print("\n🔮 Testing predictions...")
            predictions = ensemble.predict_proba(X[:10])  # Test on first 10 samples
            print(f"✅ Predictions shape: {predictions.shape}")
            print(f"Sample predictions: {predictions[:5]}")
            
            # Test weight optimization
            print("\n⚖️ Testing weight optimization...")
            results = ensemble.optimize_ensemble_weights(
                y=y, method='sharpe', cost_per_trade=0.001, slippage=0.0005
            )
            
            print(f"✅ Weight optimization successful")
            print(f"Optimal weights: {results['optimal_weights']}")
            print(f"Optimal threshold: {results['optimal_threshold']:.3f}")
            
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                print(f"Sharpe ratio: {metrics.get('sharpe_ratio', 'N/A'):.3f}")
                print(f"Annualized return: {metrics.get('annualized_return', 'N/A'):.3f}")
            
            # Test diversity control
            print("\n🌐 Testing diversity control...")
            diversity_weights = ensemble.apply_diversity_control(correlation_threshold=0.95)
            print(f"✅ Diversity control applied. Weights: {diversity_weights}")
            
            print("\n🎉 All tests passed! Ensemble is working correctly.")
            
        else:
            print("❌ No OOF probabilities generated. Check TimeSeriesSplit implementation.")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
