#!/usr/bin/env python3
"""
Example script demonstrating the Walk-Forward Ensemble ML Trading System
This script shows how to use the system with sample data
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our system components
from data_processor import DataProcessor
from models import ModelEnsemble
from walkforward import WalkForwardAnalyzer
from performance import PerformanceAnalyzer

def create_sample_data(n_samples: int = 5000, n_features: int = 100) -> pd.DataFrame:
    """
    Create sample data for demonstration purposes
    
    Args:
        n_samples: Number of samples (rows)
        n_features: Number of features (columns)
    
    Returns:
        Sample DataFrame with price data and features
    """
    print("Creating sample data...")
    
    # Create date range
    dates = pd.date_range('2010-01-01', periods=n_samples, freq='D')
    
    # Create price data
    np.random.seed(42)
    
    # Generate random walk for price
    price_changes = np.random.normal(0, 0.01, n_samples)
    prices = np.cumprod(1 + price_changes)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.002, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=dates)
    
    # Add some nulls in first 50 columns (simulating rolling average issues)
    for i in range(50):
        col_name = f'feature_{i:03d}'
        # Create rolling average with some nulls
        rolling_data = data['close'].rolling(window=20 + i).mean()
        # Add some random nulls
        null_mask = np.random.random(len(rolling_data)) < 0.1
        rolling_data[null_mask] = np.nan
        data[col_name] = rolling_data
    
    # Add more features
    for i in range(50, n_features):
        col_name = f'feature_{i:03d}'
        if i < 70:
            # Technical indicators
            data[col_name] = data['close'].rolling(window=10 + (i % 20)).std()
        elif i < 90:
            # Momentum features
            data[col_name] = data['close'].pct_change(periods=1 + (i % 10))
        else:
            # Random features
            data[col_name] = np.random.normal(0, 1, n_samples)
    
    # Create target variable (returns)
    data['returns'] = data['close'].pct_change()
    
    print(f"Sample data created: {data.shape}")
    print(f"Null values in first 50 columns: {data.iloc[:, :50].isnull().sum().sum()}")
    
    return data

def run_demo():
    """Run the complete demonstration"""
    print("=" * 60)
    print("WALK-FORWARD ENSEMBLE ML TRADING SYSTEM DEMO")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data(n_samples=3000, n_features=150)
    
    # Save sample data
    data.to_csv('sample_data.csv')
    print("Sample data saved to 'sample_data.csv'")
    
    # Initialize data processor
    print("\n1. Initializing Data Processor...")
    processor = DataProcessor(use_numba=True)
    
    # Process data
    print("\n2. Processing Data...")
    try:
        # Clean data
        cleaned_data = processor._clean_data(data)
        print(f"Data cleaned: {cleaned_data.shape}")
        
        # Engineer features
        engineered_data = processor.engineer_features(cleaned_data)
        print(f"Features engineered: {engineered_data.shape}")
        
        # Prepare walk-forward data
        X, y = processor.prepare_walkforward_data(engineered_data, target_col='returns')
        print(f"Walk-forward data prepared: X={X.shape}, y={y.shape}")
        
    except Exception as e:
        print(f"Error in data processing: {e}")
        return
    
    # Initialize and optimize ensemble
    print("\n3. Initializing Model Ensemble...")
    try:
        ensemble = ModelEnsemble(n_jobs=2)  # Use 2 cores for demo
        
        # Create ensemble (skip optimization for demo speed)
        ensemble.create_ensemble('Voting')
        
        # Fit ensemble on small subset for demo
        subset_size = min(1000, len(X))
        X_subset = X[:subset_size]
        y_subset = y[:subset_size]
        
        ensemble.fit_ensemble(X_subset, y_subset)
        print("Ensemble model fitted successfully")
        
    except Exception as e:
        print(f"Error in ensemble creation: {e}")
        return
    
    # Run walk-forward analysis (simplified for demo)
    print("\n4. Running Walk-Forward Analysis...")
    try:
        analyzer = WalkForwardAnalyzer(n_jobs=2)
        
        # Use smaller parameters for demo
        analyzer.initial_train_size = 500
        analyzer.step_size = 100
        analyzer.min_train_size = 200
        
        # Run analysis
        results = analyzer.run_walkforward_analysis(X, y, ensemble_method='Voting')
        print("Walk-forward analysis completed")
        
        # Get performance summary
        summary = analyzer.get_performance_summary()
        print(f"Analysis completed for {summary.get('n_windows', 0)} windows")
        
    except Exception as e:
        print(f"Error in walk-forward analysis: {e}")
        return
    
    # Performance analysis
    print("\n5. Performance Analysis...")
    try:
        perf_analyzer = PerformanceAnalyzer()
        
        # Calculate metrics for a subset
        subset_size = min(500, len(y))
        y_actual = y[:subset_size]
        y_pred = ensemble.predict(X[:subset_size])
        
        metrics = perf_analyzer.calculate_metrics(y_actual, y_pred)
        
        # Generate report
        report = perf_analyzer.generate_performance_report(metrics)
        print("\nPERFORMANCE REPORT:")
        print(report)
        
        # Save report
        with open('demo_performance_report.txt', 'w') as f:
            f.write(report)
        print("\nPerformance report saved to 'demo_performance_report.txt'")
        
    except Exception as e:
        print(f"Error in performance analysis: {e}")
        return
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated files:")
    print("- sample_data.csv: Sample dataset")
    print("- demo_performance_report.txt: Performance analysis")
    print("\nTo run the full system:")
    print("python main.py --data-file sample_data.csv --ensemble-method Voting")

if __name__ == "__main__":
    run_demo()

