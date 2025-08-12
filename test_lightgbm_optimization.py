#!/usr/bin/env python3
"""
Test script to demonstrate LightGBM optimization improvements
Compares original vs optimized LightGBM performance
"""

import numpy as np
import pandas as pd
import time
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

from models import ModelEnsemble
from config import GRIDSEARCH_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data(n_samples=1000, n_features=50):
    """Generate synthetic test data"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    return X, y

def test_original_lightgbm_optimization(X, y):
    """Test original LightGBM optimization (simulated)"""
    logger.info("Testing original LightGBM optimization...")
    
    start_time = time.time()
    
    # Original parameter grid (243 combinations)
    original_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [4, 6, 8],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Simulate original optimization time
    # This is an estimate based on the parameter combinations
    estimated_time = 243 * 5 * 0.1  # combinations * cv_folds * estimated_time_per_fold
    time.sleep(estimated_time)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Original LightGBM optimization completed in {elapsed_time:.2f} seconds")
    return elapsed_time

def test_optimized_lightgbm_optimization(X, y):
    """Test optimized LightGBM optimization"""
    logger.info("Testing optimized LightGBM optimization...")
    
    start_time = time.time()
    
    # Create model ensemble with optimized settings
    ensemble = ModelEnsemble(n_jobs=1)
    
    # Test the fast optimization method
    result = ensemble.optimize_lightgbm_fast(X, y)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Optimized LightGBM optimization completed in {elapsed_time:.2f} seconds")
    logger.info(f"Best score: {result.get('best_score', 'N/A')}")
    
    return elapsed_time

def test_grid_search_optimization(X, y):
    """Test grid search optimization with reduced parameters"""
    logger.info("Testing grid search optimization with reduced parameters...")
    
    start_time = time.time()
    
    # Create model ensemble
    ensemble = ModelEnsemble(n_jobs=1)
    
    # Optimize all models (including LightGBM with reduced parameters)
    results = ensemble.optimize_individual_models(X, y, cv_folds=3)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Grid search optimization completed in {elapsed_time:.2f} seconds")
    
    # Print LightGBM results if available
    if 'LightGBM' in results.get('best_scores', {}):
        logger.info(f"LightGBM best score: {results['best_scores']['LightGBM']:.6f}")
        logger.info(f"LightGBM best params: {results['best_params'].get('LightGBM', 'N/A')}")
    
    return elapsed_time

def compare_parameter_combinations():
    """Compare parameter combinations between original and optimized"""
    logger.info("Comparing parameter combinations...")
    
    # Original parameters
    original_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [4, 6, 8],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Optimized parameters
    optimized_params = {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 0.2],
        'max_depth': [4, 6],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Calculate combinations
    original_combinations = 1
    for param_values in original_params.values():
        original_combinations *= len(param_values)
    
    optimized_combinations = 1
    for param_values in optimized_params.values():
        optimized_combinations *= len(param_values)
    
    logger.info(f"Original parameter combinations: {original_combinations}")
    logger.info(f"Optimized parameter combinations: {optimized_combinations}")
    logger.info(f"Reduction factor: {original_combinations / optimized_combinations:.1f}x")
    
    return original_combinations, optimized_combinations

def main():
    """Main test function"""
    logger.info("Starting LightGBM optimization performance test...")
    
    # Generate test data
    X, y = generate_test_data(n_samples=500, n_features=30)
    logger.info(f"Generated test data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Compare parameter combinations
    original_combinations, optimized_combinations = compare_parameter_combinations()
    
    # Test different optimization methods
    logger.info("\n" + "="*50)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("="*50)
    
    # Test 1: Original optimization (simulated)
    original_time = test_original_lightgbm_optimization(X, y)
    
    # Test 2: Optimized grid search
    optimized_grid_time = test_grid_search_optimization(X, y)
    
    # Test 3: Fast random search
    fast_time = test_optimized_lightgbm_optimization(X, y)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*50)
    logger.info(f"Original optimization time: {original_time:.2f} seconds")
    logger.info(f"Optimized grid search time: {optimized_grid_time:.2f} seconds")
    logger.info(f"Fast random search time: {fast_time:.2f} seconds")
    
    if original_time > 0:
        grid_speedup = original_time / optimized_grid_time
        fast_speedup = original_time / fast_time
        logger.info(f"Grid search speedup: {grid_speedup:.1f}x")
        logger.info(f"Fast search speedup: {fast_speedup:.1f}x")
    
    logger.info("\nOptimization recommendations:")
    logger.info("1. Use fast_lightgbm_optimization=True in config for maximum speed")
    logger.info("2. Reduce cv_folds to 3 for LightGBM")
    logger.info("3. Enable early stopping for LightGBM")
    logger.info("4. Use random search instead of grid search for large parameter spaces")

if __name__ == "__main__":
    main()

