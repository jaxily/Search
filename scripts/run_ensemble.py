#!/usr/bin/env python3
"""
Top-level entry script for running the 9-model ensemble end-to-end
Optimized for trading Sharpe ratio and CAGR with no data leakage
"""

import argparse
import sys
import os
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from enhanced_ensemble import EnhancedTradingEnsemble
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Enhanced Trading Ensemble')
    
    parser.add_argument('--data_path', type=str, default='data/processed_data.parquet',
                       help='Path to processed data file')
    parser.add_argument('--output_dir', type=str, default='results/ensemble',
                       help='Output directory for results')
    parser.add_argument('--transaction_cost', type=float, default=0.001,
                       help='Transaction cost per trade')
    parser.add_argument('--slippage_bps', type=float, default=0.5,
                       help='Slippage in basis points')
    parser.add_argument('--regime_aware', action='store_true',
                       help='Enable regime-aware weights')
    parser.add_argument('--tau_grid_start', type=float, default=0.50,
                       help='Start of threshold grid search')
    parser.add_argument('--tau_grid_stop', type=float, default=0.71,
                       help='End of threshold grid search')
    parser.add_argument('--tau_grid_step', type=float, default=0.01,
                       help='Step size for threshold grid search')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='Number of TimeSeriesSplit folds')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--optimization_method', type=str, default='sharpe',
                       choices=['sharpe', 'cagr', 'sharpe_cagr'],
                       help='Weight optimization method')
    
    return parser.parse_args()

def load_data(data_path: str):
    """Load and prepare data"""
    logger.info(f"Loading data from {data_path}")
    
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Remove non-numeric columns (like Date, Ticker)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    logger.info(f"Found {len(numeric_columns)} numeric columns: {list(numeric_columns)}")
    
    # Use numeric columns only
    df_numeric = df[numeric_columns]
    
    # Assume last column is target, rest are features
    X = df_numeric.iloc[:, :-1].values
    y = df_numeric.iloc[:, -1].values
    
    # Convert target to binary if needed
    if len(np.unique(y)) > 2:
        # Assume it's returns, convert to binary
        y = (y > 0).astype(int)
        logger.info("Converted target to binary classification")
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    logger.info(f"Target distribution: {np.mean(y):.3f} positive, {1-np.mean(y):.3f} negative")
    
    return X, y

def run_ensemble_workflow(X, y, args):
    """Run the complete ensemble workflow"""
    logger.info("="*60)
    logger.info("🚀 STARTING ENHANCED ENSEMBLE WORKFLOW")
    logger.info("="*60)
    
    # Step 1: Initialize ensemble
    logger.info("Step 1: Initializing Enhanced Trading Ensemble...")
    ensemble = EnhancedTradingEnsemble(
        random_state=args.random_state, 
        n_splits=args.n_splits
    )
    
    # Step 2: Fit models with TimeSeriesSplit
    logger.info("Step 2: Fitting models using TimeSeriesSplit...")
    ensemble.fit_models(X, y)
    
    # Step 3: Calibrate probabilities
    logger.info("Step 3: Calibrating probabilities...")
    ensemble.calibrate_probabilities(method='isotonic')
    
    # Step 4: Optimize ensemble weights
    logger.info("Step 4: Optimizing ensemble weights...")
    weight_results = ensemble.optimize_ensemble_weights(
        y=y,
        method=args.optimization_method,
        cost_per_trade=args.transaction_cost,
        slippage=args.slippage_bps / 10000  # Convert bps to decimal
    )
    
    # Step 5: Apply diversity control
    logger.info("Step 5: Applying diversity control...")
    diversity_weights = ensemble.apply_diversity_control(correlation_threshold=0.95)
    
    # Step 6: Fit regime-aware weights if enabled
    if args.regime_aware:
        logger.info("Step 6: Fitting regime-aware weights...")
        regime_weights = ensemble.fit_regime_aware_weights(X, y)
    
    # Step 7: Evaluate performance
    logger.info("Step 7: Evaluating ensemble performance...")
    individual_metrics = ensemble.evaluate_individual_models(X, y)
    diversity_metrics = ensemble.get_model_diversity_metrics()
    
    # Step 8: Save artifacts
    logger.info("Step 8: Saving artifacts...")
    output_dir = Path(args.output_dir)
    artifacts = ensemble.save_artifacts(output_dir, y)
    
    # Step 9: Generate summary
    logger.info("Step 9: Generating summary...")
    summary = ensemble.get_ensemble_summary()
    
    # Print results
    logger.info("="*60)
    logger.info("📊 ENSEMBLE RESULTS SUMMARY")
    logger.info("="*60)
    
    logger.info(f"Active models: {len(ensemble.models)}")
    logger.info(f"Optimal threshold: {ensemble.optimal_threshold:.3f}")
    logger.info(f"Optimization method: {args.optimization_method}")
    
    if weight_results['performance_metrics']:
        metrics = weight_results['performance_metrics']
        logger.info(f"Annualized Sharpe: {metrics.get('sharpe_ratio', 'N/A'):.3f}")
        logger.info(f"Annualized Return: {metrics.get('annualized_return', 'N/A'):.3f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.3f}")
        logger.info(f"Trade Count: {metrics.get('trade_count', 'N/A')}")
    
    logger.info(f"Artifacts saved to: {output_dir}")
    
    return ensemble, weight_results, artifacts

def main():
    """Main function"""
    args = parse_arguments()
    
    try:
        # Load data
        X, y = load_data(args.data_path)
        
        # Run ensemble workflow
        ensemble, results, artifacts = run_ensemble_workflow(X, y, args)
        
        logger.info("✅ Ensemble workflow completed successfully!")
        
        # Save ensemble model
        ensemble_path = Path(args.output_dir) / "ensemble_model.pkl"
        ensemble.save_ensemble(str(ensemble_path))
        logger.info(f"Ensemble model saved to: {ensemble_path}")
        
    except Exception as e:
        logger.error(f"❌ Ensemble workflow failed: {e}")
        raise

if __name__ == "__main__":
    main()
