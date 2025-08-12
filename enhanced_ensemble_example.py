#!/usr/bin/env python3
"""
Enhanced Ensemble Trading Example
Demonstrates the complete workflow for the 9-model soft-voting ensemble
Optimized for trading Sharpe ratio and CAGR with probability-based position sizing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Import the enhanced ensemble
from enhanced_ensemble import EnhancedTradingEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_trading_data(n_samples=2000, n_features=30, random_state=42):
    """
    Create synthetic trading data for demonstration
    
    Args:
        n_samples: Number of samples
        n_features: Number of features (technical indicators)
        random_state: Random seed for reproducibility
    
    Returns:
        X: Feature matrix
        y: Target returns (binary: 1 for positive, 0 for negative)
        dates: Datetime index
    """
    np.random.seed(random_state)
    
    # Generate features (technical indicators)
    # Simulate realistic technical indicators with some autocorrelation
    X = np.random.randn(n_samples, n_features)
    
    # Add autocorrelation to features (realistic for financial data)
    for i in range(n_features):
        if i % 3 == 0:  # Every third feature has autocorrelation
            for j in range(1, n_samples):
                X[j, i] = 0.7 * X[j-1, i] + 0.3 * X[j, i]
    
    # Generate target returns with some predictability
    # Use first 10 features as signal, rest as noise
    signal_features = X[:, :10]
    signal_weights = np.random.randn(10) * 0.1
    
    # Create signal component
    signal = np.dot(signal_features, signal_weights)
    
    # Add trend and volatility clustering
    trend = np.cumsum(np.random.randn(n_samples) * 0.001)
    volatility = np.exp(np.cumsum(np.random.randn(n_samples) * 0.01))
    
    # Combine components
    returns = signal + trend + np.random.randn(n_samples) * 0.02 * volatility
    
    # Convert to binary classification (positive/negative returns)
    y = (returns > 0).astype(int)
    
    # Create datetime index
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    logger.info(f"Created synthetic data: {n_samples} samples, {n_features} features")
    logger.info(f"Target distribution: {np.mean(y):.3f} positive, {1-np.mean(y):.3f} negative")
    
    return X, y, dates

def run_enhanced_ensemble_workflow(X, y, dates, cost_per_trade=0.001, slippage=0.0005):
    """
    Run the complete enhanced ensemble workflow
    
    Args:
        X: Feature matrix
        y: Target values
        dates: Datetime index
        cost_per_trade: Transaction cost per trade
        slippage: Slippage cost per trade
    
    Returns:
        ensemble: Trained ensemble
        results: Complete results dictionary
    """
    logger.info("="*60)
    logger.info("🚀 STARTING ENHANCED ENSEMBLE WORKFLOW")
    logger.info("="*60)
    
    # Step 1: Initialize ensemble
    logger.info("Step 1: Initializing Enhanced Trading Ensemble...")
    ensemble = EnhancedTradingEnsemble(random_state=42, n_splits=5)
    
    # Step 2: Fit all models using TimeSeriesSplit
    logger.info("Step 2: Fitting models using TimeSeriesSplit...")
    ensemble.fit_models(X, y)
    
    # Step 3: Calibrate probabilities
    logger.info("Step 3: Calibrating probabilities...")
    ensemble.calibrate_probabilities(method='isotonic')
    
    # Step 4: Optimize ensemble weights for Sharpe ratio
    logger.info("Step 4: Optimizing ensemble weights for Sharpe ratio...")
    sharpe_results = ensemble.optimize_ensemble_weights(
        y, method='sharpe', cost_per_trade=cost_per_trade, slippage=slippage
    )
    
    # Step 5: Apply diversity control
    logger.info("Step 5: Applying diversity control...")
    diversity_weights = ensemble.apply_diversity_control(correlation_threshold=0.95)
    
    # Step 6: Evaluate individual models
    logger.info("Step 6: Evaluating individual models...")
    individual_metrics = ensemble.evaluate_individual_models(X, y)
    
    # Step 7: Get diversity metrics
    logger.info("Step 7: Calculating diversity metrics...")
    diversity_metrics = ensemble.get_model_diversity_metrics()
    
    # Step 8: Generate final predictions
    logger.info("Step 8: Generating ensemble predictions...")
    ensemble_proba = ensemble.predict_proba(X)
    
    # Compile results
    results = {
        'sharpe_optimization': sharpe_results,
        'diversity_control': diversity_weights,
        'individual_metrics': individual_metrics,
        'diversity_metrics': diversity_metrics,
        'ensemble_probabilities': ensemble_proba,
        'optimal_threshold': ensemble.optimal_threshold,
        'optimal_weights': ensemble.optimal_weights,
        'diversity_adjusted_weights': getattr(ensemble, 'diversity_adjusted_weights', {})
    }
    
    logger.info("✅ Enhanced ensemble workflow completed successfully!")
    logger.info("="*60)
    
    return ensemble, results

def analyze_results(results, dates, y):
    """
    Analyze and display comprehensive results
    
    Args:
        results: Results dictionary from workflow
        dates: Datetime index
        y: Target values
    """
    logger.info("📊 ANALYZING RESULTS")
    logger.info("="*60)
    
    # 1. Weight optimization results
    sharpe_results = results['sharpe_optimization']
    logger.info("🎯 Weight Optimization Results:")
    logger.info(f"   Optimal threshold: {sharpe_results['optimal_threshold']:.3f}")
    logger.info(f"   Performance metrics:")
    for key, value in sharpe_results['performance_metrics'].items():
        logger.info(f"     {key}: {value:.4f}")
    
    # 2. Optimal weights
    logger.info("\n⚖️  Optimal Weights:")
    for model, weight in sharpe_results['optimal_weights'].items():
        logger.info(f"   {model}: {weight:.4f}")
    
    # 3. Diversity control results
    diversity_weights = results['diversity_control']
    if diversity_weights:
        logger.info("\n🔄 Diversity-Adjusted Weights:")
        for model, weight in diversity_weights.items():
            logger.info(f"   {model}: {weight:.4f}")
    
    # 4. Individual model performance
    individual_metrics = results['individual_metrics']
    logger.info("\n📈 Individual Model Performance:")
    
    # Create performance summary
    performance_summary = []
    for model, metrics in individual_metrics.items():
        performance_summary.append({
            'Model': model,
            'AUC': metrics['auc'],
            'Brier Score': metrics['brier_score'],
            'Calibration MAE': metrics['calibration_mae'],
            'OOF Sharpe': metrics['oof_sharpe']
        })
    
    performance_df = pd.DataFrame(performance_summary)
    logger.info("\n" + performance_df.to_string(index=False))
    
    # 5. Diversity metrics
    diversity_metrics = results['diversity_metrics']
    logger.info(f"\n🌐 Diversity Metrics:")
    logger.info(f"   Average correlation: {diversity_metrics['average_correlation']:.3f}")
    logger.info(f"   Diversity score: {diversity_metrics['diversity_score']:.3f}")
    logger.info(f"   High correlation pairs: {len(diversity_metrics['high_correlation_pairs'])}")
    
    # 6. Trading strategy analysis
    ensemble_proba = results['ensemble_probabilities']
    threshold = results['optimal_threshold']
    
    # Calculate trading signals
    signals = (ensemble_proba > threshold).astype(float)
    position_sizes = np.clip((ensemble_proba - threshold) / (1 - threshold), 0, 1)
    weighted_signals = signals * position_sizes
    
    # Calculate strategy returns (assuming y represents actual returns)
    # For demonstration, we'll use synthetic returns
    synthetic_returns = np.random.uniform(-0.02, 0.02, len(y))
    strategy_returns = weighted_signals * synthetic_returns
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + strategy_returns)
    
    logger.info(f"\n💰 Trading Strategy Analysis:")
    logger.info(f"   Total trades: {np.sum(np.abs(np.diff(signals, prepend=0)))}")
    logger.info(f"   Average position size: {np.mean(position_sizes):.3f}")
    logger.info(f"   Final cumulative return: {cumulative_returns[-1]:.4f}")
    
    return performance_df, cumulative_returns

def create_visualizations(results, dates, y, cumulative_returns, save_path='reports'):
    """
    Create comprehensive visualizations
    
    Args:
        results: Results dictionary
        dates: Datetime index
        y: Target values
        cumulative_returns: Cumulative strategy returns
        save_path: Path to save visualizations
    """
    logger.info("🎨 Creating visualizations...")
    
    # Create reports directory
    Path(save_path).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Weight comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original optimal weights
    weights = results['optimal_weights']
    models = list(weights.keys())
    weight_values = list(weights.values())
    
    ax1.bar(models, weight_values, alpha=0.7, color='skyblue')
    ax1.set_title('Optimal Weights (Sharpe Optimization)')
    ax1.set_ylabel('Weight')
    ax1.tick_params(axis='x', rotation=45)
    
    # Diversity-adjusted weights
    diversity_weights = results['diversity_adjusted_weights']
    if diversity_weights:
        div_weight_values = [diversity_weights[model] for model in models]
        ax2.bar(models, div_weight_values, alpha=0.7, color='lightcoral')
        ax2.set_title('Diversity-Adjusted Weights')
        ax2.set_ylabel('Weight')
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/ensemble_weights.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Individual model performance
    individual_metrics = results['individual_metrics']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # AUC comparison
    auc_values = [individual_metrics[model]['auc'] for model in models]
    axes[0, 0].bar(models, auc_values, alpha=0.7, color='lightgreen')
    axes[0, 0].set_title('Model AUC Scores')
    axes[0, 0].set_ylabel('AUC')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Brier scores
    brier_values = [individual_metrics[model]['brier_score'] for model in models]
    axes[0, 1].bar(models, brier_values, alpha=0.7, color='lightcoral')
    axes[0, 1].set_title('Model Brier Scores (Lower is Better)')
    axes[0, 1].set_ylabel('Brier Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Calibration MAE
    cal_mae_values = [individual_metrics[model]['calibration_mae'] for model in models]
    axes[1, 0].bar(models, cal_mae_values, alpha=0.7, color='gold')
    axes[1, 0].set_title('Model Calibration MAE (Lower is Better)')
    axes[1, 0].set_ylabel('Calibration MAE')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # OOF Sharpe ratios
    sharpe_values = [individual_metrics[model]['oof_sharpe'] for model in models]
    axes[1, 1].bar(models, sharpe_values, alpha=0.7, color='plum')
    axes[1, 1].set_title('Model OOF Sharpe Ratios')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/individual_model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Trading strategy performance
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Ensemble probabilities over time
    ensemble_proba = results['ensemble_probabilities']
    threshold = results['optimal_threshold']
    
    ax1.plot(dates, ensemble_proba, alpha=0.7, label='Ensemble Probability')
    ax1.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
    ax1.set_title('Ensemble Probabilities Over Time')
    ax1.set_ylabel('Probability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative returns
    ax2.plot(dates, cumulative_returns, linewidth=2, color='green', label='Strategy Returns')
    ax2.axhline(y=1, color='black', linestyle='-', alpha=0.5, label='Baseline')
    ax2.set_title('Cumulative Strategy Returns')
    ax2.set_ylabel('Cumulative Return')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/trading_strategy_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Diversity correlation heatmap
    diversity_metrics = results['diversity_metrics']
    if 'pairwise_correlations' in diversity_metrics:
        # Create correlation matrix
        models = list(results['optimal_weights'].keys())
        n_models = len(models)
        corr_matrix = np.eye(n_models)  # Identity matrix for diagonal
        
        # Fill in correlations
        for pair_name, corr in diversity_metrics['pairwise_correlations'].items():
            model1, model2 = pair_name.split('_')
            i = models.index(model1)
            j = models.index(model2)
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   xticklabels=models, yticklabels=models, fmt='.2f')
        plt.title('Model Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{save_path}/model_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"📊 Visualizations saved to {save_path}/")

def save_results(results, ensemble, save_path='results'):
    """
    Save all results and ensemble for future use
    
    Args:
        results: Results dictionary
        ensemble: Trained ensemble
        save_path: Path to save results
    """
    logger.info("💾 Saving results and ensemble...")
    
    # Create results directory
    Path(save_path).mkdir(exist_ok=True)
    
    # Save ensemble
    ensemble_path = f'{save_path}/enhanced_ensemble.pkl'
    ensemble.save_ensemble(ensemble_path)
    logger.info(f"Ensemble saved to {ensemble_path}")
    
    # Save results summary
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    results_json = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_json[key] = value.tolist()
        elif isinstance(value, dict):
            results_json[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    results_json[key][k] = v.tolist()
                else:
                    results_json[key][k] = v
        else:
            results_json[key] = value
    
    results_path = f'{save_path}/ensemble_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    logger.info(f"Results summary saved to {results_path}")
    
    # Save performance summary
    performance_df = pd.DataFrame([
        {
            'Model': model,
            'AUC': metrics['auc'],
            'Brier_Score': metrics['brier_score'],
            'Calibration_MAE': metrics['calibration_mae'],
            'OOF_Sharpe': metrics['oof_sharpe']
        }
        for model, metrics in results['individual_metrics'].items()
    ])
    
    performance_path = f'{save_path}/model_performance.csv'
    performance_df.to_csv(performance_path, index=False)
    logger.info(f"Performance summary saved to {performance_path}")

def main():
    """Main execution function"""
    logger.info("🚀 Enhanced Ensemble Trading System - Complete Example")
    logger.info("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Step 1: Create synthetic data
    logger.info("📊 Creating synthetic trading data...")
    X, y, dates = create_synthetic_trading_data(n_samples=2000, n_features=30)
    
    # Step 2: Run enhanced ensemble workflow
    ensemble, results = run_enhanced_ensemble_workflow(
        X, y, dates, cost_per_trade=0.001, slippage=0.0005
    )
    
    # Step 3: Analyze results
    performance_df, cumulative_returns = analyze_results(results, dates, y)
    
    # Step 4: Create visualizations
    create_visualizations(results, dates, y, cumulative_returns)
    
    # Step 5: Save results
    save_results(results, ensemble)
    
    # Step 6: Final summary
    logger.info("\n" + "="*60)
    logger.info("🎉 ENHANCED ENSEMBLE WORKFLOW COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    
    logger.info("\n📋 SUMMARY:")
    logger.info(f"   • Models trained: {len(ensemble.models)}")
    logger.info(f"   • Optimal threshold: {ensemble.optimal_threshold:.3f}")
    logger.info(f"   • Best individual AUC: {performance_df['AUC'].max():.3f}")
    logger.info(f"   • Best individual Sharpe: {performance_df['OOF_Sharpe'].max():.3f}")
    logger.info(f"   • Final strategy return: {cumulative_returns[-1]:.4f}")
    
    logger.info("\n📁 Files saved:")
    logger.info("   • Ensemble: results/enhanced_ensemble.pkl")
    logger.info("   • Results: results/ensemble_results.json")
    logger.info("   • Performance: results/model_performance.csv")
    logger.info("   • Visualizations: reports/")
    
    logger.info("\n🔧 Next steps:")
    logger.info("   • Load ensemble: ensemble.load_ensemble('results/enhanced_ensemble.pkl')")
    logger.info("   • Make predictions: ensemble.predict_proba(new_X)")
    logger.info("   • Adjust parameters and retrain as needed")
    
    return ensemble, results

if __name__ == "__main__":
    # Run the complete example
    ensemble, results = main()

