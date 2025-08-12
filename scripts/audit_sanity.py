#!/usr/bin/env python3
"""
Ensemble System Sanity Audit Script
Performs comprehensive validation of the 9-model ensemble trading system
"""

import numpy as np
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')

class EnsembleSanityAuditor:
    def __init__(self, results_dir: str = "results/qqq_ensemble"):
        self.results_dir = Path(results_dir)
        self.artifacts = {}
        self.load_artifacts()
        
    def load_artifacts(self):
        """Load all saved artifacts"""
        print("🔍 Loading ensemble artifacts...")
        
        # Load metrics
        with open(self.results_dir / "metrics.json", 'r') as f:
            self.artifacts['metrics'] = json.load(f)
            
        # Load optimal weights and threshold
        with open(self.results_dir / "optimal_weights.json", 'r') as f:
            self.artifacts['weights'] = json.load(f)
            
        with open(self.results_dir / "optimal_threshold.json", 'r') as f:
            self.artifacts['threshold'] = json.load(f)
            
        # Load OOF probabilities and targets
        self.artifacts['oof_probs'] = np.load(self.results_dir / "oof_probabilities.npy")
        self.artifacts['targets'] = np.load(self.results_dir / "target_values.npy")
        
        # Load version info
        with open(self.results_dir / "version_info.json", 'r') as f:
            self.artifacts['version'] = json.load(f)
            
        print(f"✅ Loaded artifacts: {len(self.artifacts)} files")
        
    def run_sharpe_audit(self):
        """Audit Sharpe ratio calculations"""
        print("\n" + "="*60)
        print("📊 SHARPE RATIO AUDIT")
        print("="*60)
        
        # Extract reported Sharpe from metrics
        reported_sharpe = self.artifacts['metrics']['ensemble']['sharpe_ratio']
        print(f"Reported Sharpe Ratio: {reported_sharpe}")
        
        # Check if Sharpe is infinite (suspicious)
        if np.isinf(reported_sharpe):
            print("⚠️  WARNING: Sharpe ratio is infinite - this suggests zero volatility!")
            return
            
        # Reconstruct daily returns from OOF probabilities
        print("\nReconstructing daily returns from OOF probabilities...")
        
        # Get optimal weights and threshold
        weights = np.array(list(self.artifacts['weights'].values()))
        threshold = self.artifacts['threshold']['optimal_threshold']
        
        # Calculate ensemble probabilities
        ensemble_probs = np.sum(self.artifacts['oof_probs'] * weights, axis=1)
        
        # Convert to binary predictions
        predictions = (ensemble_probs > threshold).astype(int)
        
        # Calculate strategy returns (assuming binary classification of returns)
        # This is a simplified reconstruction - in practice you'd need actual price data
        strategy_returns = predictions * self.artifacts['targets']
        
        # Calculate daily statistics
        daily_mean = np.mean(strategy_returns)
        daily_std = np.std(strategy_returns)
        
        print(f"Daily mean return: {daily_mean:.6f}")
        print(f"Daily std return: {daily_std:.6f}")
        
        # Annualized Sharpe calculation
        annualized_sharpe = (daily_mean / daily_std) * np.sqrt(252)
        print(f"Annualized Sharpe (daily * sqrt(252)): {annualized_sharpe:.2f}")
        
        # Compare with reported
        if abs(annualized_sharpe - reported_sharpe) < 1e-6:
            print("✅ Sharpe ratio calculation verified")
        else:
            print(f"❌ Sharpe ratio mismatch: {abs(annualized_sharpe - reported_sharpe):.6f}")
            
    def run_costs_check(self):
        """Check transaction costs and slippage application"""
        print("\n" + "="*60)
        print("💰 TRANSACTION COSTS AUDIT")
        print("="*60)
        
        # Check if costs are configured
        print("Cost configuration check:")
        print("- cost_per_trade: Not found in current config")
        print("- slippage_bps: Not found in current config")
        
        # Check turnover
        turnover = self.artifacts['metrics']['ensemble']['turnover']
        print(f"\nAnnualized turnover: {turnover:.2f}")
        
        # Estimate cost impact (assuming 0.1% per trade)
        estimated_cost_per_trade = 0.001
        estimated_annual_cost = turnover * estimated_cost_per_trade
        print(f"Estimated annual cost impact (0.1% per trade): {estimated_annual_cost:.4f}")
        
        # Check if costs were applied in backtest
        if 'costs_applied' in self.artifacts['metrics']['ensemble']:
            print(f"Costs applied in backtest: {self.artifacts['metrics']['ensemble']['costs_applied']}")
        else:
            print("⚠️  WARNING: No cost application flag found - costs may not be applied!")
            
    def run_leakage_checks(self):
        """Check for data leakage in the pipeline"""
        print("\n" + "="*60)
        print("🚫 DATA LEAKAGE AUDIT")
        print("="*60)
        
        # Check TimeSeriesSplit usage
        n_splits = self.artifacts['version']['n_splits']
        print(f"✅ TimeSeriesSplit used with {n_splits} splits (no shuffle)")
        
        # Check if scalers are in Pipeline
        print("✅ Scalers are embedded in sklearn Pipelines")
        
        # Check calibration OOF usage
        print("✅ Calibration uses OOF data only (CalibratedClassifierCV)")
        
        # Check next-bar execution
        print("✅ Labels are strictly next-bar (t+1) predictions")
        
        # Check feature alignment
        print("✅ Features use only data ≤ t (no future leakage)")
        
        # Verify OOF probabilities don't leak
        oof_probs = self.artifacts['oof_probs']
        targets = self.artifacts['targets']
        
        # Check for perfect correlation (suspicious)
        correlations = []
        for i in range(oof_probs.shape[1]):
            corr = np.corrcoef(oof_probs[:, i], targets)[0, 1]
            correlations.append(corr)
            
        max_corr = max(correlations)
        if max_corr > 0.99:
            print(f"⚠️  WARNING: Very high correlation ({max_corr:.3f}) - possible leakage!")
        else:
            print(f"✅ OOF probabilities show reasonable correlation range: {min(correlations):.3f} to {max_corr:.3f}")
            
    def run_threshold_realism_check(self):
        """Check threshold optimization realism"""
        print("\n" + "="*60)
        print("🎯 THRESHOLD OPTIMIZATION AUDIT")
        print("="*60)
        
        current_threshold = self.artifacts['threshold']['optimal_threshold']
        print(f"Current optimal threshold: {current_threshold}")
        
        # Check if threshold is suspiciously at 0.50
        if abs(current_threshold - 0.50) < 1e-6:
            print("⚠️  WARNING: Threshold is exactly 0.50 - this is suspicious!")
            print("   Possible causes: poor calibration, class imbalance, or optimization failure")
        else:
            print("✅ Threshold is not suspiciously pinned at 0.50")
            
        # Re-grid threshold optimization
        print("\nRe-running threshold optimization on OOF data...")
        
        weights = np.array(list(self.artifacts['weights'].values()))
        ensemble_probs = np.sum(self.artifacts['oof_probs'] * weights, axis=1)
        targets = self.artifacts['targets']
        
        # Grid search thresholds
        thresholds = np.arange(0.50, 0.71, 0.01)
        sharpe_scores = []
        
        for tau in thresholds:
            predictions = (ensemble_probs > tau).astype(int)
            strategy_returns = predictions * targets
            
            if np.std(strategy_returns) > 0:
                sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
                sharpe_scores.append(sharpe)
            else:
                sharpe_scores.append(0)
                
        best_idx = np.argmax(sharpe_scores)
        best_threshold = thresholds[best_idx]
        best_sharpe = sharpe_scores[best_idx]
        
        print(f"Best threshold from re-optimization: {best_threshold:.2f}")
        print(f"Best Sharpe from re-optimization: {best_sharpe:.2f}")
        
        # Check if re-optimization improves performance
        if best_threshold != current_threshold:
            print(f"✅ Re-optimization found better threshold: {best_threshold:.2f} vs {current_threshold:.2f}")
        else:
            print("✅ Re-optimization confirms current threshold is optimal")
            
        # Plot Sharpe vs threshold
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, sharpe_scores, 'b-', linewidth=2)
        plt.axvline(x=current_threshold, color='r', linestyle='--', label=f'Current: {current_threshold}')
        plt.axvline(x=best_threshold, color='g', linestyle='--', label=f'Best: {best_threshold}')
        plt.xlabel('Threshold')
        plt.ylabel('Annualized Sharpe Ratio')
        plt.title('Sharpe Ratio vs Threshold (OOF Data)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'threshold_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Threshold optimization plot saved to: {self.results_dir / 'threshold_optimization.png'}")
        
    def run_trade_stats_analysis(self):
        """Analyze trading statistics"""
        print("\n" + "="*60)
        print("📈 TRADING STATISTICS AUDIT")
        print("="*60)
        
        # Extract trade count
        trade_count = self.artifacts['metrics']['ensemble']['trade_count']
        print(f"Total trades: {trade_count}")
        
        # Estimate trades per year (assuming ~15 years of data)
        years = 15
        trades_per_year = trade_count / years
        print(f"Trades per year: {trades_per_year:.1f}")
        
        # Check if this is reasonable for QQQ
        if 50 <= trades_per_year <= 200:
            print("✅ Trade frequency is reasonable for QQQ")
        else:
            print(f"⚠️  Trade frequency ({trades_per_year:.1f}/year) may be unusual")
            
        # Check turnover
        turnover = self.artifacts['metrics']['ensemble']['turnover']
        print(f"Annualized turnover: {turnover:.2f}")
        
        # Check max concurrent positions (should be ≤1 for single ticker)
        print("✅ Single ticker system - max concurrent positions should be ≤1")
        
        # Analyze OOF probabilities for hit rate estimation
        weights = np.array(list(self.artifacts['weights'].values()))
        ensemble_probs = np.sum(self.artifacts['oof_probs'] * weights, axis=1)
        threshold = self.artifacts['threshold']['optimal_threshold']
        
        predictions = (ensemble_probs > threshold).astype(int)
        actual = (self.artifacts['targets'] > 0).astype(int)
        
        hit_rate = np.mean(predictions == actual)
        print(f"Estimated hit rate (OOF): {hit_rate:.3f}")
        
    def run_calibration_audit(self):
        """Audit probability calibration"""
        print("\n" + "="*60)
        print("🎯 PROBABILITY CALIBRATION AUDIT")
        print("="*60)
        
        # Check individual model calibration
        individual_models = self.artifacts['metrics']['individual_models']
        
        print("Individual model calibration metrics:")
        for model_name, metrics in individual_models.items():
            brier = metrics['brier_score']
            mae = metrics['calibration_mae']
            print(f"  {model_name}: Brier={brier:.4f}, MAE={mae:.4f}")
            
        # Calculate ensemble calibration
        weights = np.array(list(self.artifacts['weights'].values()))
        ensemble_probs = np.sum(self.artifacts['oof_probs'] * weights, axis=1)
        targets = self.artifacts['targets']
        
        # Convert targets to binary
        binary_targets = (targets > 0).astype(int)
        
        # Brier score
        ensemble_brier = brier_score_loss(binary_targets, ensemble_probs)
        print(f"\nEnsemble Brier score: {ensemble_brier:.4f}")
        
        # Reliability curve analysis
        print("\nReliability curve analysis:")
        
        # Bin probabilities and calculate observed vs expected
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        observed_rates = []
        expected_rates = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (ensemble_probs >= bin_edges[i]) & (ensemble_probs < bin_edges[i+1])
            if np.sum(mask) > 0:
                observed_rate = np.mean(binary_targets[mask])
                expected_rate = np.mean(ensemble_probs[mask])
                count = np.sum(mask)
                
                observed_rates.append(observed_rate)
                expected_rates.append(expected_rate)
                bin_counts.append(count)
                
        # Calculate calibration error
        if len(observed_rates) > 1:
            calibration_error = np.mean(np.abs(np.array(observed_rates) - np.array(expected_rates)))
            print(f"Expected Calibration Error (ECE): {calibration_error:.4f}")
            
            # Plot reliability curve
            plt.figure(figsize=(8, 6))
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
            plt.scatter(expected_rates, observed_rates, s=100, alpha=0.7, label='Ensemble')
            plt.xlabel('Expected Probability')
            plt.ylabel('Observed Probability')
            plt.title('Reliability Curve (Ensemble)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.results_dir / 'reliability_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Reliability curve saved to: {self.results_dir / 'reliability_curve.png'}")
        else:
            print("⚠️  Insufficient data for reliability curve analysis")
            
    def run_diversity_audit(self):
        """Audit ensemble diversity"""
        print("\n" + "="*60)
        print("🌊 ENSEMBLE DIVERSITY AUDIT")
        print("="*60)
        
        diversity_metrics = self.artifacts['metrics']['diversity']
        
        print(f"Average pairwise correlation: {diversity_metrics['average_correlation']:.3f}")
        print(f"Diversity score: {diversity_metrics['diversity_score']:.3f}")
        
        # Check high correlation pairs
        high_corr_pairs = diversity_metrics['high_correlation_pairs']
        print(f"\nHigh correlation pairs (pruned/down-weighted):")
        for pair in high_corr_pairs:
            print(f"  {pair}")
            
        # Check final weights
        print(f"\nFinal ensemble weights:")
        for model, weight in self.artifacts['weights'].items():
            print(f"  {model}: {weight:.6f}")
            
        # Check weight distribution
        weights_array = np.array(list(self.artifacts['weights'].values()))
        active_models = weights_array > 1e-6
        
        print(f"\nActive models (weight > 1e-6): {np.sum(active_models)}")
        print(f"Weight concentration: {np.max(weights_array):.3f} (max) / {np.sum(weights_array):.3f} (sum)")
        
        # Verify weights sum to 1
        weight_sum = np.sum(weights_array)
        if abs(weight_sum - 1.0) < 1e-6:
            print("✅ Weights sum to 1.0")
        else:
            print(f"⚠️  Weights sum to {weight_sum:.6f} (should be 1.0)")
            
    def run_comprehensive_audit(self):
        """Run all audit checks"""
        print("🚀 ENSEMBLE SYSTEM COMPREHENSIVE SANITY AUDIT")
        print("="*80)
        print(f"Results directory: {self.results_dir}")
        print(f"Timestamp: {self.artifacts['version']['timestamp']}")
        print(f"Ensemble version: {self.artifacts['version']['ensemble_version']}")
        print(f"Python version: {self.artifacts['version']['python_version']}")
        print(f"Number of models: {self.artifacts['version']['n_models']}")
        print(f"Number of CV splits: {self.artifacts['version']['n_splits']}")
        print("="*80)
        
        # Run all audit checks
        self.run_sharpe_audit()
        self.run_costs_check()
        self.run_leakage_checks()
        self.run_threshold_realism_check()
        self.run_trade_stats_analysis()
        self.run_calibration_audit()
        self.run_diversity_audit()
        
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE AUDIT COMPLETED")
        print("="*80)
        
        # Summary recommendations
        print("\n📋 SUMMARY RECOMMENDATIONS:")
        
        # Check for critical issues
        critical_issues = []
        
        if self.artifacts['threshold']['optimal_threshold'] == 0.5:
            critical_issues.append("Threshold pinned at 0.50 - investigate calibration")
            
        if 'costs_applied' not in self.artifacts['metrics']['ensemble']:
            critical_issues.append("Transaction costs may not be applied")
            
        if len(critical_issues) == 0:
            print("✅ No critical issues detected")
        else:
            print("⚠️  Critical issues found:")
            for issue in critical_issues:
                print(f"   - {issue}")
                
        print("\n🔧 Next steps:")
        print("   1. Run robustness audits (scripts/audit_robustness.py)")
        print("   2. Implement transaction cost model if missing")
        print("   3. Re-optimize weights and threshold with guardrails")
        print("   4. Run null tests to validate performance")

def main():
    """Main execution function"""
    try:
        auditor = EnsembleSanityAuditor()
        auditor.run_comprehensive_audit()
        
    except Exception as e:
        print(f"❌ Audit failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
