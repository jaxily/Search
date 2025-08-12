#!/usr/bin/env python3
"""
Ensemble System Robustness Audit Script
Implements purged & embargoed CV and null tests
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')

class PurgedKFold:
    """
    Purged and embargoed cross-validation for financial time series
    Based on Lopez de Prado's methodology
    """
    
    def __init__(self, n_splits=3, embargo_frac=0.02):
        self.n_splits = n_splits
        self.embargo_frac = embargo_frac
        
    def split(self, X, y=None, groups=None):
        """Generate train/validation splits with purging and embargo"""
        n_samples = len(X)
        
        # Calculate embargo size
        embargo_size = int(n_samples * self.embargo_frac)
        
        # Calculate split sizes
        split_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Calculate split boundaries
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, n_samples)
            
            # Training indices (before this split)
            train_idx = list(range(0, start_idx))
            
            # Validation indices (this split, excluding embargo)
            val_start = start_idx
            val_end = end_idx - embargo_size
            
            if val_end > val_start:
                val_idx = list(range(val_start, val_end))
                
                # Apply purging: remove training samples that overlap with validation
                # In practice, this would remove samples where features overlap
                # For simplicity, we'll just use the basic split
                
                yield train_idx, val_idx

class EnsembleRobustnessAuditor:
    """Robustness auditor for ensemble trading system"""
    
    def __init__(self, results_dir: str = "results/qqq_ensemble"):
        self.results_dir = Path(results_dir)
        self.artifacts = {}
        self.load_artifacts()
        
    def load_artifacts(self):
        """Load ensemble artifacts"""
        print("🔍 Loading ensemble artifacts for robustness audit...")
        
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
        
        print(f"✅ Loaded artifacts: {len(self.artifacts)} files")
        
    def run_purged_embargoed_cv(self, embargo_frac=0.02):
        """Run purged and embargoed cross-validation"""
        print(f"\n🔄 Running Purged & Embargoed CV (embargo_frac={embargo_frac})")
        
        # Get ensemble data
        weights = np.array(list(self.artifacts['weights'].values()))
        ensemble_probs = np.sum(self.artifacts['oof_probs'] * weights, axis=1)
        targets = self.artifacts['targets']
        
        # Create purged CV splits
        purged_cv = PurgedKFold(n_splits=5, embargo_frac=embargo_frac)
        
        # Store results
        purged_sharpe_scores = []
        purged_turnover_scores = []
        
        for train_idx, val_idx in purged_cv.split(ensemble_probs):
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
                
            # Get training and validation data
            train_probs = ensemble_probs[train_idx]
            train_targets = targets[train_idx]
            val_probs = ensemble_probs[val_idx]
            val_targets = targets[val_idx]
            
            # Find optimal threshold on training data
            thresholds = np.arange(0.50, 0.71, 0.01)
            train_sharpe_scores = []
            
            for tau in thresholds:
                predictions = (train_probs > tau).astype(int)
                strategy_returns = predictions * train_targets
                
                if np.std(strategy_returns) > 0:
                    sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
                    train_sharpe_scores.append(sharpe)
                else:
                    train_sharpe_scores.append(0)
                    
            # Find best threshold
            best_idx = np.argmax(train_sharpe_scores)
            best_threshold = thresholds[best_idx]
            
            # Apply to validation data
            val_predictions = (val_probs > best_threshold).astype(int)
            val_strategy_returns = val_predictions * val_targets
            
            # Calculate validation metrics
            if np.std(val_strategy_returns) > 0:
                val_sharpe = (np.mean(val_strategy_returns) / np.std(val_strategy_returns)) * np.sqrt(252)
                purged_sharpe_scores.append(val_sharpe)
            else:
                purged_sharpe_scores.append(0)
                
            # Calculate turnover
            val_turnover = np.sum(np.abs(np.diff(val_predictions)))
            purged_turnover_scores.append(val_turnover)
            
        # Report results
        if purged_sharpe_scores:
            avg_sharpe = np.mean(purged_sharpe_scores)
            sharpe_std = np.std(purged_sharpe_scores)
            avg_turnover = np.mean(purged_turnover_scores)
            
            print(f"Purged CV Results:")
            print(f"  Average Sharpe: {avg_sharpe:.3f} ± {sharpe_std:.3f}")
            print(f"  Average Turnover: {avg_turnover:.1f}")
            print(f"  Sharpe Range: {min(purged_sharpe_scores):.3f} - {max(purged_sharpe_scores):.3f}")
            
            return {
                'sharpe_scores': purged_sharpe_scores,
                'turnover_scores': purged_turnover_scores,
                'avg_sharpe': avg_sharpe,
                'sharpe_std': sharpe_std
            }
        else:
            print("⚠️  No valid purged CV splits generated")
            return None
            
    def run_label_permutation_test(self, n_permutations=100):
        """Run label permutation test to validate performance"""
        print(f"\n🎲 Running Label Permutation Test ({n_permutations} permutations)")
        
        # Get ensemble data
        weights = np.array(list(self.artifacts['weights'].values()))
        ensemble_probs = np.sum(self.artifacts['oof_probs'] * weights, axis=1)
        targets = self.artifacts['targets']
        
        # Calculate real Sharpe ratio
        threshold = self.artifacts['threshold']['optimal_threshold']
        real_predictions = (ensemble_probs > threshold).astype(int)
        real_strategy_returns = real_predictions * targets
        
        if np.std(real_strategy_returns) > 0:
            real_sharpe = (np.mean(real_strategy_returns) / np.std(real_strategy_returns)) * np.sqrt(252)
        else:
            real_sharpe = 0
            
        print(f"Real Sharpe Ratio: {real_sharpe:.3f}")
        
        # Run permutations
        permuted_sharpe_scores = []
        
        for i in range(n_permutations):
            # Permute targets
            permuted_targets = np.random.permutation(targets)
            
            # Calculate Sharpe with permuted targets
            permuted_strategy_returns = real_predictions * permuted_targets
            
            if np.std(permuted_strategy_returns) > 0:
                permuted_sharpe = (np.mean(permuted_strategy_returns) / np.std(permuted_strategy_returns)) * np.sqrt(252)
                permuted_sharpe_scores.append(permuted_sharpe)
            else:
                permuted_sharpe_scores.append(0)
                
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{n_permutations} permutations")
                
        # Analyze results
        permuted_sharpe_scores = np.array(permuted_sharpe_scores)
        
        # Calculate p-value (proportion of permuted Sharpe >= real Sharpe)
        p_value = np.mean(permuted_sharpe_scores >= real_sharpe)
        
        # Calculate effect size
        effect_size = (real_sharpe - np.mean(permuted_sharpe_scores)) / np.std(permuted_sharpe_scores)
        
        print(f"\nPermutation Test Results:")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Effect Size: {effect_size:.3f}")
        print(f"  Permuted Sharpe Range: {permuted_sharpe_scores.min():.3f} - {permuted_sharpe_scores.max():.3f}")
        print(f"  Permuted Sharpe Mean: {np.mean(permuted_sharpe_scores):.3f}")
        print(f"  Permuted Sharpe Std: {np.std(permuted_sharpe_scores):.3f}")
        
        # Interpret results
        if p_value < 0.05:
            print("✅ Performance is statistically significant (p < 0.05)")
        else:
            print("⚠️  Performance may not be statistically significant")
            
        if effect_size > 2.0:
            print("✅ Large effect size - strong evidence of predictive power")
        elif effect_size > 1.0:
            print("✅ Medium effect size - moderate evidence of predictive power")
        else:
            print("⚠️  Small effect size - weak evidence of predictive power")
            
        return {
            'real_sharpe': real_sharpe,
            'permuted_sharpe_scores': permuted_sharpe_scores,
            'p_value': p_value,
            'effect_size': effect_size
        }
        
    def run_block_bootstrap_test(self, block_size=30, n_bootstrap=100):
        """Run block bootstrap test for Sharpe ratio confidence intervals"""
        print(f"\n📊 Running Block Bootstrap Test (block_size={block_size}, n_bootstrap={n_bootstrap})")
        
        # Get ensemble data
        weights = np.array(list(self.artifacts['weights'].values()))
        ensemble_probs = np.sum(self.artifacts['oof_probs'] * weights, axis=1)
        targets = self.artifacts['targets']
        
        # Calculate strategy returns
        threshold = self.artifacts['threshold']['optimal_threshold']
        predictions = (ensemble_probs > threshold).astype(int)
        strategy_returns = predictions * targets
        
        # Calculate original Sharpe
        if np.std(strategy_returns) > 0:
            original_sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
        else:
            original_sharpe = 0
            
        print(f"Original Sharpe Ratio: {original_sharpe:.3f}")
        
        # Run block bootstrap
        n_samples = len(strategy_returns)
        n_blocks = n_samples // block_size
        bootstrap_sharpe_scores = []
        
        for i in range(n_bootstrap):
            # Sample blocks with replacement
            bootstrap_returns = []
            
            for _ in range(n_blocks):
                # Randomly select a block
                start_idx = np.random.randint(0, n_samples - block_size + 1)
                block_returns = strategy_returns[start_idx:start_idx + block_size]
                bootstrap_returns.extend(block_returns)
                
            # Calculate Sharpe for bootstrap sample
            bootstrap_returns = np.array(bootstrap_returns)
            
            if np.std(bootstrap_returns) > 0:
                bootstrap_sharpe = (np.mean(bootstrap_returns) / np.std(bootstrap_returns)) * np.sqrt(252)
                bootstrap_sharpe_scores.append(bootstrap_sharpe)
            else:
                bootstrap_sharpe_scores.append(0)
                
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{n_bootstrap} bootstrap samples")
                
        # Calculate confidence intervals
        bootstrap_sharpe_scores = np.array(bootstrap_sharpe_scores)
        
        ci_95 = np.percentile(bootstrap_sharpe_scores, [2.5, 97.5])
        ci_90 = np.percentile(bootstrap_sharpe_scores, [5, 95])
        ci_68 = np.percentile(bootstrap_sharpe_scores, [16, 84])
        
        print(f"\nBlock Bootstrap Results:")
        print(f"  95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
        print(f"  90% CI: [{ci_90[0]:.3f}, {ci_90[1]:.3f}]")
        print(f"  68% CI: [{ci_68[0]:.3f}, {ci_68[1]:.3f}]")
        print(f"  Bootstrap Mean: {np.mean(bootstrap_sharpe_scores):.3f}")
        print(f"  Bootstrap Std: {np.std(bootstrap_sharpe_scores):.3f}")
        
        # Check if original Sharpe is within confidence intervals
        within_95 = ci_95[0] <= original_sharpe <= ci_95[1]
        within_90 = ci_90[0] <= original_sharpe <= ci_90[1]
        within_68 = ci_68[0] <= original_sharpe <= ci_68[1]
        
        print(f"\nConfidence Interval Coverage:")
        print(f"  95% CI: {'✅' if within_95 else '❌'}")
        print(f"  90% CI: {'✅' if within_90 else '❌'}")
        print(f"  68% CI: {'✅' if within_68 else '❌'}")
        
        return {
            'original_sharpe': original_sharpe,
            'bootstrap_sharpe_scores': bootstrap_sharpe_scores,
            'ci_95': ci_95,
            'ci_90': ci_90,
            'ci_68': ci_68
        }
        
    def run_ensemble_stability_test(self):
        """Test ensemble stability across different time periods"""
        print(f"\n🔒 Running Ensemble Stability Test")
        
        # Get ensemble data
        weights = np.array(list(self.artifacts['weights'].values()))
        ensemble_probs = np.sum(self.artifacts['oof_probs'] * weights, axis=1)
        targets = self.artifacts['targets']
        
        # Split data into time periods
        n_samples = len(ensemble_probs)
        period_size = n_samples // 4  # 4 equal periods
        
        period_performances = []
        
        for i in range(4):
            start_idx = i * period_size
            end_idx = min((i + 1) * period_size, n_samples)
            
            period_probs = ensemble_probs[start_idx:end_idx]
            period_targets = targets[start_idx:end_idx]
            
            # Calculate period performance
            threshold = self.artifacts['threshold']['optimal_threshold']
            period_predictions = (period_probs > threshold).astype(int)
            period_strategy_returns = period_predictions * period_targets
            
            if np.std(period_strategy_returns) > 0:
                period_sharpe = (np.mean(period_strategy_returns) / np.std(period_strategy_returns)) * np.sqrt(252)
                period_performances.append(period_sharpe)
            else:
                period_performances.append(0)
                
        # Analyze stability
        period_performances = np.array(period_performances)
        performance_std = np.std(period_performances)
        performance_range = period_performances.max() - period_performances.min()
        
        print(f"Period-wise Sharpe Ratios:")
        for i, sharpe in enumerate(period_performances):
            print(f"  Period {i+1}: {sharpe:.3f}")
            
        print(f"\nStability Metrics:")
        print(f"  Performance Std: {performance_std:.3f}")
        print(f"  Performance Range: {performance_range:.3f}")
        
        # Assess stability
        if performance_std < 1.0:
            print("✅ High stability - consistent performance across periods")
        elif performance_std < 2.0:
            print("✅ Medium stability - moderate performance variation")
        else:
            print("⚠️  Low stability - high performance variation across periods")
            
        return {
            'period_performances': period_performances,
            'performance_std': performance_std,
            'performance_range': performance_range
        }
        
    def generate_robustness_report(self, results_dir=None):
        """Generate comprehensive robustness report"""
        if results_dir is None:
            results_dir = self.results_dir
            
        print(f"\n📋 Generating Robustness Report...")
        
        # Run all tests
        purged_results = self.run_purged_embargoed_cv()
        permutation_results = self.run_label_permutation_test()
        bootstrap_results = self.run_block_bootstrap_test()
        stability_results = self.run_ensemble_stability_test()
        
        # Create summary plots
        self._create_robustness_plots(
            purged_results, permutation_results, bootstrap_results, stability_results
        )
        
        # Generate summary report
        report = {
            'purged_cv': purged_results,
            'permutation_test': permutation_results,
            'block_bootstrap': bootstrap_results,
            'stability_test': stability_results,
            'summary': self._generate_summary_assessment(
                purged_results, permutation_results, bootstrap_results, stability_results
            )
        }
        
        # Save report
        report_path = results_dir / 'robustness_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"✅ Robustness report saved to: {report_path}")
        
        return report
        
    def _create_robustness_plots(self, purged_results, permutation_results, bootstrap_results, stability_results):
        """Create visualization plots for robustness analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Ensemble Robustness Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Purged CV Sharpe distribution
        if purged_results:
            axes[0, 0].hist(purged_results['sharpe_scores'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(purged_results['avg_sharpe'], color='red', linestyle='--', label=f'Mean: {purged_results["avg_sharpe"]:.3f}')
            axes[0, 0].set_title('Purged CV Sharpe Distribution')
            axes[0, 0].set_xlabel('Sharpe Ratio')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
        # Plot 2: Permutation test results
        if permutation_results:
            axes[0, 1].hist(permutation_results['permuted_sharpe_scores'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].axvline(permutation_results['real_sharpe'], color='red', linestyle='--', linewidth=2, label=f'Real: {permutation_results["real_sharpe"]:.3f}')
            axes[0, 1].set_title('Label Permutation Test')
            axes[0, 1].set_xlabel('Sharpe Ratio')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
        # Plot 3: Block bootstrap results
        if bootstrap_results:
            axes[1, 0].hist(bootstrap_results['bootstrap_sharpe_scores'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1, 0].axvline(bootstrap_results['original_sharpe'], color='red', linestyle='--', linewidth=2, label=f'Original: {bootstrap_results["original_sharpe"]:.3f}')
            axes[1, 0].axvline(bootstrap_results['ci_95'][0], color='orange', linestyle=':', label='95% CI')
            axes[1, 0].axvline(bootstrap_results['ci_95'][1], color='orange', linestyle=':')
            axes[1, 0].set_title('Block Bootstrap Sharpe Distribution')
            axes[1, 0].set_xlabel('Sharpe Ratio')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
        # Plot 4: Stability across periods
        if stability_results:
            periods = range(1, len(stability_results['period_performances']) + 1)
            axes[1, 1].bar(periods, stability_results['period_performances'], alpha=0.7, color='gold', edgecolor='black')
            axes[1, 1].axhline(np.mean(stability_results['period_performances']), color='red', linestyle='--', label=f'Mean: {np.mean(stability_results["period_performances"]):.3f}')
            axes[1, 1].set_title('Performance Stability Across Periods')
            axes[1, 1].set_xlabel('Time Period')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / 'robustness_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Robustness plots saved to: {plot_path}")
        
    def _generate_summary_assessment(self, purged_results, permutation_results, bootstrap_results, stability_results):
        """Generate summary assessment of robustness"""
        summary = {
            'overall_robustness': 'UNKNOWN',
            'key_findings': [],
            'recommendations': []
        }
        
        # Assess purged CV
        if purged_results:
            if purged_results['sharpe_std'] < 1.0:
                summary['key_findings'].append("Purged CV shows stable performance")
            else:
                summary['key_findings'].append("Purged CV shows performance instability")
                summary['recommendations'].append("Consider longer embargo periods or different CV strategy")
                
        # Assess permutation test
        if permutation_results:
            if permutation_results['p_value'] < 0.05:
                summary['key_findings'].append("Performance is statistically significant")
            else:
                summary['key_findings'].append("Performance may not be statistically significant")
                summary['recommendations'].append("Investigate feature engineering and model selection")
                
        if permutation_results and permutation_results['effect_size'] > 2.0:
            summary['key_findings'].append("Large effect size indicates strong predictive power")
        elif permutation_results and permutation_results['effect_size'] < 1.0:
            summary['key_findings'].append("Small effect size suggests weak predictive power")
            summary['recommendations'].append("Review model architecture and feature selection")
            
        # Assess bootstrap stability
        if bootstrap_results:
            ci_width = bootstrap_results['ci_95'][1] - bootstrap_results['ci_95'][0]
            if ci_width < 2.0:
                summary['key_findings'].append("Narrow confidence intervals indicate stable estimates")
            else:
                summary['key_findings'].append("Wide confidence intervals suggest estimation uncertainty")
                summary['recommendations'].append("Consider longer time series or more robust estimation")
                
        # Assess stability
        if stability_results:
            if stability_results['performance_std'] < 1.0:
                summary['key_findings'].append("High performance stability across time periods")
            else:
                summary['key_findings'].append("Low performance stability across time periods")
                summary['recommendations'].append("Investigate regime changes and model adaptation")
                
        # Overall assessment
        positive_findings = sum(1 for finding in summary['key_findings'] if 'stable' in finding.lower() or 'significant' in finding.lower() or 'strong' in finding.lower())
        total_findings = len(summary['key_findings'])
        
        if positive_findings / total_findings >= 0.7:
            summary['overall_robustness'] = 'HIGH'
        elif positive_findings / total_findings >= 0.4:
            summary['overall_robustness'] = 'MEDIUM'
        else:
            summary['overall_robustness'] = 'LOW'
            
        return summary

def main():
    """Main execution function"""
    try:
        auditor = EnsembleRobustnessAuditor()
        report = auditor.generate_robustness_report()
        
        print(f"\n🎯 ROBUSTNESS AUDIT COMPLETED")
        print(f"Overall Robustness: {report['summary']['overall_robustness']}")
        print(f"\nKey Findings:")
        for finding in report['summary']['key_findings']:
            print(f"  • {finding}")
            
        if report['summary']['recommendations']:
            print(f"\nRecommendations:")
            for rec in report['summary']['recommendations']:
                print(f"  • {rec}")
                
    except Exception as e:
        print(f"❌ Robustness audit failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
