"""
Enhanced Trading Utilities for Ensemble System
Implements transaction costs, slippage, turnover control, and robust Sharpe calculations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List, Any
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class EnhancedTradingCalculator:
    """Enhanced trading calculator with transaction costs and turnover control"""
    
    def __init__(self, 
                 cost_per_trade: float = 0.001,
                 slippage_bps: float = 5.0,
                 min_hold_days: int = 1,
                 hysteresis_buffer: float = 0.02):
        """
        Initialize enhanced trading calculator
        
        Args:
            cost_per_trade: Transaction cost per trade (e.g., 0.001 = 0.1%)
            slippage_bps: Slippage in basis points (e.g., 5 = 0.05%)
            min_hold_days: Minimum holding period in days
            hysteresis_buffer: Buffer to avoid flip-flopping (e.g., 0.02 = 2%)
        """
        self.cost_per_trade = cost_per_trade
        self.slippage_bps = slippage_bps
        self.min_hold_days = min_hold_days
        self.hysteresis_buffer = hysteresis_buffer
        
        # Convert slippage from bps to decimal
        self.slippage = slippage_bps / 10000.0
        
        logger.info(f"EnhancedTradingCalculator initialized:")
        logger.info(f"  Cost per trade: {cost_per_trade:.4f}")
        logger.info(f"  Slippage: {slippage_bps} bps ({self.slippage:.6f})")
        logger.info(f"  Min hold days: {min_hold_days}")
        logger.info(f"  Hysteresis buffer: {hysteresis_buffer:.4f}")
    
    def calculate_enhanced_returns(self, 
                                 probabilities: np.ndarray,
                                 returns: np.ndarray,
                                 threshold: float,
                                 apply_costs: bool = True,
                                 apply_slippage: bool = True,
                                 apply_holding_period: bool = True) -> Dict[str, np.ndarray]:
        """
        Calculate enhanced trading returns with transaction costs and turnover control
        
        Args:
            probabilities: Ensemble probabilities
            returns: Asset returns
            threshold: Trading threshold
            apply_costs: Whether to apply transaction costs
            apply_slippage: Whether to apply slippage
            apply_holding_period: Whether to enforce minimum holding period
        
        Returns:
            Dictionary containing:
            - strategy_returns: Raw strategy returns
            - net_returns: Returns after costs
            - positions: Position sizes
            - costs: Transaction costs
            - slippage_costs: Slippage costs
            - turnover: Position changes
        """
        n_samples = len(probabilities)
        
        # Handle NaN values (warm-up period)
        valid_mask = ~np.isnan(probabilities)
        if not np.any(valid_mask):
            logger.error("⚠️  All probabilities are NaN - no valid predictions!")
            return self._empty_results(n_samples)
        
        # Calculate initial positions (only for valid probabilities)
        positions = self._calculate_positions(probabilities, threshold)
        
        # Set positions to 0 for warm-up period (NaN probabilities)
        positions[~valid_mask] = 0
        
        # Apply holding period constraint if enabled
        if apply_holding_period and self.min_hold_days > 1:
            positions = self._apply_holding_period(positions, self.min_hold_days)
        
        # Apply hysteresis buffer to avoid flip-flopping
        positions = self._apply_hysteresis(positions, threshold)
        
        # Calculate strategy returns
        strategy_returns = positions * returns
        
        # Initialize cost arrays
        transaction_costs = np.zeros(n_samples)
        slippage_costs = np.zeros(n_samples)
        turnover = np.zeros(n_samples)
        
        if apply_costs or apply_slippage:
            # Calculate position changes
            position_changes = np.diff(positions, prepend=0)
            turnover = np.abs(position_changes)
            
            # Apply transaction costs
            if apply_costs:
                transaction_costs = turnover * self.cost_per_trade
            
            # Apply slippage costs
            if apply_slippage:
                slippage_costs = turnover * self.slippage
        
        # Calculate net returns
        net_returns = strategy_returns - transaction_costs - slippage_costs
        
        # Log diagnostics for debugging
        self._log_trading_diagnostics(positions, returns, strategy_returns, net_returns, 
                                    transaction_costs, slippage_costs, turnover)
        
        return {
            'strategy_returns': strategy_returns,
            'net_returns': net_returns,
            'positions': positions,
            'costs': transaction_costs,
            'slippage_costs': slippage_costs,
            'turnover': turnover
        }
    
    def _calculate_positions(self, probabilities: np.ndarray, threshold: float) -> np.ndarray:
        """Calculate position sizes based on probabilities and threshold"""
        # Binary signal: long if p > threshold
        signals = (probabilities > threshold).astype(float)
        
        # Position size: (p - threshold) / (1 - threshold), clipped to [0, 1]
        position_sizes = np.clip((probabilities - threshold) / (1 - threshold), 0, 1)
        
        # Apply signals to position sizes
        positions = signals * position_sizes
        
        return positions
    
    def _apply_holding_period(self, positions: np.ndarray, min_hold_days: int) -> np.ndarray:
        """Apply minimum holding period constraint"""
        if min_hold_days <= 1:
            return positions
        
        n_samples = len(positions)
        constrained_positions = positions.copy()
        
        for i in range(n_samples):
            if i > 0 and positions[i] != positions[i-1]:
                # Check if we can change position (respecting holding period)
                if i < min_hold_days:
                    # Can't change position yet, keep previous
                    constrained_positions[i] = constrained_positions[i-1]
                else:
                    # Check if enough time has passed since last change
                    last_change_idx = i - 1
                    while last_change_idx > 0 and constrained_positions[last_change_idx] == constrained_positions[last_change_idx-1]:
                        last_change_idx -= 1
                    
                    if i - last_change_idx < min_hold_days:
                        # Keep previous position
                        constrained_positions[i] = constrained_positions[i-1]
        
        return constrained_positions
    
    def _apply_hysteresis(self, positions: np.ndarray, threshold: float) -> np.ndarray:
        """Apply hysteresis buffer to avoid flip-flopping"""
        if self.hysteresis_buffer <= 0:
            return positions
        
        n_samples = len(positions)
        smoothed_positions = positions.copy()
        
        for i in range(1, n_samples):
            prev_pos = smoothed_positions[i-1]
            curr_pos = positions[i]
            
            # If position change is small, keep previous position
            if abs(curr_pos - prev_pos) < self.hysteresis_buffer:
                smoothed_positions[i] = prev_pos
        
        return smoothed_positions
    
    def _log_trading_diagnostics(self, positions: np.ndarray, returns: np.ndarray,
                                strategy_returns: np.ndarray, net_returns: np.ndarray,
                                transaction_costs: np.ndarray, slippage_costs: np.ndarray,
                                turnover: np.ndarray):
        """Log trading diagnostics for debugging"""
        # Check for zero volatility (infinite Sharpe)
        strategy_std = np.std(strategy_returns)
        net_std = np.std(net_returns)
        
        if strategy_std == 0:
            logger.error("⚠️  ZERO VOLATILITY DETECTED in strategy returns!")
            logger.error(f"  Positions: min={positions.min():.4f}, max={positions.max():.4f}, mean={positions.mean():.4f}")
            logger.error(f"  Returns: min={returns.min():.4f}, max={returns.max():.4f}, mean={returns.mean():.4f}")
            logger.error(f"  Strategy returns: all values = {strategy_returns[0] if len(strategy_returns) > 0 else 'N/A'}")
        
        if net_std == 0:
            logger.error("⚠️  ZERO VOLATILITY DETECTED in net returns!")
            logger.error(f"  Transaction costs: total={transaction_costs.sum():.6f}, mean={transaction_costs.mean():.6f}")
            logger.error(f"  Slippage costs: total={slippage_costs.sum():.6f}, mean={slippage_costs.mean():.6f}")
        
        # Log cost impact
        total_costs = transaction_costs.sum() + slippage_costs.sum()
        total_strategy_return = strategy_returns.sum()
        cost_drag = total_costs / abs(total_strategy_return) if total_strategy_return != 0 else 0
        
        logger.info(f"Trading Diagnostics:")
        logger.info(f"  Strategy return: {total_strategy_return:.6f}")
        logger.info(f"  Total costs: {total_costs:.6f} ({cost_drag:.2%} of strategy return)")
        logger.info(f"  Annualized turnover: {turnover.sum() * 252 / len(turnover):.2f}")
        logger.info(f"  Strategy volatility: {strategy_std:.6f}")
        logger.info(f"  Net volatility: {net_std:.6f}")
    
    def calculate_robust_sharpe(self, returns: np.ndarray, 
                               annualization_factor: float = 252.0) -> Dict[str, float]:
        """
        Calculate robust Sharpe ratio with comprehensive diagnostics
        
        Args:
            returns: Daily returns
            annualization_factor: Factor to annualize (default: 252 for daily data)
        
        Returns:
            Dictionary with Sharpe ratio and diagnostics
        """
        if len(returns) == 0:
            return {'sharpe_ratio': 0, 'error': 'Empty returns array'}
        
        # Check for zero volatility
        returns_std = np.std(returns)
        if returns_std == 0:
            logger.error("⚠️  ZERO VOLATILITY - Cannot calculate Sharpe ratio!")
            logger.error(f"  Returns array: length={len(returns)}, all values={returns[0] if len(returns) > 0 else 'N/A'}")
            return {'sharpe_ratio': float('inf'), 'error': 'Zero volatility', 'diagnostics': self._get_zero_volatility_diagnostics(returns)}
        
        # Check for infinite or NaN values
        if np.any(np.isinf(returns)) or np.any(np.isnan(returns)):
            logger.error("⚠️  INFINITE OR NAN VALUES in returns!")
            return {'sharpe_ratio': 0, 'error': 'Invalid values in returns'}
        
        # Calculate basic statistics
        returns_mean = np.mean(returns)
        returns_std = np.std(returns, ddof=1)  # Sample standard deviation
        
        # Calculate Sharpe ratio
        sharpe_ratio = (returns_mean / returns_std) * np.sqrt(annualization_factor)
        
        # Additional diagnostics
        diagnostics = {
            'returns_length': len(returns),
            'returns_mean': returns_mean,
            'returns_std': returns_std,
            'returns_min': np.min(returns),
            'returns_max': np.max(returns),
            'zero_returns_count': np.sum(returns == 0),
            'positive_returns_count': np.sum(returns > 0),
            'negative_returns_count': np.sum(returns < 0),
            'annualization_factor': annualization_factor,
            'daily_sharpe': returns_mean / returns_std,
            'annualized_sharpe': sharpe_ratio
        }
        
        # Log diagnostics
        logger.info(f"Sharpe Ratio Calculation:")
        logger.info(f"  Daily mean: {returns_mean:.6f}")
        logger.info(f"  Daily std: {returns_std:.6f}")
        logger.info(f"  Daily Sharpe: {diagnostics['daily_sharpe']:.6f}")
        logger.info(f"  Annualized Sharpe: {sharpe_ratio:.6f}")
        logger.info(f"  Zero returns: {diagnostics['zero_returns_count']}/{len(returns)} ({diagnostics['zero_returns_count']/len(returns)*100:.1f}%)")
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'diagnostics': diagnostics
        }
    
    def _get_zero_volatility_diagnostics(self, returns: np.ndarray) -> Dict[str, Any]:
        """Get diagnostics when volatility is zero"""
        return {
            'returns_length': len(returns),
            'returns_unique_values': np.unique(returns),
            'returns_mode': float(stats.mode(returns)[0]) if len(returns) > 0 else 0,
            'all_same': len(np.unique(returns)) == 1,
            'zero_count': np.sum(returns == 0),
            'non_zero_count': np.sum(returns != 0)
        }
    
    def plot_probability_histograms(self, probabilities: np.ndarray, 
                                  threshold: float = 0.5,
                                  save_path: Optional[str] = None) -> None:
        """Plot histograms of ensemble probabilities for threshold analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall probability distribution
        ax1.hist(probabilities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
        ax1.set_xlabel('Probability')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Ensemble Probability Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Zoom in around threshold
        threshold_range = 0.1  # ±0.1 around threshold
        mask = (probabilities >= threshold - threshold_range) & (probabilities <= threshold + threshold_range)
        threshold_probs = probabilities[mask]
        
        if len(threshold_probs) > 0:
            ax2.hist(threshold_probs, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
            ax2.set_xlabel('Probability')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Probability Distribution Around Threshold (±{threshold_range})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No probabilities in threshold range', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'Probability Distribution Around Threshold (±{threshold_range})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Probability histograms saved to: {save_path}")
        
        plt.show()
    
    def compare_calibration_methods(self, probabilities: np.ndarray, 
                                  targets: np.ndarray,
                                  cv_splits: int = 5) -> Dict[str, float]:
        """
        Compare Platt vs Isotonic calibration methods
        
        Args:
            probabilities: Raw probabilities
            targets: Binary targets
            cv_splits: Number of CV splits
        
        Returns:
            Dictionary with Brier scores for each method
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import brier_score_loss
        
        # Create dummy classifier for calibration comparison
        from sklearn.ensemble import RandomForestClassifier
        dummy_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        # Test both calibration methods
        methods = ['sigmoid', 'isotonic']
        results = {}
        
        for method in methods:
            try:
                # Create calibrated classifier
                calibrated = CalibratedClassifierCV(
                    dummy_classifier, method=method, cv=tscv
                )
                
                # Fit and predict
                calibrated.fit(probabilities.reshape(-1, 1), targets)
                calibrated_probs = calibrated.predict_proba(probabilities.reshape(-1, 1))[:, 1]
                
                # Calculate Brier score
                brier_score = brier_score_loss(targets, calibrated_probs)
                results[method] = brier_score
                
                logger.info(f"  {method.capitalize()} calibration Brier score: {brier_score:.6f}")
                
            except Exception as e:
                logger.warning(f"  {method.capitalize()} calibration failed: {e}")
                results[method] = float('inf')
        
        # Determine best method
        if results['sigmoid'] < results['isotonic']:
            best_method = 'sigmoid'
            logger.info(f"✅ Platt (sigmoid) calibration performs better")
        else:
            best_method = 'isotonic'
            logger.info(f"✅ Isotonic calibration performs better")
        
        results['best_method'] = best_method
        return results
    
    def _empty_results(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Return empty results when no valid predictions"""
        return {
            'strategy_returns': np.zeros(n_samples),
            'net_returns': np.zeros(n_samples),
            'positions': np.zeros(n_samples),
            'costs': np.zeros(n_samples),
            'slippage_costs': np.zeros(n_samples),
            'turnover': np.zeros(n_samples)
        }

def calculate_cost_impact_analysis(returns_without_costs: np.ndarray,
                                 returns_with_costs: np.ndarray,
                                 transaction_costs: np.ndarray,
                                 slippage_costs: np.ndarray) -> Dict[str, float]:
    """
    Analyze the impact of transaction costs on performance
    
    Args:
        returns_without_costs: Returns before costs
        returns_with_costs: Returns after costs
        transaction_costs: Transaction costs array
        slippage_costs: Slippage costs array
    
    Returns:
        Dictionary with cost impact analysis
    """
    # Calculate Sharpe ratios
    def calculate_sharpe(returns):
        if np.std(returns) > 0:
            return (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        return 0
    
    sharpe_without_costs = calculate_sharpe(returns_without_costs)
    sharpe_with_costs = calculate_sharpe(returns_with_costs)
    
    # Calculate cost metrics
    total_transaction_costs = np.sum(transaction_costs)
    total_slippage_costs = np.sum(slippage_costs)
    total_costs = total_transaction_costs + total_slippage_costs
    
    # Calculate cost drag
    total_return_without_costs = np.sum(returns_without_costs)
    cost_drag = total_costs / abs(total_return_without_costs) if total_return_without_costs != 0 else 0
    
    # Annualized metrics
    n_days = len(returns_without_costs)
    annualized_costs = total_costs * 252 / n_days
    annualized_turnover = np.sum(np.abs(np.diff(returns_without_costs > 0, prepend=False))) * 252 / n_days
    
    analysis = {
        'sharpe_without_costs': sharpe_without_costs,
        'sharpe_with_costs': sharpe_with_costs,
        'sharpe_degradation': sharpe_without_costs - sharpe_with_costs,
        'total_transaction_costs': total_transaction_costs,
        'total_slippage_costs': total_slippage_costs,
        'total_costs': total_costs,
        'cost_drag_pct': cost_drag * 100,
        'annualized_costs': annualized_costs,
        'annualized_turnover': annualized_turnover,
        'cost_per_trade': total_costs / max(1, annualized_turnover)
    }
    
    logger.info(f"Cost Impact Analysis:")
    logger.info(f"  Sharpe without costs: {sharpe_without_costs:.3f}")
    logger.info(f"  Sharpe with costs: {sharpe_with_costs:.3f}")
    logger.info(f"  Sharpe degradation: {analysis['sharpe_degradation']:.3f}")
    logger.info(f"  Total costs: {total_costs:.6f} ({cost_drag:.2%} of returns)")
    logger.info(f"  Annualized costs: {annualized_costs:.6f}")
    logger.info(f"  Annualized turnover: {annualized_turnover:.2f}")
    
    return analysis
