"""
Performance Analysis Module for the ML Trading System
Calculates comprehensive trading performance metrics including Sharpe ratio and CAGR
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for trading strategies
    Calculates Sharpe ratio, CAGR, drawdown, and other key metrics
    """
    
    def __init__(self, risk_free_rate: float = 0.02, trading_days: int = 252):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.annualization_factor = np.sqrt(trading_days)
        
        logger.info(f"PerformanceAnalyzer initialized with {trading_days} trading days")
    
    def calculate_metrics(self, actual_returns: np.ndarray, 
                         predicted_returns: np.ndarray,
                         transaction_costs: float = 0.001) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            actual_returns: Actual asset returns
            predicted_returns: Predicted returns from model
            transaction_costs: Transaction costs per trade (default: 0.1%)
        
        Returns:
            Dictionary of performance metrics
        """
        logger.info("Calculating performance metrics...")
        
        # Calculate strategy returns (directional trading)
        strategy_returns = self._calculate_strategy_returns(
            actual_returns, predicted_returns, transaction_costs
        )
        
        # Basic return metrics
        metrics = self._calculate_return_metrics(strategy_returns)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(strategy_returns)
        metrics.update(risk_metrics)
        
        # Advanced metrics
        advanced_metrics = self._calculate_advanced_metrics(
            strategy_returns, actual_returns, predicted_returns
        )
        metrics.update(advanced_metrics)
        
        # Trading metrics
        trading_metrics = self._calculate_trading_metrics(
            actual_returns, predicted_returns, strategy_returns
        )
        metrics.update(trading_metrics)
        
        logger.info("Performance metrics calculated successfully")
        return metrics
    
    def _calculate_strategy_returns(self, actual_returns: np.ndarray,
                                  predicted_returns: np.ndarray,
                                  transaction_costs: float) -> np.ndarray:
        """Calculate strategy returns based on predictions"""
        
        # Create trading signals (1 for long, -1 for short, 0 for no position)
        signals = np.sign(predicted_returns)
        
        # Calculate strategy returns
        strategy_returns = signals * actual_returns
        
        # Apply transaction costs when signals change
        signal_changes = np.diff(signals, prepend=0)
        transaction_costs_total = np.abs(signal_changes) * transaction_costs
        
        # Net strategy returns
        net_strategy_returns = strategy_returns - transaction_costs_total
        
        return net_strategy_returns
    
    def _calculate_return_metrics(self, strategy_returns: np.ndarray) -> Dict[str, float]:
        """Calculate basic return metrics"""
        
        # Total return
        total_return = np.prod(1 + strategy_returns) - 1
        
        # Annualized return (CAGR)
        n_periods = len(strategy_returns)
        if n_periods > 1:
            cagr = (1 + total_return) ** (self.trading_days / n_periods) - 1
        else:
            cagr = 0
        
        # Mean return
        mean_return = np.mean(strategy_returns)
        
        # Annualized mean return
        annualized_mean_return = mean_return * self.trading_days
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'mean_return': mean_return,
            'annualized_mean_return': annualized_mean_return
        }
    
    def _calculate_risk_metrics(self, strategy_returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk metrics"""
        
        # Volatility
        volatility = np.std(strategy_returns)
        annualized_volatility = volatility * self.annualization_factor
        
        # Sharpe ratio
        if volatility > 0:
            sharpe_ratio = (np.mean(strategy_returns) - self.risk_free_rate / self.trading_days) / volatility
            sharpe_ratio_annualized = sharpe_ratio * self.annualization_factor
        else:
            sharpe_ratio = 0
            sharpe_ratio_annualized = 0
        
        # Sortino ratio (downside deviation)
        downside_returns = strategy_returns[strategy_returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            downside_deviation = np.std(downside_returns)
            sortino_ratio = (np.mean(strategy_returns) - self.risk_free_rate / self.trading_days) / downside_deviation
            sortino_ratio_annualized = sortino_ratio * self.annualization_factor
        else:
            sortino_ratio = 0
            sortino_ratio_annualized = 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(strategy_returns, 5)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = np.mean(strategy_returns[strategy_returns <= var_95])
        
        return {
            'volatility': volatility,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sharpe_ratio_annualized': sharpe_ratio_annualized,
            'sortino_ratio': sortino_ratio,
            'sortino_ratio_annualized': sortino_ratio_annualized,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def _calculate_advanced_metrics(self, strategy_returns: np.ndarray,
                                  actual_returns: np.ndarray,
                                  predicted_returns: np.ndarray) -> Dict[str, float]:
        """Calculate advanced performance metrics"""
        
        # Information ratio
        excess_returns = strategy_returns - actual_returns
        if np.std(excess_returns) > 0:
            information_ratio = np.mean(excess_returns) / np.std(excess_returns)
            information_ratio_annualized = information_ratio * self.annualization_factor
        else:
            information_ratio = 0
            information_ratio_annualized = 0
        
        # Calmar ratio (CAGR / Max Drawdown)
        if abs(max_drawdown := self._calculate_max_drawdown(strategy_returns)) > 0:
            calmar_ratio = self._calculate_cagr(strategy_returns) / abs(max_drawdown)
        else:
            calmar_ratio = 0
        
        # Sterling ratio (excess return / average drawdown)
        avg_drawdown = self._calculate_average_drawdown(strategy_returns)
        if avg_drawdown > 0:
            sterling_ratio = (np.mean(strategy_returns) - self.risk_free_rate / self.trading_days) / avg_drawdown
        else:
            sterling_ratio = 0
        
        # Treynor ratio (excess return / beta)
        beta = self._calculate_beta(strategy_returns, actual_returns)
        if beta != 0:
            treynor_ratio = (np.mean(strategy_returns) - self.risk_free_rate / self.trading_days) / beta
        else:
            treynor_ratio = 0
        
        # Jensen's alpha
        alpha = self._calculate_jensen_alpha(strategy_returns, actual_returns)
        
        return {
            'information_ratio': information_ratio,
            'information_ratio_annualized': information_ratio_annualized,
            'calmar_ratio': calmar_ratio,
            'sterling_ratio': sterling_ratio,
            'treynor_ratio': treynor_ratio,
            'jensen_alpha': alpha
        }
    
    def _calculate_trading_metrics(self, actual_returns: np.ndarray,
                                 predicted_returns: np.ndarray,
                                 strategy_returns: np.ndarray) -> Dict[str, float]:
        """Calculate trading-specific metrics"""
        
        # Directional accuracy (win rate)
        correct_directions = np.sum(np.sign(actual_returns) == np.sign(predicted_returns))
        directional_accuracy = correct_directions / len(actual_returns)
        
        # Profit factor
        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]
        
        if len(losing_trades) > 0 and np.sum(losing_trades) != 0:
            profit_factor = np.sum(winning_trades) / abs(np.sum(losing_trades))
        else:
            profit_factor = float('inf') if len(winning_trades) > 0 else 0
        
        # Average win/loss ratio
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            avg_win = np.mean(winning_trades)
            avg_loss = abs(np.mean(losing_trades))
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        else:
            win_loss_ratio = 0
        
        # Number of trades
        signal_changes = np.diff(np.sign(predicted_returns), prepend=0)
        n_trades = np.sum(np.abs(signal_changes))
        
        # Hit rate
        if n_trades > 0:
            hit_rate = np.sum(strategy_returns > 0) / n_trades
        else:
            hit_rate = 0
        
        # Maximum consecutive wins/losses
        consecutive_wins = self._calculate_max_consecutive(strategy_returns > 0)
        consecutive_losses = self._calculate_max_consecutive(strategy_returns < 0)
        
        return {
            'directional_accuracy': directional_accuracy,
            'win_rate': directional_accuracy,
            'profit_factor': profit_factor,
            'win_loss_ratio': win_loss_ratio,
            'n_trades': n_trades,
            'hit_rate': hit_rate,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_average_drawdown(self, returns: np.ndarray) -> float:
        """Calculate average drawdown"""
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Calculate average of negative drawdowns
        negative_drawdowns = drawdown[drawdown < 0]
        return np.mean(negative_drawdowns) if len(negative_drawdowns) > 0 else 0
    
    def _calculate_cagr(self, returns: np.ndarray) -> float:
        """Calculate CAGR"""
        total_return = np.prod(1 + returns) - 1
        n_periods = len(returns)
        if n_periods > 1:
            return (1 + total_return) ** (self.trading_days / n_periods) - 1
        return 0
    
    def _calculate_beta(self, strategy_returns: np.ndarray, 
                       market_returns: np.ndarray) -> float:
        """Calculate beta relative to market"""
        if len(strategy_returns) != len(market_returns):
            return 0
        
        covariance = np.cov(strategy_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance if market_variance > 0 else 0
    
    def _calculate_jensen_alpha(self, strategy_returns: np.ndarray,
                               market_returns: np.ndarray) -> float:
        """Calculate Jensen's alpha"""
        beta = self._calculate_beta(strategy_returns, market_returns)
        strategy_mean = np.mean(strategy_returns)
        market_mean = np.mean(market_returns)
        
        return strategy_mean - (self.risk_free_rate / self.trading_days + beta * market_mean)
    
    def _calculate_max_consecutive(self, condition: np.ndarray) -> int:
        """Calculate maximum consecutive occurrences of a condition"""
        max_consecutive = 0
        current_consecutive = 0
        
        for value in condition:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def calculate_rolling_metrics(self, returns: np.ndarray, 
                                window: int = 252) -> Dict[str, np.ndarray]:
        """Calculate rolling performance metrics"""
        
        if len(returns) < window:
            return {}
        
        rolling_metrics = {}
        
        # Rolling Sharpe ratio
        rolling_sharpe = []
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            if np.std(window_returns) > 0:
                sharpe = (np.mean(window_returns) - self.risk_free_rate / self.trading_days) / np.std(window_returns)
                rolling_sharpe.append(sharpe * self.annualization_factor)
            else:
                rolling_sharpe.append(0)
        
        rolling_metrics['rolling_sharpe'] = np.array(rolling_sharpe)
        
        # Rolling volatility
        rolling_vol = []
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            vol = np.std(window_returns) * self.annualization_factor
            rolling_vol.append(vol)
        
        rolling_metrics['rolling_volatility'] = np.array(rolling_vol)
        
        # Rolling drawdown
        rolling_dd = []
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            cumulative = np.cumprod(1 + window_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            rolling_dd.append(np.min(drawdown))
        
        rolling_metrics['rolling_drawdown'] = np.array(rolling_dd)
        
        return rolling_metrics
    
    def generate_performance_report(self, metrics: Dict[str, float]) -> str:
        """Generate a formatted performance report"""
        
        report = "=" * 60 + "\n"
        report += "PERFORMANCE REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Return Metrics
        report += "RETURN METRICS:\n"
        report += "-" * 30 + "\n"
        report += f"Total Return: {metrics.get('total_return', 0):.2%}\n"
        report += f"CAGR: {metrics.get('cagr', 0):.2%}\n"
        report += f"Annualized Mean Return: {metrics.get('annualized_mean_return', 0):.2%}\n\n"
        
        # Risk Metrics
        report += "RISK METRICS:\n"
        report += "-" * 30 + "\n"
        report += f"Sharpe Ratio (Annualized): {metrics.get('sharpe_ratio_annualized', 0):.2f}\n"
        report += f"Sortino Ratio (Annualized): {metrics.get('sortino_ratio_annualized', 0):.2f}\n"
        report += f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
        report += f"Annualized Volatility: {metrics.get('annualized_volatility', 0):.2%}\n"
        report += f"VaR (95%): {metrics.get('var_95', 0):.2%}\n\n"
        
        # Trading Metrics
        report += "TRADING METRICS:\n"
        report += "-" * 30 + "\n"
        report += f"Win Rate: {metrics.get('win_rate', 0):.2%}\n"
        report += f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
        report += f"Win/Loss Ratio: {metrics.get('win_loss_ratio', 0):.2f}\n"
        report += f"Number of Trades: {metrics.get('n_trades', 0):.0f}\n"
        report += f"Hit Rate: {metrics.get('hit_rate', 0):.2%}\n\n"
        
        # Advanced Metrics
        report += "ADVANCED METRICS:\n"
        report += "-" * 30 + "\n"
        report += f"Information Ratio: {metrics.get('information_ratio_annualized', 0):.2f}\n"
        report += f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}\n"
        report += f"Sterling Ratio: {metrics.get('sterling_ratio', 0):.2f}\n"
        report += f"Treynor Ratio: {metrics.get('treynor_ratio', 0):.2f}\n"
        report += f"Jensen's Alpha: {metrics.get('jensen_alpha', 0):.4f}\n\n"
        
        # Performance Targets Check
        report += "PERFORMANCE TARGETS:\n"
        report += "-" * 30 + "\n"
        sharpe_target = 3.0
        cagr_target = 0.25
        
        sharpe_met = metrics.get('sharpe_ratio_annualized', 0) >= sharpe_target
        cagr_met = metrics.get('cagr', 0) >= cagr_target
        
        report += f"Sharpe Ratio ≥ {sharpe_target}: {'✓' if sharpe_met else '✗'}\n"
        report += f"CAGR ≥ {cagr_target:.1%}: {'✓' if cagr_met else '✗'}\n"
        
        report += "=" * 60 + "\n"
        
        return report

