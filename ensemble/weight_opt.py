"""
Weight Optimization Module
Optimizes ensemble weights for trading performance metrics
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class WeightOptimizer:
    """Optimizes ensemble weights for trading performance"""
    
    def __init__(self, method: str = 'sharpe', cost_per_trade: float = 0.001, 
                 slippage: float = 0.0005):
        self.method = method
        self.cost_per_trade = cost_per_trade
        self.slippage = slippage
        
    def optimize_weights(self, proba_matrix: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Optimize weights based on specified method"""
        if self.method == 'sharpe':
            return self._optimize_sharpe(proba_matrix, y)
        elif self.method == 'cagr':
            return self._optimize_cagr(proba_matrix, y)
        elif self.method == 'sharpe_cagr':
            return self._optimize_combined(proba_matrix, y)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
    
    def _optimize_sharpe(self, proba_matrix: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Optimize weights to maximize Sharpe ratio"""
        def objective(weights):
            weights = weights / np.sum(weights)
            ensemble_proba = np.sum(proba_matrix * weights, axis=1)
            returns = self._calculate_trading_returns(ensemble_proba, y)
            
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                return -sharpe
            return 0
        
        n_models = proba_matrix.shape[1]
        initial_weights = np.ones(n_models) / n_models
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(
            objective, initial_weights, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)
            final_sharpe = -result.fun
        else:
            optimal_weights = initial_weights
            final_sharpe = 0
            
        return optimal_weights, {'sharpe': final_sharpe, 'success': result.success}
    
    def _optimize_cagr(self, proba_matrix: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Optimize weights to maximize CAGR"""
        def objective(weights):
            weights = weights / np.sum(weights)
            ensemble_proba = np.sum(proba_matrix * weights, axis=1)
            returns = self._calculate_trading_returns(ensemble_proba, y)
            
            cumulative_returns = np.cumprod(1 + returns)
            if len(cumulative_returns) > 1:
                cagr = (cumulative_returns[-1] / cumulative_returns[0]) ** (252 / len(cumulative_returns)) - 1
                return -cagr
            return 0
        
        n_models = proba_matrix.shape[1]
        initial_weights = np.ones(n_models) / n_models
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(
            objective, initial_weights, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)
            final_cagr = -result.fun
        else:
            optimal_weights = initial_weights
            final_cagr = 0
            
        return optimal_weights, {'cagr': final_cagr, 'success': result.success}
    
    def _optimize_combined(self, proba_matrix: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Optimize weights to maximize combined Sharpe and CAGR"""
        def objective(weights):
            weights = weights / np.sum(weights)
            ensemble_proba = np.sum(proba_matrix * weights, axis=1)
            returns = self._calculate_trading_returns(ensemble_proba, y)
            
            # Sharpe ratio
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe = 0
            
            # CAGR
            cumulative_returns = np.cumprod(1 + returns)
            if len(cumulative_returns) > 1:
                cagr = (cumulative_returns[-1] / cumulative_returns[0]) ** (252 / len(cumulative_returns)) - 1
            else:
                cagr = 0
            
            combined_score = sharpe + cagr
            return -combined_score
        
        n_models = proba_matrix.shape[1]
        initial_weights = np.ones(n_models) / n_models
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(
            objective, initial_weights, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)
            final_score = -result.fun
        else:
            optimal_weights = initial_weights
            final_score = 0
            
        return optimal_weights, {'combined_score': final_score, 'success': result.success}
    
    def _calculate_trading_returns(self, ensemble_proba: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate trading returns based on probability threshold strategy"""
        threshold = 0.5
        signals = (ensemble_proba > threshold).astype(float)
        position_sizes = np.clip((ensemble_proba - threshold) / (1 - threshold), 0, 1)
        weighted_signals = signals * position_sizes
        strategy_returns = weighted_signals * y
        
        signal_changes = np.diff(weighted_signals, prepend=0)
        transaction_costs = np.abs(signal_changes) * (self.cost_per_trade + self.slippage)
        
        return strategy_returns - transaction_costs
