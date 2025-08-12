"""
Walk-Forward Analysis Module for the ML Trading System
Implements time series cross-validation with multi-threading optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    WALKFORWARD_CONFIG, SYSTEM_CONFIG, PERFORMANCE_TARGETS,
    PATHS, LOGGING_CONFIG
)
from models import ModelEnsemble
from performance import PerformanceAnalyzer

# Configure logging
logging.basicConfig(
    level=LOGGING_CONFIG['level'],
    format=LOGGING_CONFIG['format']
)
logger = logging.getLogger(__name__)

class WalkForwardAnalyzer:
    """
    Advanced walk-forward analysis with multi-threading optimization
    Designed for M1 chip performance
    """
    
    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.performance_analyzer = PerformanceAnalyzer()
        self.results = {}
        self.models = {}
        self.predictions = {}
        self.performance_metrics = {}
        
        # Configuration
        self.initial_train_size = WALKFORWARD_CONFIG['initial_train_size']
        self.step_size = WALKFORWARD_CONFIG['step_size']
        self.min_train_size = WALKFORWARD_CONFIG['min_train_size']
        self.validation_split = WALKFORWARD_CONFIG['validation_split']
        
        logger.info(f"WalkForwardAnalyzer initialized with {self.n_jobs} jobs")
    
    def run_walkforward_analysis(self, X: np.ndarray, y: np.ndarray, 
                                dates: Optional[pd.DatetimeIndex] = None,
                                ensemble_method: str = 'Voting') -> Dict[str, Any]:
        """
        Run comprehensive walk-forward analysis
        """
        logger.info("Starting walk-forward analysis...")
        
        n_samples = len(X)
        logger.info(f"Total samples: {n_samples}")
        logger.info(f"Initial training size: {self.initial_train_size}")
        logger.info(f"Step size: {self.step_size}")
        
        # Validate parameters
        if self.initial_train_size >= n_samples:
            raise ValueError("Initial training size must be less than total samples")
        
        # Generate walk-forward windows
        windows = self._generate_windows(n_samples)
        logger.info(f"Generated {len(windows)} walk-forward windows")
        
        # Run analysis in parallel
        results = self._run_parallel_analysis(X, y, dates, windows, ensemble_method)
        
        # Process results
        self._process_results(results, dates)
        
        # Calculate overall performance
        overall_performance = self._calculate_overall_performance()
        
        logger.info("Walk-forward analysis completed")
        
        return {
            'results': self.results,
            'performance_metrics': self.performance_metrics,
            'overall_performance': overall_performance,
            'models': self.models,
            'predictions': self.predictions
        }
    
    def _generate_windows(self, n_samples: int) -> List[Dict[str, int]]:
        """Generate walk-forward windows"""
        windows = []
        
        train_start = 0
        train_end = self.initial_train_size
        
        while train_end < n_samples:
            # Validation split
            val_start = train_start + int((train_end - train_start) * (1 - self.validation_split))
            val_end = train_end
            
            # Test period
            test_start = train_end
            test_end = min(test_start + self.step_size, n_samples)
            
            window = {
                'train_start': train_start,
                'train_end': train_end,
                'val_start': val_start,
                'val_end': val_end,
                'test_start': test_start,
                'test_end': test_end,
                'window_id': len(windows)
            }
            
            windows.append(window)
            
            # Move to next window
            train_start += self.step_size
            train_end = min(train_start + self.initial_train_size, n_samples)
            
            # Ensure minimum training size
            if train_end - train_start < self.min_train_size:
                break
        
        return windows
    
    def _run_parallel_analysis(self, X: np.ndarray, y: np.ndarray,
                              dates: Optional[pd.DatetimeIndex],
                              windows: List[Dict[str, int]],
                              ensemble_method: str) -> List[Dict[str, Any]]:
        """Run walk-forward analysis in parallel"""
        
        def analyze_window(window):
            """Analyze a single window"""
            try:
                # Extract data for this window
                X_train = X[window['train_start']:window['train_end']]
                y_train = y[window['train_start']:window['train_end']]
                
                X_val = X[window['val_start']:window['val_end']]
                y_val = y[window['val_start']:window['val_end']]
                
                X_test = X[window['test_start']:window['test_end']]
                y_test = y[window['test_start']:window['test_end']]
                
                # Create and optimize ensemble
                ensemble = ModelEnsemble(n_jobs=1)  # Single job per window
                
                # Optimize individual models
                optimization_results = ensemble.optimize_individual_models(
                    X_train, y_train, cv_folds=3
                )
                
                # Create ensemble
                ensemble.create_ensemble(ensemble_method)
                
                # Fit ensemble
                ensemble.fit_ensemble(X_train, y_train)
                
                # Make predictions
                y_train_pred = ensemble.predict(X_train)
                y_val_pred = ensemble.predict(X_val)
                y_test_pred = ensemble.predict(X_test)
                
                # Calculate performance metrics
                train_metrics = ensemble.evaluate_model(X_train, y_train)
                val_metrics = ensemble.evaluate_model(X_val, y_val)
                test_metrics = ensemble.evaluate_model(X_test, y_test)
                
                # Store results
                window_result = {
                    'window_id': window['window_id'],
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'predictions': {
                        'train': y_train_pred,
                        'validation': y_val_pred,
                        'test': y_test_pred
                    },
                    'actual': {
                        'train': y_train,
                        'validation': y_val,
                        'test': y_test
                    },
                    'model': ensemble,
                    'window_info': window
                }
                
                return window_result
                
            except Exception as e:
                logger.error(f"Error in window {window['window_id']}: {e}")
                return None
        
        # Run analysis in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(tqdm(
                executor.map(analyze_window, windows),
                total=len(windows),
                desc="Walk-forward analysis"
            ))
        
        # Filter out failed windows
        results = [r for r in results if r is not None]
        logger.info(f"Successfully completed {len(results)} windows")
        
        return results
    
    def _process_results(self, results: List[Dict[str, Any]], 
                        dates: Optional[pd.DatetimeIndex]) -> None:
        """Process and organize results"""
        logger.info("Processing results...")
        
        for result in results:
            window_id = result['window_id']
            
            # Store model
            self.models[window_id] = result['model']
            
            # Store predictions
            self.predictions[window_id] = result['predictions']
            
            # Store performance metrics
            self.performance_metrics[window_id] = {
                'train': result['train_metrics'],
                'validation': result['validation_metrics'],
                'test': result['test_metrics']
            }
            
            # Store detailed results
            self.results[window_id] = result
        
        logger.info(f"Processed {len(self.results)} window results")
    
    def _calculate_overall_performance(self) -> Dict[str, Any]:
        """Calculate overall performance across all windows"""
        logger.info("Calculating overall performance...")
        
        # Collect all test predictions and actual values
        all_predictions = []
        all_actuals = []
        all_dates = []
        
        for window_id, result in self.results.items():
            test_pred = result['predictions']['test']
            test_actual = result['actual']['test']
            
            all_predictions.extend(test_pred)
            all_actuals.extend(test_actual)
            
            # Add dates if available
            if 'dates' in result:
                all_dates.extend(result['dates']['test'])
        
        # Convert to arrays
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        # Calculate overall metrics
        overall_metrics = self.performance_analyzer.calculate_metrics(
            all_actuals, all_predictions
        )
        
        # Calculate rolling performance
        if len(all_predictions) > 252:  # At least 1 year
            rolling_metrics = self._calculate_rolling_performance(
                all_predictions, all_actuals
            )
            overall_metrics.update(rolling_metrics)
        
        # Check performance targets
        performance_check = self._check_performance_targets(overall_metrics)
        overall_metrics['targets_met'] = performance_check
        
        logger.info("Overall performance calculated")
        return overall_metrics
    
    def _calculate_rolling_performance(self, predictions: np.ndarray, 
                                     actuals: np.ndarray) -> Dict[str, Any]:
        """Calculate rolling performance metrics"""
        rolling_metrics = {}
        
        # Rolling Sharpe ratio (1 year window)
        if len(predictions) >= 252:
            rolling_sharpe = []
            for i in range(252, len(predictions)):
                window_pred = predictions[i-252:i]
                window_actual = actuals[i-252:i]
                
                # Calculate returns
                returns = window_actual * np.sign(window_pred)  # Directional returns
                
                if np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                    rolling_sharpe.append(sharpe)
                else:
                    rolling_sharpe.append(0)
            
            rolling_metrics['rolling_sharpe_mean'] = np.mean(rolling_sharpe)
            rolling_metrics['rolling_sharpe_std'] = np.std(rolling_sharpe)
            rolling_metrics['rolling_sharpe_min'] = np.min(rolling_sharpe)
            rolling_metrics['rolling_sharpe_max'] = np.max(rolling_sharpe)
        
        return rolling_metrics
    
    def _check_performance_targets(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Check if performance targets are met"""
        targets = PERFORMANCE_TARGETS
        
        checks = {
            'sharpe_ratio': metrics.get('sharpe_ratio', 0) >= targets['min_sharpe'],
            'cagr': metrics.get('cagr', 0) >= targets['min_cagr'],
            'max_drawdown': metrics.get('max_drawdown', -1) >= targets['max_drawdown'],
            'win_rate': metrics.get('win_rate', 0) >= targets['min_win_rate'],
            'volatility': metrics.get('volatility', 1) <= targets['max_volatility']
        }
        
        return checks
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.performance_metrics:
            return {}
        
        summary = {
            'n_windows': len(self.performance_metrics),
            'overall_performance': self._calculate_overall_performance(),
            'window_performance': {}
        }
        
        # Aggregate window performance
        for window_id, metrics in self.performance_metrics.items():
            summary['window_performance'][window_id] = {
                'test_sharpe': metrics['test'].get('sharpe_ratio', 0),
                'test_cagr': metrics['test'].get('cagr', 0),
                'test_win_rate': metrics['test'].get('directional_accuracy', 0)
            }
        
        return summary
    
    def plot_performance(self, save_path: Optional[str] = None) -> None:
        """Plot performance metrics"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Collect data
            window_ids = list(self.performance_metrics.keys())
            test_sharpes = [self.performance_metrics[w]['test'].get('sharpe_ratio', 0) 
                           for w in window_ids]
            test_cagrs = [self.performance_metrics[w]['test'].get('cagr', 0) 
                         for w in window_ids]
            test_win_rates = [self.performance_metrics[w]['test'].get('directional_accuracy', 0) 
                             for w in window_ids]
            
            # Plot 1: Sharpe Ratio over time
            axes[0, 0].plot(window_ids, test_sharpes, marker='o', linewidth=2)
            axes[0, 0].axhline(y=PERFORMANCE_TARGETS['min_sharpe'], 
                              color='red', linestyle='--', 
                              label=f"Target: {PERFORMANCE_TARGETS['min_sharpe']}")
            axes[0, 0].set_title('Sharpe Ratio Over Time')
            axes[0, 0].set_xlabel('Window ID')
            axes[0, 0].set_ylabel('Sharpe Ratio')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Plot 2: CAGR over time
            axes[0, 1].plot(window_ids, test_cagrs, marker='s', linewidth=2, color='green')
            axes[0, 1].axhline(y=PERFORMANCE_TARGETS['min_cagr'], 
                              color='red', linestyle='--', 
                              label=f"Target: {PERFORMANCE_TARGETS['min_cagr']:.1%}")
            axes[0, 1].set_title('CAGR Over Time')
            axes[0, 1].set_xlabel('Window ID')
            axes[0, 1].set_ylabel('CAGR')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Plot 3: Win Rate over time
            axes[1, 0].plot(window_ids, test_win_rates, marker='^', linewidth=2, color='orange')
            axes[1, 0].axhline(y=PERFORMANCE_TARGETS['min_win_rate'], 
                              color='red', linestyle='--', 
                              label=f"Target: {PERFORMANCE_TARGETS['min_win_rate']:.1%}")
            axes[1, 0].set_title('Win Rate Over Time')
            axes[1, 0].set_xlabel('Window ID')
            axes[1, 0].set_ylabel('Win Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Plot 4: Performance distribution
            axes[1, 1].hist(test_sharpes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 1].axvline(x=PERFORMANCE_TARGETS['min_sharpe'], 
                              color='red', linestyle='--', 
                              label=f"Target: {PERFORMANCE_TARGETS['min_sharpe']}")
            axes[1, 1].set_title('Sharpe Ratio Distribution')
            axes[1, 1].set_xlabel('Sharpe Ratio')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Performance plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
    
    def save_results(self, filepath: str) -> None:
        """Save walk-forward results"""
        import joblib
        
        # Prepare data for saving (remove non-serializable objects)
        save_data = {
            'results': self.results,
            'performance_metrics': self.performance_metrics,
            'predictions': self.predictions,
            'config': {
                'initial_train_size': self.initial_train_size,
                'step_size': self.step_size,
                'min_train_size': self.min_train_size,
                'validation_split': self.validation_split
            }
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """Load walk-forward results"""
        import joblib
        
        save_data = joblib.load(filepath)
        
        self.results = save_data['results']
        self.performance_metrics = save_data['performance_metrics']
        self.predictions = save_data['predictions']
        
        # Restore configuration
        config = save_data['config']
        self.initial_train_size = config['initial_train_size']
        self.step_size = config['step_size']
        self.min_train_size = config['min_train_size']
        self.validation_split = config['validation_split']
        
        logger.info(f"Results loaded from {filepath}")
