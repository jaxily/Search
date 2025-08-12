"""
Enhanced Ensemble Class for 9-Model Soft-Voting Trading System
Implements proper sklearn Pipelines with TimeSeriesSplit to avoid data leakage
Optimized for trading Sharpe ratio and CAGR with probability-based position sizing
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from typing import Dict, List, Tuple, Any, Optional
import joblib
import logging
import warnings
import sklearn
from ensemble.trading_utils import EnhancedTradingCalculator, calculate_cost_impact_analysis
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTradingEnsemble:
    """
    Enhanced 9-model ensemble optimized for trading performance
    Features:
    - Proper sklearn Pipelines with TimeSeriesSplit (no data leakage)
    - Probability calibration using OOF data only
    - Weight optimization for Sharpe ratio and CAGR
    - Threshold optimization for trading decisions
    - Diversity control via correlation analysis
    - Comprehensive trading metrics
    """
    
    def __init__(self, random_state: int = 42, n_splits: int = 5, 
                 cost_per_trade: float = 0.001, slippage_bps: float = 5.0,
                 min_hold_days: int = 1, hysteresis_buffer: float = 0.02):
        self.random_state = random_state
        self.n_splits = n_splits
        self.models = {}
        self.pipelines = {}
        self.calibrators = {}
        self.optimal_weights = {}
        self.optimal_threshold = 0.5
        self.oof_probabilities = {}
        self.is_fitted = False
        
        # Initialize enhanced trading calculator
        self.trading_calculator = EnhancedTradingCalculator(
            cost_per_trade=cost_per_trade,
            slippage_bps=slippage_bps,
            min_hold_days=min_hold_days,
            hysteresis_buffer=hysteresis_buffer
        )
        
        # Store transaction parameters
        self.cost_per_trade = cost_per_trade
        self.slippage_bps = slippage_bps
        self.min_hold_days = min_hold_days
        self.hysteresis_buffer = hysteresis_buffer
        
        # Initialize models with proper pipelines
        self._initialize_models()
        
        logger.info("EnhancedTradingEnsemble initialized successfully")
        logger.info(f"Transaction costs: {cost_per_trade:.4f}, Slippage: {slippage_bps} bps")
        logger.info(f"Min hold days: {min_hold_days}, Hysteresis: {hysteresis_buffer:.4f}")
    
    def _validate_n_splits(self, n_samples: int) -> int:
        """Validate and adjust n_splits based on data size"""
        # Ensure minimum fold size to prevent zero predictions
        min_fold_size = 200  # Minimum samples per fold to avoid zero predictions
        max_splits = max(1, n_samples // min_fold_size)
        
        if self.n_splits > max_splits:
            adjusted_splits = max_splits
            logger.warning(f"Data size {n_samples} too small for {self.n_splits} splits. "
                          f"Adjusted to {adjusted_splits} splits to ensure minimum fold size {min_fold_size}.")
            return adjusted_splits
        
        # Additional validation for very small datasets
        if n_samples < self.n_splits * 2:
            adjusted_splits = max(1, n_samples // 2)
            logger.warning(f"Data size {n_samples} too small for {self.n_splits} splits. "
                          f"Adjusted to {adjusted_splits} splits.")
            return adjusted_splits
        
        return self.n_splits
    
    def _calculate_warm_up_period(self, n_splits: int, n_samples: int) -> int:
        """Calculate warm-up period to avoid zero predictions from early folds"""
        # For TimeSeriesSplit, the first fold has no training data
        # We need to ensure sufficient training data before making predictions
        min_training_samples = 500  # Minimum samples needed for training
        
        # Calculate warm-up period based on fold structure
        if n_splits == 1:
            warm_up = min_training_samples
        else:
            # First fold size (no training data)
            first_fold_size = n_samples // n_splits
            # Second fold size (minimal training data)
            second_fold_size = n_samples // n_splits
            # Warm-up = first two folds
            warm_up = first_fold_size + second_fold_size
        
        # Ensure minimum warm-up period
        warm_up = max(warm_up, min_training_samples)
        
        # Cap at reasonable size
        warm_up = min(warm_up, n_samples // 3)
        
        return warm_up
    
    def _initialize_models(self):
        """Initialize all 9 models with proper sklearn Pipelines"""
        
        # 1. Random Forest
        rf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced'
            ))
        ])
        self.models['RandomForest'] = rf_pipeline
        
        # 2. Gradient Boosting
        gb_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ))
        ])
        self.models['GradientBoosting'] = gb_pipeline
        
        # 3. XGBoost
        try:
            xgb_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    eval_metric='logloss',
                    use_label_encoder=False
                ))
            ])
            self.models['XGBoost'] = xgb_pipeline
        except ImportError:
            logger.warning("XGBoost not available")
        
        # 4. LightGBM
        try:
            lgb_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    verbose=-1,
                    class_weight='balanced'
                ))
            ])
            self.models['LightGBM'] = lgb_pipeline
        except ImportError:
            logger.warning("LightGBM not available")
        
        # 5. CatBoost
        try:
            cb_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', cb.CatBoostClassifier(
                    iterations=100,
                    learning_rate=0.1,
                    depth=6,
                    random_seed=self.random_state,
                    verbose=False,
                    class_weights=[1, 1]
                ))
            ])
            self.models['CatBoost'] = cb_pipeline
        except ImportError:
            logger.warning("CatBoost not available")
        
        # 6. Logistic Regression (ElasticNet penalty)
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                l1_ratio=0.5,
                C=1.0,
                random_state=self.random_state,
                max_iter=2000,
                class_weight='balanced'
            ))
        ])
        self.models['LogisticRegression'] = lr_pipeline
        
        # 7. Ridge Classifier (using LogisticRegression instead since RidgeClassifier has no predict_proba)
        ridge_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                penalty='l2',
                C=1.0,
                random_state=self.random_state,
                class_weight='balanced'
            ))
        ])
        self.models['RidgeClassifier'] = ridge_pipeline
        
        # 8. SVC (with probability=True)
        svc_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            ))
        ])
        self.models['SVC'] = svc_pipeline
        
        # 9. MLP (Neural Network)
        mlp_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate='adaptive',
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                verbose=0,
                random_state=self.random_state
            ))
        ])
        self.models['MLP'] = mlp_pipeline
        
        logger.info(f"Initialized {len(self.models)} models with proper pipelines")
    
    def fit_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit all models using TimeSeriesSplit to avoid data leakage"""
        logger.info("Fitting all models using TimeSeriesSplit...")
        
        # Store training data for calibration
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Validate and adjust n_splits based on data size
        actual_n_splits = self._validate_n_splits(len(X))
        
        # Calculate warm-up period to avoid zero predictions
        self.warm_up_samples = self._calculate_warm_up_period(actual_n_splits, len(X))
        logger.info(f"Warm-up period: {self.warm_up_samples} samples (first {self.warm_up_samples/len(X)*100:.1f}% of data)")
        
        # Use TimeSeriesSplit for proper temporal validation
        tscv = TimeSeriesSplit(n_splits=actual_n_splits)
        
        # Create a copy of models to avoid dictionary modification during iteration
        models_to_fit = list(self.models.items())
        failed_models = []
        
        for name, pipeline in models_to_fit:
            try:
                logger.info(f"Fitting {name}...")
                
                # Fit pipeline on full data
                pipeline.fit(X, y)
                
                # Generate OOF probabilities for this model using manual TimeSeriesSplit
                oof_proba = self._generate_oof_probabilities_manual(pipeline, X, y, tscv)
                
                # Store OOF probabilities (probability of positive class)
                if oof_proba.shape[1] == 2:
                    self.oof_probabilities[name] = oof_proba[:, 1]
                else:
                    self.oof_probabilities[name] = oof_proba[:, 0]
                
                logger.info(f"{name} fitted successfully")
                
            except Exception as e:
                logger.error(f"Error fitting {name}: {e}")
                # Mark for removal
                failed_models.append(name)
        
        # Remove failed models after iteration
        for name in failed_models:
            if name in self.models:
                del self.models[name]
            if name in self.oof_probabilities:
                del self.oof_probabilities[name]
        
        self.is_fitted = True
        logger.info(f"All models fitted successfully. Active models: {len(self.models)}")
    
    def _generate_oof_probabilities_manual(self, pipeline, X: np.ndarray, y: np.ndarray, tscv) -> np.ndarray:
        """Generate OOF probabilities manually using TimeSeriesSplit with warm-up period handling"""
        oof_proba = np.zeros((len(X), 2))  # 2 classes
        
        # Skip predictions for warm-up period
        warm_up_samples = getattr(self, 'warm_up_samples', 0)
        
        for train_idx, val_idx in tscv.split(X):
            # Skip early folds that have insufficient training data
            if len(train_idx) < 100:  # Need at least 100 training samples
                logger.warning(f"Skipping fold with only {len(train_idx)} training samples")
                continue
                
            # Train on training fold
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            
            # Create a copy of the pipeline for this fold
            from copy import deepcopy
            fold_pipeline = deepcopy(pipeline)
            fold_pipeline.fit(X_train_fold, y_train_fold)
            
            # Predict on validation fold
            val_proba = fold_pipeline.predict_proba(X[val_idx])
            oof_proba[val_idx] = val_proba
        
        # Set warm-up period to NaN (not zero) to indicate missing data
        if warm_up_samples > 0:
            oof_proba[:warm_up_samples] = np.nan
        
        return oof_proba
    
    def calibrate_probabilities(self, method: str = 'auto', plot_histograms: bool = True) -> None:
        """
        Calibrate probabilities using OOF data only (no leakage)
        
        Args:
            method: Calibration method ('isotonic', 'sigmoid', or 'auto')
            plot_histograms: Whether to plot probability histograms
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before calibration")
        
        logger.info(f"Calibrating probabilities using {method} method...")
        
        # Plot probability histograms before calibration if requested
        if plot_histograms and hasattr(self, 'X_train') and hasattr(self, 'y_train'):
            self._plot_pre_calibration_histograms()
        
        # Use TimeSeriesSplit for calibration
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # If auto method, compare both calibration methods
        if method == 'auto':
            logger.info("Comparing calibration methods...")
            best_method = self._compare_calibration_methods()
            method = best_method
            logger.info(f"Selected {method} calibration method")
        
        for name, pipeline in self.models.items():
            try:
                if name in self.oof_probabilities:
                    # Create calibrated model
                    if method == 'isotonic':
                        calibrated_model = CalibratedClassifierCV(
                            pipeline, method='isotonic', cv=tscv
                        )
                    elif method == 'sigmoid':
                        calibrated_model = CalibratedClassifierCV(
                            pipeline, method='sigmoid', cv=tscv
                        )
                    else:
                        logger.warning(f"Unknown calibration method: {method}")
                        continue
                    
                    # Fit calibrated model using the original data (this is correct for sklearn)
                    # The calibration will use the same TimeSeriesSplit internally
                    # We need to get the original data from somewhere - this should be stored
                    if hasattr(self, 'X_train') and hasattr(self, 'y_train'):
                        X_train, y_train = self.X_train, self.y_train
                    else:
                        logger.warning(f"No training data stored for {name}. Skipping calibration.")
                        continue
                    
                    calibrated_model.fit(X_train, y_train)
                    
                    # Store calibrated model
                    self.calibrators[name] = calibrated_model
                    
                    logger.info(f"Calibrated {name} using {method}")
                    
            except Exception as e:
                logger.warning(f"Could not calibrate {name}: {e}")
        
        logger.info(f"Probability calibration completed for {len(self.calibrators)} models")
    
    def _plot_pre_calibration_histograms(self):
        """Plot probability histograms before calibration"""
        try:
            # Get ensemble probabilities before calibration
            if self.oof_probabilities:
                # Use equal weights for initial ensemble
                n_models = len(self.oof_probabilities)
                equal_weights = np.ones(n_models) / n_models
                
                proba_matrix = np.column_stack(list(self.oof_probabilities.values()))
                ensemble_probs = np.sum(proba_matrix * equal_weights, axis=1)
                
                # Plot histograms
                self.trading_calculator.plot_probability_histograms(
                    ensemble_probs, 
                    threshold=0.5,
                    save_path='results/qqq_ensemble/pre_calibration_histograms.png'
                )
        except Exception as e:
            logger.warning(f"Could not plot pre-calibration histograms: {e}")
    
    def _compare_calibration_methods(self) -> str:
        """Compare Platt vs Isotonic calibration methods"""
        try:
            # Get ensemble probabilities and targets
            if not self.oof_probabilities or not hasattr(self, 'y_train'):
                logger.warning("Cannot compare calibration methods - missing data")
                return 'isotonic'  # Default fallback
            
            # Use equal weights for comparison
            n_models = len(self.oof_probabilities)
            equal_weights = np.ones(n_models) / n_models
            
            proba_matrix = np.column_stack(list(self.oof_probabilities.values()))
            ensemble_probs = np.sum(proba_matrix * equal_weights, axis=1)
            
            # Compare methods
            results = self.trading_calculator.compare_calibration_methods(
                ensemble_probs, self.y_train, cv_splits=self.n_splits
            )
            
            return results.get('best_method', 'isotonic')
            
        except Exception as e:
            logger.warning(f"Error comparing calibration methods: {e}")
            return 'isotonic'  # Default fallback
    
    def optimize_ensemble_weights(self, y: np.ndarray, 
                                method: str = 'sharpe',
                                cost_per_trade: float = 0.001,
                                slippage: float = 0.0005) -> Dict[str, Any]:
        """
        Optimize ensemble weights to maximize trading performance
        
        Args:
            y: Target values (returns)
            method: Optimization method ('sharpe', 'cagr', 'sharpe_cagr')
            cost_per_trade: Transaction cost per trade
            slippage: Slippage cost per trade
        
        Returns:
            Dictionary with optimal weights and performance metrics
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before weight optimization")
        
        logger.info(f"Optimizing ensemble weights using {method} method...")
        
        # Convert OOF probabilities to matrix
        proba_matrix = np.column_stack(list(self.oof_probabilities.values()))
        model_names = list(self.oof_probabilities.keys())
        
        # Optimize weights
        if method == 'sharpe':
            optimal_weights = self._optimize_weights_sharpe(
                proba_matrix, y, cost_per_trade, slippage
            )
        elif method == 'cagr':
            optimal_weights = self._optimize_weights_cagr(
                proba_matrix, y, cost_per_trade, slippage
            )
        elif method == 'sharpe_cagr':
            optimal_weights = self._optimize_weights_sharpe_cagr(
                proba_matrix, y, cost_per_trade, slippage
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Calculate ensemble probabilities with optimal weights
        ensemble_proba = np.sum(proba_matrix * optimal_weights, axis=1)
        
        # Find optimal threshold
        optimal_threshold = self._optimize_threshold(ensemble_proba, y, cost_per_trade, slippage)
        
        # Calculate final performance metrics
        final_metrics = self._calculate_trading_metrics(
            ensemble_proba, y, optimal_threshold, cost_per_trade, slippage
        )
        
        # Store optimal weights and threshold
        self.optimal_weights = dict(zip(model_names, optimal_weights))
        self.optimal_threshold = optimal_threshold
        
        logger.info(f"Weight optimization completed. Optimal threshold: {optimal_threshold:.3f}")
        
        return {
            'optimal_weights': self.optimal_weights,
            'optimal_threshold': optimal_threshold,
            'performance_metrics': final_metrics,
            'model_names': model_names
        }
    
    def _optimize_weights_sharpe(self, proba_matrix: np.ndarray, y: np.ndarray,
                                cost_per_trade: float, slippage: float) -> np.ndarray:
        """Optimize weights to maximize Sharpe ratio"""
        def objective(weights):
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Calculate ensemble probabilities
            ensemble_proba = np.sum(proba_matrix * weights, axis=1)
            
            # Calculate trading returns
            returns = self._calculate_trading_returns(ensemble_proba, y, 0.5, cost_per_trade, slippage)
            
            # Calculate Sharpe ratio (negative for minimization)
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                return -sharpe
            else:
                return 0
        
        # Initial weights (equal)
        n_models = proba_matrix.shape[1]
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        result = minimize(
            objective, initial_weights, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)  # Ensure normalization
            logger.info(f"Sharpe optimization successful. Final Sharpe: {-result.fun:.3f}")
        else:
            logger.warning("Sharpe optimization failed. Using equal weights.")
            optimal_weights = np.ones(n_models) / n_models
        
        return optimal_weights
    
    def _optimize_weights_cagr(self, proba_matrix: np.ndarray, y: np.ndarray,
                              cost_per_trade: float, slippage: float) -> np.ndarray:
        """Optimize weights to maximize CAGR"""
        def objective(weights):
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Calculate ensemble probabilities
            ensemble_proba = np.sum(proba_matrix * weights, axis=1)
            
            # Calculate trading returns
            returns = self._calculate_trading_returns(ensemble_proba, y, 0.5, cost_per_trade, slippage)
            
            # Calculate CAGR (negative for minimization)
            cumulative_returns = np.cumprod(1 + returns)
            if len(cumulative_returns) > 1:
                cagr = (cumulative_returns[-1] / cumulative_returns[0]) ** (252 / len(cumulative_returns)) - 1
                return -cagr
            else:
                return 0
        
        # Initial weights (equal)
        n_models = proba_matrix.shape[1]
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        result = minimize(
            objective, initial_weights, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)  # Ensure normalization
            logger.info(f"CAGR optimization successful. Final CAGR: {-result.fun:.3f}")
        else:
            logger.warning("CAGR optimization failed. Using equal weights.")
            optimal_weights = np.ones(n_models) / n_models
        
        return optimal_weights
    
    def _optimize_weights_sharpe_cagr(self, proba_matrix: np.ndarray, y: np.ndarray,
                                    cost_per_trade: float, slippage: float) -> np.ndarray:
        """Optimize weights to maximize combined Sharpe and CAGR"""
        def objective(weights):
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Calculate ensemble probabilities
            ensemble_proba = np.sum(proba_matrix * weights, axis=1)
            
            # Calculate trading returns
            returns = self._calculate_trading_returns(ensemble_proba, y, 0.5, cost_per_trade, slippage)
            
            # Calculate Sharpe ratio
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe = 0
            
            # Calculate CAGR
            cumulative_returns = np.cumprod(1 + returns)
            if len(cumulative_returns) > 1:
                cagr = (cumulative_returns[-1] / cumulative_returns[0]) ** (252 / len(cumulative_returns)) - 1
            else:
                cagr = 0
            
            # Combined objective (negative for minimization)
            combined_score = sharpe + cagr
            return -combined_score
        
        # Initial weights (equal)
        n_models = proba_matrix.shape[1]
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        result = minimize(
            objective, initial_weights, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)  # Ensure normalization
            logger.info(f"Combined optimization successful. Final score: {-result.fun:.3f}")
        else:
            logger.warning("Combined optimization failed. Using equal weights.")
            optimal_weights = np.ones(n_models) / n_models
        
        return optimal_weights
    
    def _optimize_threshold(self, ensemble_proba: np.ndarray, y: np.ndarray,
                           cost_per_trade: float = None, slippage: float = None) -> float:
        """
        Find optimal threshold for trading decisions with enhanced diagnostics
        
        Args:
            ensemble_proba: Ensemble probabilities
            y: Asset returns
            cost_per_trade: Transaction cost per trade
            slippage: Slippage cost
        
        Returns:
            Optimal threshold
        """
        logger.info("Optimizing trading threshold with enhanced diagnostics...")
        
        # Plot probability distribution around threshold
        self.trading_calculator.plot_probability_histograms(
            ensemble_proba, 
            threshold=0.5,
            save_path='results/qqq_ensemble/threshold_optimization_histograms.png'
        )
        
        # Extended grid search for threshold
        thresholds = np.arange(0.45, 0.76, 0.01)  # Wider range
        threshold_results = []
        
        for threshold in thresholds:
            returns = self._calculate_trading_returns(ensemble_proba, y, threshold, cost_per_trade, slippage)
            
            # Use robust Sharpe calculation
            sharpe_result = self.trading_calculator.calculate_robust_sharpe(returns)
            sharpe = sharpe_result['sharpe_ratio']
            
            threshold_results.append({
                'threshold': threshold,
                'sharpe': sharpe,
                'volatility': np.std(returns),
                'mean_return': np.mean(returns)
            })
        
        # Find best threshold
        valid_results = [r for r in threshold_results if not np.isinf(r['sharpe']) and not np.isnan(r['sharpe'])]
        
        if not valid_results:
            logger.error("⚠️  No valid thresholds found - all Sharpe ratios are infinite or NaN!")
            return 0.5  # Fallback
        
        best_result = max(valid_results, key=lambda x: x['sharpe'])
        optimal_threshold = best_result['threshold']
        best_sharpe = best_result['sharpe']
        
        # Check for suspicious threshold values
        if abs(optimal_threshold - 0.50) < 0.01:
            logger.warning("⚠️  Threshold is suspiciously close to 0.50!")
            logger.warning("   This may indicate poor calibration or class imbalance")
            
            # Look for alternative thresholds
            alternative_thresholds = [r for r in valid_results if abs(r['threshold'] - 0.50) > 0.05]
            if alternative_thresholds:
                alt_best = max(alternative_thresholds, key=lambda x: x['sharpe'])
                if alt_best['sharpe'] > best_sharpe * 0.95:  # Within 5% of best
                    logger.info(f"   Alternative threshold {alt_best['threshold']:.3f} (Sharpe: {alt_best['sharpe']:.3f})")
                    optimal_threshold = alt_best['threshold']
                    best_sharpe = alt_best['sharpe']
        
        # Log optimization results
        logger.info(f"Threshold optimization completed:")
        logger.info(f"  Optimal threshold: {optimal_threshold:.3f}")
        logger.info(f"  Best Sharpe: {best_sharpe:.3f}")
        logger.info(f"  Threshold range tested: {thresholds[0]:.2f} - {thresholds[-1]:.2f}")
        
        # Plot Sharpe vs threshold curve
        self._plot_sharpe_vs_threshold(threshold_results, optimal_threshold)
        
        return optimal_threshold
    
    def _plot_sharpe_vs_threshold(self, threshold_results: List[Dict], optimal_threshold: float):
        """Plot Sharpe ratio vs threshold curve"""
        try:
            thresholds = [r['threshold'] for r in threshold_results]
            sharpes = [r['sharpe'] for r in threshold_results]
            
            plt.figure(figsize=(12, 8))
            
            # Main plot
            plt.subplot(2, 1, 1)
            plt.plot(thresholds, sharpes, 'b-', linewidth=2, alpha=0.7)
            plt.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, 
                       label=f'Optimal: {optimal_threshold:.3f}')
            plt.axvline(x=0.50, color='orange', linestyle=':', alpha=0.7, 
                       label='Default: 0.50')
            plt.xlabel('Threshold')
            plt.ylabel('Annualized Sharpe Ratio')
            plt.title('Sharpe Ratio vs Threshold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Zoom around optimal threshold
            plt.subplot(2, 1, 2)
            optimal_idx = np.argmin(np.abs(np.array(thresholds) - optimal_threshold))
            start_idx = max(0, optimal_idx - 5)
            end_idx = min(len(thresholds), optimal_idx + 6)
            
            plt.plot(thresholds[start_idx:end_idx], sharpes[start_idx:end_idx], 'g-', linewidth=2)
            plt.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, 
                       label=f'Optimal: {optimal_threshold:.3f}')
            plt.xlabel('Threshold')
            plt.ylabel('Annualized Sharpe Ratio')
            plt.title('Sharpe Ratio vs Threshold (Zoomed)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('results/qqq_ensemble/sharpe_vs_threshold.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Sharpe vs threshold plot saved to: results/qqq_ensemble/sharpe_vs_threshold.png")
            
        except Exception as e:
            logger.warning(f"Could not create Sharpe vs threshold plot: {e}")
    
    def _calculate_trading_returns(self, ensemble_proba: np.ndarray, y: np.ndarray,
                                 threshold: float, cost_per_trade: float = None, slippage: float = None) -> np.ndarray:
        """
        Calculate trading returns using enhanced trading calculator
        
        Args:
            ensemble_proba: Ensemble probabilities
            y: Asset returns
            threshold: Trading threshold
            cost_per_trade: Transaction cost per trade (uses instance default if None)
            slippage: Slippage cost (uses instance default if None)
        
        Returns:
            Net returns after costs
        """
        # Use instance defaults if not specified
        if cost_per_trade is None:
            cost_per_trade = self.cost_per_trade
        if slippage is None:
            slippage = self.slippage_bps / 10000.0
        
        # Use enhanced trading calculator
        results = self.trading_calculator.calculate_enhanced_returns(
            probabilities=ensemble_proba,
            returns=y,
            threshold=threshold,
            apply_costs=True,
            apply_slippage=True,
            apply_holding_period=True
        )
        
        return results['net_returns']
    
    def _calculate_trading_metrics(self, ensemble_proba: np.ndarray, y: np.ndarray,
                                 threshold: float, cost_per_trade: float = None, slippage: float = None) -> Dict[str, float]:
        """Calculate comprehensive trading metrics with enhanced diagnostics"""
        # Calculate returns with and without costs for comparison
        returns_with_costs = self._calculate_trading_returns(ensemble_proba, y, threshold, cost_per_trade, slippage)
        
        # Calculate returns without costs for comparison
        results_no_costs = self.trading_calculator.calculate_enhanced_returns(
            probabilities=ensemble_proba,
            returns=y,
            threshold=threshold,
            apply_costs=False,
            apply_slippage=False,
            apply_holding_period=False
        )
        returns_without_costs = results_no_costs['strategy_returns']
        
        # Get detailed results for cost analysis
        results_with_costs = self.trading_calculator.calculate_enhanced_returns(
            probabilities=ensemble_proba,
            returns=y,
            threshold=threshold,
            apply_costs=True,
            apply_slippage=True,
            apply_holding_period=True
        )
        
        # Calculate robust Sharpe ratios
        sharpe_with_costs = self.trading_calculator.calculate_robust_sharpe(returns_with_costs)
        sharpe_without_costs = self.trading_calculator.calculate_robust_sharpe(returns_without_costs)
        
        # Basic metrics
        total_return = np.prod(1 + returns_with_costs) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns_with_costs)) - 1
        
        # Risk metrics
        volatility = np.std(returns_with_costs) * np.sqrt(252)
        sharpe_ratio = sharpe_with_costs['sharpe_ratio']
        
        # Drawdown
        cumulative_returns = np.cumprod(1 + returns_with_costs)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Trading metrics
        trades = np.sum(np.abs(np.diff(returns_with_costs > 0, prepend=False)))
        turnover = np.sum(results_with_costs['turnover'])
        
        # Cost analysis
        cost_analysis = calculate_cost_impact_analysis(
            returns_without_costs, returns_with_costs,
            results_with_costs['costs'], results_with_costs['slippage_costs']
        )
        
        # Enhanced metrics
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trade_count': trades,
            'turnover': turnover,
            'costs_applied': True,
            'cost_analysis': cost_analysis,
            'sharpe_without_costs': sharpe_without_costs['sharpe_ratio'],
            'sharpe_degradation': cost_analysis['sharpe_degradation'],
            'annualized_costs': cost_analysis['annualized_costs'],
            'cost_drag_pct': cost_analysis['cost_drag_pct']
        }
        
        # Log cost impact
        logger.info(f"Cost Impact Summary:")
        logger.info(f"  Sharpe without costs: {sharpe_without_costs['sharpe_ratio']:.3f}")
        logger.info(f"  Sharpe with costs: {sharpe_ratio:.3f}")
        logger.info(f"  Sharpe degradation: {cost_analysis['sharpe_degradation']:.3f}")
        logger.info(f"  Cost drag: {cost_analysis['cost_drag_pct']:.2f}%")
        
        return metrics
    
    def apply_diversity_control(self, correlation_threshold: float = 0.95) -> Dict[str, float]:
        """Apply diversity control by down-weighting highly correlated models"""
        if not self.optimal_weights:
            logger.warning("No optimal weights available. Run optimize_ensemble_weights() first.")
            return {}
        
        logger.info(f"Applying diversity control with correlation threshold: {correlation_threshold}")
        
        # Calculate pairwise correlations
        model_names = list(self.optimal_weights.keys())
        n_models = len(model_names)
        correlations = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                corr, _ = spearmanr(
                    self.oof_probabilities[model_names[i]], 
                    self.oof_probabilities[model_names[j]]
                )
                correlations[i, j] = corr
                correlations[j, i] = corr
        
        # Identify highly correlated model pairs
        high_corr_pairs = []
        for i in range(n_models):
            for j in range(i+1, n_models):
                if abs(correlations[i, j]) > correlation_threshold:
                    high_corr_pairs.append((i, j))
        
        # Apply diversity penalty
        adjusted_weights = self.optimal_weights.copy()
        
        if high_corr_pairs:
            logger.info(f"Found {len(high_corr_pairs)} highly correlated model pairs")
            
            for i, j in high_corr_pairs:
                model_i, model_j = model_names[i], model_names[j]
                corr = correlations[i, j]
                
                # Calculate diversity factor
                diversity_factor = 1 - abs(corr)
                
                # Apply penalty to both models
                adjusted_weights[model_i] *= diversity_factor
                adjusted_weights[model_j] *= diversity_factor
                
                logger.info(f"Applied diversity penalty to {model_i} and {model_j} (corr: {corr:.3f})")
        
        # Renormalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        self.diversity_adjusted_weights = adjusted_weights
        logger.info("Diversity control applied successfully")
        
        return adjusted_weights
    
    def get_model_diversity_metrics(self) -> Dict[str, Any]:
        """Calculate model diversity metrics"""
        if not self.oof_probabilities:
            return {}
        
        model_names = list(self.oof_probabilities.keys())
        n_models = len(model_names)
        
        # Calculate pairwise correlations
        correlations = {}
        for i in range(n_models):
            for j in range(i+1, n_models):
                corr, _ = spearmanr(
                    self.oof_probabilities[model_names[i]], 
                    self.oof_probabilities[model_names[j]]
                )
                pair_name = f"{model_names[i]}_{model_names[j]}"
                correlations[pair_name] = corr
        
        # Calculate diversity score (lower is more diverse)
        avg_correlation = np.mean(list(correlations.values()))
        diversity_score = 1 - avg_correlation
        
        return {
            'pairwise_correlations': correlations,
            'average_correlation': avg_correlation,
            'diversity_score': diversity_score,
            'high_correlation_pairs': [
                pair for pair, corr in correlations.items() 
                if abs(corr) > 0.95
            ]
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions using the ensemble with optimal weights"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        if not self.optimal_weights:
            logger.warning("No optimal weights available. Using equal weights.")
            weights = np.ones(len(self.models)) / len(self.models)
            model_names = list(self.models.keys())
        else:
            weights = list(self.optimal_weights.values())
            model_names = list(self.optimal_weights.keys())
        
        # Get probabilities from each model
        proba_predictions = []
        for name in model_names:
            if name in self.models:
                if name in self.calibrators:
                    # Use calibrated model
                    proba = self.calibrators[name].predict_proba(X)
                else:
                    # Use base model
                    proba = self.models[name].predict_proba(X)
                
                if proba.shape[1] == 2:
                    proba = proba[:, 1]  # Probability of positive class
                else:
                    proba = proba[:, 0]  # Single probability
                
                proba_predictions.append(proba)
        
        # Weighted ensemble
        ensemble_proba = np.average(proba_predictions, axis=0, weights=weights)
        
        return ensemble_proba
    
    def evaluate_individual_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate individual model performance"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before evaluation")
        
        logger.info("Evaluating individual model performance...")
        
        results = {}
        
        for name, pipeline in self.models.items():
            try:
                # Get predictions
                if name in self.calibrators:
                    proba = self.calibrators[name].predict_proba(X)[:, 1]
                else:
                    proba = pipeline.predict_proba(X)[:, 1]
                
                # Calculate metrics
                auc = roc_auc_score(y, proba)
                brier = brier_score_loss(y, proba)
                
                # Calculate calibration error
                calibration_error = self._calculate_calibration_error(proba, y)
                
                # Calculate OOF Sharpe (if available)
                oof_sharpe = 0
                if name in self.oof_probabilities:
                    oof_returns = self._calculate_trading_returns(
                        self.oof_probabilities[name], y, 0.5, 0.001, 0.0005
                    )
                    if np.std(oof_returns) > 0:
                        oof_sharpe = np.mean(oof_returns) / np.std(oof_returns) * np.sqrt(252)
                
                results[name] = {
                    'auc': auc,
                    'brier_score': brier,
                    'calibration_mae': calibration_error,
                    'oof_sharpe': oof_sharpe
                }
                
            except Exception as e:
                logger.warning(f"Error evaluating {name}: {e}")
                results[name] = {
                    'auc': 0,
                    'brier_score': 1,
                    'calibration_mae': 1,
                    'oof_sharpe': 0
                }
        
        return results
    
    def _calculate_calibration_error(self, proba: np.ndarray, y: np.ndarray) -> float:
        """Calculate calibration error using MAE"""
        # Bin probabilities and calculate calibration error
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(proba, bins) - 1
        
        calibration_error = 0
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_proba = np.mean(proba[mask])
                bin_actual = np.mean(y[mask])
                calibration_error += np.abs(bin_proba - bin_actual)
        
        return calibration_error / (len(bins) - 1)
    
    def save_ensemble(self, filepath: str) -> None:
        """Save the trained ensemble"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before saving")
        
        ensemble_data = {
            'models': self.models,
            'calibrators': self.calibrators,
            'optimal_weights': self.optimal_weights,
            'optimal_threshold': self.optimal_threshold,
            'oof_probabilities': self.oof_probabilities,
            'is_fitted': self.is_fitted,
            'random_state': self.random_state,
            'n_splits': self.n_splits
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble saved to {filepath}")
    
    def save_artifacts(self, base_path: str, y: np.ndarray = None) -> Dict[str, str]:
        """
        Save all ensemble artifacts for reproducibility
        
        Args:
            base_path: Base directory for saving artifacts
            y: Target values (if provided, will be saved)
        
        Returns:
            Dictionary mapping artifact names to file paths
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before saving artifacts")
        
        import json
        from pathlib import Path
        
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        artifacts = {}
        
        # Save OOF probabilities matrix P (T×M)
        if self.oof_probabilities:
            oof_matrix = np.column_stack(list(self.oof_probabilities.values()))
            model_names = list(self.oof_probabilities.keys())
            
            oof_file = base_path / "oof_probabilities.npy"
            np.save(oof_file, oof_matrix)
            artifacts['oof_probabilities'] = str(oof_file)
            
            # Save model names for reference
            model_names_file = base_path / "model_names.json"
            with open(model_names_file, 'w') as f:
                json.dump(model_names, f)
            artifacts['model_names'] = str(model_names_file)
        
        # Save target values y
        if y is not None:
            y_file = base_path / "target_values.npy"
            np.save(y_file, y)
            artifacts['target_values'] = str(y_file)
        
        # Save optimal weights
        if self.optimal_weights:
            weights_file = base_path / "optimal_weights.json"
            with open(weights_file, 'w') as f:
                json.dump(self.optimal_weights, f, indent=2)
            artifacts['optimal_weights'] = str(weights_file)
        
        # Save optimal threshold
        threshold_file = base_path / "optimal_threshold.json"
        with open(threshold_file, 'w') as f:
            json.dump({'optimal_threshold': self.optimal_threshold}, f, indent=2)
        artifacts['optimal_threshold'] = str(threshold_file)
        
        # Save regime weights if available
        if hasattr(self, 'regime_weights') and self.regime_weights:
            regime_weights_file = base_path / "regime_weights.json"
            with open(regime_weights_file, 'w') as f:
                json.dump(self.regime_weights, f, indent=2)
            artifacts['regime_weights'] = str(regime_weights_file)
        
        # Save comprehensive metrics
        if hasattr(self, 'X_train') and hasattr(self, 'y_train'):
            # Calculate and save metrics
            metrics = self._calculate_comprehensive_metrics()
            
            # Convert numpy types to Python types for JSON serialization
            metrics = self._convert_numpy_types(metrics)
            
            metrics_file = base_path / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            artifacts['metrics'] = str(metrics_file)
        
        # Save version and reproducibility info
        version_info = {
            'ensemble_version': '1.0.0',
            'random_state': self.random_state,
            'n_splits': self.n_splits,
            'n_models': len(self.models),
            'active_models': list(self.models.keys()),
            'timestamp': pd.Timestamp.now().isoformat(),
            'python_version': f"{pd.__version__}",
            'numpy_version': f"{np.__version__}",
            'sklearn_version': f"{sklearn.__version__}"
        }
        
        version_file = base_path / "version_info.json"
        with open(version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        artifacts['version_info'] = str(version_file)
        
        logger.info(f"All artifacts saved to {base_path}")
        return artifacts
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive metrics for the ensemble"""
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            return {}
        
        metrics = {}
        
        # Individual model metrics
        individual_metrics = self.evaluate_individual_models(self.X_train, self.y_train)
        metrics['individual_models'] = individual_metrics
        
        # Ensemble metrics
        if self.optimal_weights and self.optimal_threshold:
            ensemble_proba = self.predict_proba(self.X_train)
            ensemble_metrics = self._calculate_trading_metrics(
                ensemble_proba, self.y_train, self.optimal_threshold, 0.001, 0.0005
            )
            metrics['ensemble'] = ensemble_metrics
        
        # Diversity metrics
        diversity_metrics = self.get_model_diversity_metrics()
        metrics['diversity'] = diversity_metrics
        
        # OOF performance metrics
        if self.oof_probabilities:
            oof_metrics = {}
            for name, proba in self.oof_probabilities.items():
                try:
                    oof_returns = self._calculate_trading_returns(
                        proba, self.y_train, 0.5, 0.001, 0.0005
                    )
                    if np.std(oof_returns) > 0:
                        oof_sharpe = np.mean(oof_returns) / np.std(oof_returns) * np.sqrt(252)
                        oof_cagr = (np.prod(1 + oof_returns) ** (252 / len(oof_returns))) - 1
                    else:
                        oof_sharpe = 0
                        oof_cagr = 0
                    
                    oof_metrics[name] = {
                        'sharpe': oof_sharpe,
                        'cagr': oof_cagr,
                        'auc': roc_auc_score(self.y_train, proba),
                        'brier': brier_score_loss(self.y_train, proba)
                    }
                except Exception as e:
                    logger.warning(f"Could not calculate OOF metrics for {name}: {e}")
                    oof_metrics[name] = {'error': str(e)}
            
            metrics['oof_performance'] = oof_metrics
        
        return metrics
    
    def load_ensemble(self, filepath: str) -> None:
        """Load a trained ensemble"""
        ensemble_data = joblib.load(filepath)
        
        self.models = ensemble_data['models']
        self.calibrators = ensemble_data.get('calibrators', {})
        self.optimal_weights = ensemble_data.get('optimal_weights', {})
        self.optimal_threshold = ensemble_data.get('optimal_threshold', 0.5)
        self.oof_probabilities = ensemble_data.get('oof_probabilities', {})
        self.is_fitted = ensemble_data['is_fitted']
        self.random_state = ensemble_data.get('random_state', 42)
        self.n_splits = ensemble_data.get('n_splits', 5)
        
        logger.info(f"Ensemble loaded from {filepath}")
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get comprehensive ensemble summary"""
        summary = {
            'n_models': len(self.models),
            'active_models': list(self.models.keys()),
            'is_fitted': self.is_fitted,
            'optimal_weights': self.optimal_weights,
            'optimal_threshold': self.optimal_threshold,
            'n_splits': self.n_splits,
            'random_state': self.random_state
        }
        
        if self.oof_probabilities:
            summary['oof_probabilities_shape'] = {
                name: proba.shape for name, proba in self.oof_probabilities.items()
            }
        
        return summary
    
    def fit_regime_aware_weights(self, X: np.ndarray, y: np.ndarray, 
                                regime_features: np.ndarray = None,
                                regime_method: str = 'vix_adx') -> Dict[str, Dict[str, float]]:
        """
        Fit regime-aware weights for different market conditions
        
        Args:
            X: Feature matrix
            y: Target values
            regime_features: Optional regime features (e.g., VIX, ADX)
            regime_method: Method for regime detection ('vix_adx', 'volatility', 'trend')
        
        Returns:
            Dictionary mapping regimes to optimal weights
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before regime-aware weight fitting")
        
        logger.info(f"Fitting regime-aware weights using {regime_method} method...")
        
        # Detect regimes if not provided
        if regime_features is None:
            regime_features = self._detect_market_regimes(X, y, method=regime_method)
        
        # Get unique regimes
        unique_regimes = np.unique(regime_features)
        regime_weights = {}
        
        for regime in unique_regimes:
            logger.info(f"Fitting weights for regime: {regime}")
            
            # Get data for this regime
            regime_mask = regime_features == regime
            regime_y = y[regime_mask]
            
            if len(regime_y) < 50:  # Need sufficient data
                logger.warning(f"Insufficient data for regime {regime}. Skipping.")
                continue
            
            # Get OOF probabilities for this regime
            regime_oof = {}
            for name, proba in self.oof_probabilities.items():
                regime_oof[name] = proba[regime_mask]
            
            # Convert to matrix
            proba_matrix = np.column_stack(list(regime_oof.values()))
            
            # Optimize weights for this regime
            try:
                optimal_weights = self._optimize_weights_sharpe(
                    proba_matrix, regime_y, 0.001, 0.0005
                )
                
                regime_weights[str(regime)] = dict(zip(regime_oof.keys(), optimal_weights))
                logger.info(f"Regime {regime} weights fitted successfully")
                
            except Exception as e:
                logger.warning(f"Failed to fit weights for regime {regime}: {e}")
                # Use equal weights as fallback
                n_models = len(regime_oof)
                equal_weights = np.ones(n_models) / n_models
                regime_weights[str(regime)] = dict(zip(regime_oof.keys(), equal_weights))
        
        self.regime_weights = regime_weights
        logger.info(f"Regime-aware weights fitted for {len(regime_weights)} regimes")
        
        return regime_weights
    
    def _detect_market_regimes(self, X: np.ndarray, y: np.ndarray, 
                              method: str = 'vix_adx') -> np.ndarray:
        """Detect market regimes based on various methods"""
        logger.info(f"Detecting market regimes using {method} method...")
        
        if method == 'vix_adx':
            # Simple regime detection based on volatility and trend
            # In practice, you'd use actual VIX and ADX data
            # For now, we'll simulate with synthetic features
            
            # Calculate rolling volatility (simulating VIX)
            returns = np.diff(y, prepend=y[0])
            rolling_vol = pd.Series(returns).rolling(window=20).std().fillna(0.02).values
            
            # Calculate rolling trend (simulating ADX)
            rolling_trend = pd.Series(y).rolling(window=20).mean().fillna(0.5).values
            
            # Define regimes
            regimes = np.zeros(len(y))
            
            # High volatility, low trend = crisis/choppy
            high_vol_mask = rolling_vol > np.percentile(rolling_vol, 75)
            low_trend_mask = np.abs(rolling_trend - 0.5) < 0.1
            regimes[high_vol_mask & low_trend_mask] = 0  # Crisis/Choppy
            
            # Low volatility, high trend = trending
            low_vol_mask = rolling_vol < np.percentile(rolling_vol, 25)
            high_trend_mask = np.abs(rolling_trend - 0.5) > 0.2
            regimes[low_vol_mask & high_trend_mask] = 1  # Trending
            
            # Default regime
            regimes[regimes == 0] = 2  # Normal
            
        elif method == 'volatility':
            # Simple volatility-based regimes
            returns = np.diff(y, prepend=y[0])
            rolling_vol = pd.Series(returns).rolling(window=20).std().fillna(0.02).values
            
            regimes = np.zeros(len(y))
            regimes[rolling_vol > np.percentile(rolling_vol, 75)] = 0  # High vol
            regimes[rolling_vol < np.percentile(rolling_vol, 25)] = 1  # Low vol
            regimes[regimes == 0] = 2  # Medium vol
            
        elif method == 'trend':
            # Simple trend-based regimes
            rolling_trend = pd.Series(y).rolling(window=20).mean().fillna(0.5).values
            
            regimes = np.zeros(len(y))
            regimes[rolling_trend > 0.6] = 0  # Bullish
            regimes[rolling_trend < 0.4] = 1  # Bearish
            regimes[regimes == 0] = 2  # Sideways
            
        else:
            logger.warning(f"Unknown regime method: {method}. Using default regime.")
            regimes = np.zeros(len(y))
        
        logger.info(f"Detected {len(np.unique(regimes))} market regimes")
        return regimes
    
    def predict_proba_regime_aware(self, X: np.ndarray, 
                                  regime_features: np.ndarray = None,
                                  regime_method: str = 'vix_adx') -> np.ndarray:
        """Make regime-aware probability predictions"""
        if not hasattr(self, 'regime_weights') or not self.regime_weights:
            logger.warning("No regime weights available. Using global weights.")
            return self.predict_proba(X)
        
        # Detect regimes if not provided
        if regime_features is None:
            regime_features = self._detect_market_regimes(X, np.zeros(len(X)), method=regime_method)
        
        # Make predictions for each regime
        predictions = np.zeros(len(X))
        
        for regime in np.unique(regime_features):
            regime_mask = regime_features == regime
            regime_str = str(regime)
            
            if regime_str in self.regime_weights:
                # Use regime-specific weights
                weights = list(self.regime_weights[regime_str].values())
                model_names = list(self.regime_weights[regime_str].keys())
                
                # Get probabilities from each model for this regime
                proba_predictions = []
                for name in model_names:
                    if name in self.models:
                        if name in self.calibrators:
                            proba = self.calibrators[name].predict_proba(X[regime_mask])
                        else:
                            proba = self.models[name].predict_proba(X[regime_mask])
                        
                        if proba.shape[1] == 2:
                            proba = proba[:, 1]
                        else:
                            proba = proba[:, 0]
                        
                        proba_predictions.append(proba)
                
                if proba_predictions:
                    # Weighted ensemble for this regime
                    ensemble_proba = np.average(proba_predictions, axis=0, weights=weights)
                    predictions[regime_mask] = ensemble_proba
            else:
                # Use global weights as fallback
                logger.warning(f"No weights for regime {regime}. Using global weights.")
                global_predictions = self.predict_proba(X[regime_mask])
                predictions[regime_mask] = global_predictions
        
        return predictions

