"""
Advanced Model Ensemble for the Walk-Forward ML Trading System
Includes multiple base models, ensemble methods, and grid search optimization
Optimized for classification with probability outputs and trading performance
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, StackingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
    VotingRegressor, StackingRegressor
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, ElasticNet, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from typing import Dict, List, Tuple, Any, Optional
import joblib
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

from config import (
    MODEL_CONFIG, GRIDSEARCH_CONFIG, SYSTEM_CONFIG, 
    PERFORMANCE_TARGETS, PATHS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEnsemble:
    """
    Advanced ensemble model system with grid search optimization
    Optimized for M1 chip with multi-threading
    Now supports classification with probability outputs for trading
    """
    
    def __init__(self, n_jobs: int = -1, task_type: str = 'classification'):
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.task_type = task_type
        self.base_models = {}
        self.ensemble_model = None
        self.best_params = {}
        self.feature_importance = {}
        self.scalers = {}
        self.calibrators = {}
        self.is_fitted = False
        
        logger.info(f"ModelEnsemble initialized with {self.n_jobs} jobs for {task_type}")
        
        # Initialize base models
        self._initialize_base_models()
    
    def _initialize_base_models(self):
        """Initialize all base models with default parameters"""
        
        if self.task_type == 'classification':
            # Tree-based models
            self.base_models['RandomForest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,  # Will be handled by ensemble
                class_weight='balanced'
            )
            
            self.base_models['GradientBoosting'] = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            # XGBoost
            try:
                self.base_models['XGBoost'] = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=1,
                    eval_metric='logloss',
                    use_label_encoder=False
                )
            except ImportError:
                logger.warning("XGBoost not available")
            
            # LightGBM
            try:
                self.base_models['LightGBM'] = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=1,
                    verbose=-1,
                    # Performance optimizations
                    force_col_wise=True,  # Faster for wide datasets
                    force_row_wise=False,  # Disable row-wise for better performance
                    boost_from_average=True,  # Faster training
                    reg_alpha=0.0,  # L1 regularization
                    reg_lambda=0.0,  # L2 regularization
                    min_child_samples=20,  # Reduce overfitting
                    min_child_weight=1e-3,  # Reduce overfitting
                    class_weight='balanced'
                )
            except ImportError:
                logger.warning("LightGBM not available")
            
            # CatBoost
            try:
                self.base_models['CatBoost'] = cb.CatBoostClassifier(
                    iterations=100,
                    learning_rate=0.1,
                    depth=6,
                    random_seed=42,
                    verbose=False,
                    class_weights=[1, 1]  # Balanced classes
                )
            except ImportError:
                logger.warning("CatBoost not available")
            
            # Linear models
            self.base_models['LogisticRegression'] = LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                l1_ratio=0.5,
                C=1.0,
                random_state=42,
                max_iter=2000,
                class_weight='balanced'
            )
            
            self.base_models['RidgeClassifier'] = RidgeClassifier(
                alpha=1.0,
                random_state=42,
                class_weight='balanced'
            )
            
            # SVM
            self.base_models['SVC'] = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,  # Enable probability estimates
                random_state=42,
                class_weight='balanced'
            )
            
            # Neural Network
            self.base_models['MLP'] = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate='adaptive',
                max_iter=GRIDSEARCH_CONFIG.get('neural_network_max_iter', 1000),
                early_stopping=GRIDSEARCH_CONFIG.get('neural_network_early_stopping', True),
                validation_fraction=0.1,  # Use 10% for validation
                n_iter_no_change=10,  # Stop if no improvement for 10 iterations
                verbose=GRIDSEARCH_CONFIG.get('neural_network_verbose', 0),  # Control verbosity
                random_state=42
            )
        else:
            # Keep regression models for backward compatibility
            # Tree-based models
            self.base_models['RandomForest'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1  # Will be handled by ensemble
            )
            
            self.base_models['GradientBoosting'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            # XGBoost
            try:
                self.base_models['XGBoost'] = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=1
                )
            except ImportError:
                logger.warning("XGBoost not available")
            
            # LightGBM
            try:
                self.base_models['LightGBM'] = lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=1,
                    verbose=-1,
                    # Performance optimizations
                    force_col_wise=True,  # Faster for wide datasets
                    force_row_wise=False,  # Disable row-wise for better performance
                    boost_from_average=True,  # Faster training
                    reg_alpha=0.0,  # L1 regularization
                    reg_lambda=0.0,  # L2 regularization
                    min_child_samples=20,  # Reduce overfitting
                    min_child_weight=1e-3  # Reduce overfitting
                )
            except ImportError:
                logger.warning("LightGBM not available")
            
            # CatBoost
            try:
                self.base_models['CatBoost'] = cb.CatBoostRegressor(
                    iterations=100,
                    learning_rate=0.1,
                    depth=6,
                    random_seed=42,
                    verbose=False
                )
            except ImportError:
                logger.warning("CatBoost not available")
            
            # Linear models
            self.base_models['ElasticNet'] = ElasticNet(
                alpha=0.01,
                l1_ratio=0.5,
                random_state=42,
                max_iter=2000
            )
            
            self.base_models['Ridge'] = Ridge(
                alpha=1.0,
                random_state=42,
                max_iter=2000
            )
            
            # SVM
            self.base_models['SVM'] = SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                max_iter=2000
            )
            
            # Neural Network
            self.base_models['NeuralNetwork'] = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate='adaptive',
                max_iter=GRIDSEARCH_CONFIG.get('neural_network_max_iter', 1000),
                early_stopping=GRIDSEARCH_CONFIG.get('neural_network_early_stopping', True),
                validation_fraction=0.1,  # Use 10% for validation
                n_iter_no_change=10,  # Stop if no improvement for 10 iterations
                verbose=GRIDSEARCH_CONFIG.get('neural_network_verbose', 0),  # Control verbosity
                random_state=42
            )
        
        logger.info(f"Initialized {len(self.base_models)} base models for {self.task_type}")
    
    def get_grid_search_params(self) -> Dict[str, Dict[str, List]]:
        """Get comprehensive grid search parameters for all models"""
        
        grid_params = {}
        
        if self.task_type == 'classification':
            # Random Forest
            grid_params['RandomForest'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Gradient Boosting
            grid_params['GradientBoosting'] = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [4, 6, 8],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            # XGBoost
            if 'XGBoost' in self.base_models:
                grid_params['XGBoost'] = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [4, 6, 8],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            
            # LightGBM
            if 'LightGBM' in self.base_models:
                grid_params['LightGBM'] = {
                    'n_estimators': [50, 100],  # Reduced from 3 to 2 values
                    'learning_rate': [0.1, 0.2],  # Reduced from 3 to 2 values
                    'max_depth': [4, 6],  # Reduced from 3 to 2 values
                    'subsample': [0.8, 1.0],  # Reduced from 3 to 2 values
                    'colsample_bytree': [0.8, 1.0]  # Reduced from 3 to 2 values
                }
            
            # CatBoost
            if 'CatBoost' in self.base_models:
                grid_params['CatBoost'] = {
                    'iterations': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'depth': [4, 6, 8]
                }
            
            # LogisticRegression
            grid_params['LogisticRegression'] = {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
            
            # RidgeClassifier
            grid_params['RidgeClassifier'] = {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
            
            # SVC
            grid_params['SVC'] = {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
            
            # MLP
            grid_params['MLP'] = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],  # Reduced from 4 to 3 combinations
                'alpha': [0.0001, 0.01],  # Reduced from 3 to 2 combinations
                'learning_rate': ['adaptive']  # Reduced from 2 to 1 combination (adaptive is usually better)
            }
        else:
            # Keep regression grid params for backward compatibility
            # Random Forest
            grid_params['RandomForest'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Gradient Boosting
            grid_params['GradientBoosting'] = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [4, 6, 8],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            # XGBoost
            if 'XGBoost' in self.base_models:
                grid_params['XGBoost'] = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [4, 6, 8],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            
            # LightGBM
            if 'LightGBM' in self.base_models:
                grid_params['LightGBM'] = {
                    'n_estimators': [50, 100],  # Reduced from 3 to 2 values
                    'learning_rate': [0.1, 0.2],  # Reduced from 3 to 2 values
                    'max_depth': [4, 6],  # Reduced from 3 to 2 values
                    'subsample': [0.8, 1.0],  # Reduced from 3 to 2 values
                    'colsample_bytree': [0.8, 1.0]  # Reduced from 3 to 2 values
                }
            
            # CatBoost
            if 'CatBoost' in self.base_models:
                grid_params['CatBoost'] = {
                    'iterations': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'depth': [4, 6, 8]
                }
            
            # ElasticNet
            grid_params['ElasticNet'] = {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
            
            # Ridge
            grid_params['Ridge'] = {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
            
            # SVM
            grid_params['SVM'] = {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
            
            # Neural Network
            grid_params['NeuralNetwork'] = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],  # Reduced from 4 to 3 combinations
                'alpha': [0.0001, 0.01],  # Reduced from 3 to 2 combinations
                'learning_rate': ['adaptive']  # Reduced from 2 to 1 combination (adaptive is usually better)
            }
        
        return grid_params
    
    def optimize_individual_models(self, X: np.ndarray, y: np.ndarray,
                                 cv_folds: int = 5) -> Dict[str, Any]:
        """
        Optimize individual models using grid search with time series cross-validation
        """
        logger.info("Starting individual model optimization...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        grid_params = self.get_grid_search_params()
        optimized_models = {}
        best_scores = {}
        
        # Move function outside to avoid pickling issues
        from functools import partial
        
        def create_neural_network_model(base_model, config):
            """Safely create a neural network model without parameter conflicts"""
            base_params = base_model.get_params()
            
            # Remove all parameters that we want to override
            override_params = [
                'max_iter', 'early_stopping', 'validation_fraction', 
                'n_iter_no_change', 'verbose'
            ]
            for param in override_params:
                base_params.pop(param, None)
            
            # Create model with clean parameters
            return MLPRegressor(
                **base_params,
                max_iter=config.get('neural_network_max_iter', 1000),
                early_stopping=config.get('neural_network_early_stopping', True),
                validation_fraction=0.1,
                n_iter_no_change=10,
                verbose=config.get('neural_network_verbose', 0)
            )
        
        def optimize_model_wrapper(model_name, base_models, grid_params, X, y, tscv):
            """Optimize a single model"""
            try:
                logger.info(f"Optimizing {model_name}...")
                
                model = base_models[model_name]
                params = grid_params[model_name]
                
                # Create scaler for this model
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Special optimization for LightGBM
                if model_name == 'LightGBM':
                    # Use configuration-based optimization
                    if GRIDSEARCH_CONFIG.get('fast_lightgbm_optimization', False):
                        # Use fast random search instead of grid search
                        from sklearn.model_selection import RandomizedSearchCV
                        from scipy.stats import uniform, randint
                        
                        param_distributions = {
                            'n_estimators': randint(50, 150),
                            'learning_rate': uniform(0.05, 0.15),
                            'max_depth': randint(3, 7),
                            'subsample': uniform(0.7, 0.3),
                            'colsample_bytree': uniform(0.7, 0.3)
                        }
                        
                        # Add early stopping if enabled
                        if GRIDSEARCH_CONFIG.get('lightgbm_early_stopping', True):
                            lightgbm_model = lgb.LGBMRegressor(
                                **model.get_params(),
                                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
                            )
                        else:
                            lightgbm_model = model
                        
                        grid_search = RandomizedSearchCV(
                            estimator=lightgbm_model,
                            param_distributions=param_distributions,
                            n_iter=GRIDSEARCH_CONFIG.get('lightgbm_random_search_iterations', 20),
                            cv=TimeSeriesSplit(n_splits=GRIDSEARCH_CONFIG.get('lightgbm_cv_folds', 3)),
                            scoring='neg_mean_squared_error',
                            n_jobs=1,
                            verbose=0,
                            random_state=42
                        )
                    else:
                        # Use reduced grid search
                        lightgbm_tscv = TimeSeriesSplit(n_splits=GRIDSEARCH_CONFIG.get('lightgbm_cv_folds', 3))
                        
                        # Add early stopping if enabled
                        if GRIDSEARCH_CONFIG.get('lightgbm_early_stopping', True):
                            lightgbm_model = lgb.LGBMRegressor(
                                **model.get_params(),
                                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
                            )
                        else:
                            lightgbm_model = model
                        
                        grid_search = GridSearchCV(
                            estimator=lightgbm_model,
                            param_grid=params,
                            cv=lightgbm_tscv,
                            scoring='neg_mean_squared_error',
                            n_jobs=1,
                            verbose=0
                        )
                # Special optimization for Neural Network
                elif model_name == 'NeuralNetwork':
                    # Skip neural network optimization if configured
                    if GRIDSEARCH_CONFIG.get('skip_neural_network_optimization', False):
                        logger.info("Skipping neural network optimization as configured")
                        return None
                    
                    # Use configuration-based optimization
                    nn_tscv = TimeSeriesSplit(n_splits=GRIDSEARCH_CONFIG.get('neural_network_cv_folds', 3))
                    
                    # Use helper function to safely create neural network model
                    nn_model = create_neural_network_model(model, GRIDSEARCH_CONFIG)
                    
                    grid_search = GridSearchCV(
                        estimator=nn_model,
                        param_grid=params,
                        cv=nn_tscv,
                        scoring='neg_mean_squared_error',
                        n_jobs=1,
                        verbose=0
                    )
                else:
                    # Standard grid search for other models
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=params,
                        cv=tscv,
                        scoring='neg_mean_squared_error',
                        n_jobs=1,  # Single job per model
                        verbose=0
                    )
                
                grid_search.fit(X_scaled, y)
                
                # Store results
                best_model = grid_search.best_estimator_
                best_score = -grid_search.best_score_  # Convert back to positive
                
                # Fit best model on full data
                best_model.fit(X_scaled, y)
                
                return {
                    'model': best_model,
                    'scaler': scaler,
                    'best_params': grid_search.best_params_,
                    'best_score': best_score,
                    'cv_results': grid_search.cv_results_
                }
                
            except Exception as e:
                logger.error(f"Error optimizing {model_name}: {e}")
                logger.error(f"Model parameters: {model.get_params() if hasattr(model, 'get_params') else 'N/A'}")
                logger.error(f"Grid parameters: {params}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return None
        
        # Optimize models in parallel (using ThreadPoolExecutor to avoid pickling issues)
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(executor.map(
                lambda name: optimize_model_wrapper(name, self.base_models, grid_params, X, y, tscv),
                list(grid_params.keys())
            ))
        
        # Process results
        for i, (model_name, result) in enumerate(zip(grid_params.keys(), results)):
            if result is not None:
                optimized_models[model_name] = result['model']
                self.scalers[model_name] = result['scaler']
                self.best_params[model_name] = result['best_params']
                best_scores[model_name] = result['best_score']
                
                logger.info(f"{model_name}: Best score = {result['best_score']:.6f}")
        
        logger.info(f"Optimized {len(optimized_models)} models")
        
        # Update base models with optimized versions
        self.base_models.update(optimized_models)
        
        return {
            'optimized_models': optimized_models,
            'best_scores': best_scores,
            'best_params': self.best_params
        }
    
    def optimize_lightgbm_fast(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Fast LightGBM optimization using random search instead of grid search
        """
        logger.info("Starting fast LightGBM optimization...")
        
        if 'LightGBM' not in self.base_models:
            logger.warning("LightGBM not available")
            return {}
        
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import uniform, randint
        
        # Create scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use random search with fewer iterations for speed
        param_distributions = {
            'n_estimators': randint(50, 150),
            'learning_rate': uniform(0.05, 0.15),
            'max_depth': randint(3, 7),
            'subsample': uniform(0.7, 0.3),
            'colsample_bytree': uniform(0.7, 0.3),
            'reg_alpha': uniform(0, 0.1),
            'reg_lambda': uniform(0, 0.1)
        }
        
        # Use early stopping
        lightgbm_model = lgb.LGBMRegressor(
            **self.base_models['LightGBM'].get_params(),
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        # Random search with fewer iterations
        random_search = RandomizedSearchCV(
            estimator=lightgbm_model,
            param_distributions=param_distributions,
            n_iter=20,  # Only 20 iterations instead of full grid search
            cv=TimeSeriesSplit(n_splits=3),
            scoring='neg_mean_squared_error',
            n_jobs=1,
            verbose=0,
            random_state=42
        )
        
        random_search.fit(X_scaled, y)
        
        # Store results
        best_model = random_search.best_estimator_
        best_score = -random_search.best_score_
        
        # Fit best model on full data
        best_model.fit(X_scaled, y)
        
        self.base_models['LightGBM'] = best_model
        self.scalers['LightGBM'] = scaler
        self.best_params['LightGBM'] = random_search.best_params_
        
        logger.info(f"Fast LightGBM optimization completed. Best score: {best_score:.6f}")
        
        return {
            'model': best_model,
            'scaler': scaler,
            'best_params': random_search.best_params_,
            'best_score': best_score
        }
    
    def create_ensemble(self, ensemble_method: str = 'Voting',
                       weights: Optional[List[float]] = None) -> None:
        """
        Create ensemble model using specified method
        """
        logger.info(f"Creating {ensemble_method} ensemble...")
        
        if ensemble_method == 'Voting':
            # Voting ensemble
            estimators = [(name, model) for name, model in self.base_models.items()]
            
            if weights is None:
                weights = [1.0] * len(estimators)
            
            self.ensemble_model = VotingClassifier(
                estimators=estimators,
                weights=weights,
                n_jobs=self.n_jobs
            )
            
        elif ensemble_method == 'Stacking':
            # Stacking ensemble
            estimators = [(name, model) for name, model in self.base_models.items()]
            
            self.ensemble_model = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(), # Changed to LogisticRegression for classification
                cv=5,
                n_jobs=self.n_jobs
            )
            
        elif ensemble_method == 'Blending':
            # Simple blending (average predictions)
            self.ensemble_model = 'Blending'
            
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
        
        logger.info(f"Ensemble created: {ensemble_method}")
    
    def fit_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the ensemble model
        """
        logger.info("Fitting ensemble model...")
        
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not created. Call create_ensemble() first.")
        
        if self.ensemble_model == 'Blending':
            # For blending, just fit individual models
            for name, model in self.base_models.items():
                if name in self.scalers:
                    X_scaled = self.scalers[name].transform(X)
                else:
                    X_scaled = X
                model.fit(X_scaled, y)
        else:
            # For voting/stacking, fit the ensemble
            self.ensemble_model.fit(X, y)
        
        self.is_fitted = True
        logger.info("Ensemble model fitted successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_ensemble() first.")
        
        if self.ensemble_model == 'Blending':
            # Average predictions from all models
            predictions = []
            for name, model in self.base_models.items():
                if name in self.scalers:
                    X_scaled = self.scalers[name].transform(X)
                else:
                    X_scaled = X
                pred = model.predict(X_scaled)
                predictions.append(pred)
            
            # Average predictions
            ensemble_pred = np.mean(predictions, axis=0)
            
        else:
            # Use ensemble model
            ensemble_pred = self.ensemble_model.predict(X)
        
        return ensemble_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions using the ensemble
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_ensemble() first.")
        
        if self.task_type != 'classification':
            raise ValueError("Probability prediction only available for classification tasks")
        
        if self.ensemble_model == 'Blending':
            # Average probability predictions from all models
            proba_predictions = []
            for name, model in self.base_models.items():
                if name in self.scalers:
                    X_scaled = self.scalers[name].transform(X)
                else:
                    X_scaled = X
                
                # Handle different probability output formats
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)
                    if proba.shape[1] == 2:
                        proba = proba[:, 1]  # Probability of positive class
                    else:
                        proba = proba[:, 0]  # Single probability
                else:
                    # For models without predict_proba, use decision_function
                    if hasattr(model, 'decision_function'):
                        decision = model.decision_function(X_scaled)
                        # Convert to probability using sigmoid
                        proba = 1 / (1 + np.exp(-decision))
                    else:
                        # Fallback to hard predictions converted to probabilities
                        pred = model.predict(X_scaled)
                        proba = (pred > 0).astype(float)
                
                proba_predictions.append(proba)
            
            # Average probability predictions
            ensemble_proba = np.mean(proba_predictions, axis=0)
            
        else:
            # Use ensemble model
            if hasattr(self.ensemble_model, 'predict_proba'):
                proba = self.ensemble_model.predict_proba(X)
                if proba.shape[1] == 2:
                    ensemble_proba = proba[:, 1]  # Probability of positive class
                else:
                    ensemble_proba = proba[:, 0]  # Single probability
            else:
                # Fallback for ensemble models without predict_proba
                pred = self.ensemble_model.predict(X)
                ensemble_proba = (pred > 0).astype(float)
        
        return ensemble_proba
    
    def calibrate_probabilities(self, X: np.ndarray, y: np.ndarray, 
                               method: str = 'isotonic') -> None:
        """
        Calibrate probabilities using OOF data to avoid data leakage
        """
        if self.task_type != 'classification':
            logger.warning("Probability calibration only available for classification tasks")
            return
        
        logger.info(f"Calibrating probabilities using {method} method...")
        
        # Use TimeSeriesSplit for calibration to avoid leakage
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in self.base_models.items():
            try:
                if hasattr(model, 'predict_proba') or hasattr(model, 'decision_function'):
                    # Create calibrated model
                    if method == 'isotonic':
                        calibrated_model = CalibratedClassifierCV(
                            model, method='isotonic', cv=tscv
                        )
                    elif method == 'sigmoid':
                        calibrated_model = CalibratedClassifierCV(
                            model, method='sigmoid', cv=tscv
                        )
                    else:
                        logger.warning(f"Unknown calibration method: {method}")
                        continue
                    
                    # Fit calibrated model
                    if name in self.scalers:
                        X_scaled = self.scalers[name].transform(X)
                    else:
                        X_scaled = X
                    
                    calibrated_model.fit(X_scaled, y)
                    
                    # Store calibrated model
                    self.calibrators[name] = calibrated_model
                    
                    logger.info(f"Calibrated {name} using {method}")
                    
            except Exception as e:
                logger.warning(f"Could not calibrate {name}: {e}")
        
        logger.info(f"Probability calibration completed for {len(self.calibrators)} models")
    
    def get_calibrated_probabilities(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get calibrated probabilities from all models
        """
        if not self.calibrators:
            logger.warning("No calibrated models available. Run calibrate_probabilities() first.")
            return {}
        
        calibrated_probas = {}
        
        for name, calibrated_model in self.calibrators.items():
            try:
                if name in self.scalers:
                    X_scaled = self.scalers[name].transform(X)
                else:
                    X_scaled = X
                
                proba = calibrated_model.predict_proba(X_scaled)
                if proba.shape[1] == 2:
                    calibrated_probas[name] = proba[:, 1]  # Probability of positive class
                else:
                    calibrated_probas[name] = proba[:, 0]  # Single probability
                    
            except Exception as e:
                logger.warning(f"Could not get calibrated probabilities for {name}: {e}")
        
        return calibrated_probas
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance from all models
        """
        feature_importance = {}
        
        for name, model in self.base_models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_)
                else:
                    continue
                
                # Create feature importance dictionary
                imp_dict = dict(zip(feature_names, importance))
                imp_dict = dict(sorted(imp_dict.items(), 
                                     key=lambda x: x[1], reverse=True)[:20])
                
                feature_importance[name] = imp_dict
                
            except Exception as e:
                logger.warning(f"Could not get feature importance for {name}: {e}")
        
        self.feature_importance = feature_importance
        return feature_importance
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_ensemble() first.")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        if self.task_type == 'classification':
            # For classification, we need to calibrate probabilities
            calibrated_model = CalibratedClassifierCV(self.ensemble_model, method='isotonic')
            calibrated_model.fit(X, y)
            y_pred_calibrated = calibrated_model.predict_proba(X)[:, 1] # Probability of positive class
            
            # Calculate metrics for calibrated probabilities
            mse = mean_squared_error(y, y_pred_calibrated)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred_calibrated)
            r2 = r2_score(y, y_pred_calibrated)
            
            # Calculate directional accuracy
            direction_correct = np.sum(np.sign(y) == np.sign(y_pred_calibrated))
            directional_accuracy = direction_correct / len(y)
            
            # Calculate Sharpe ratio (assuming y are returns)
            if np.std(y_pred_calibrated) > 0:
                sharpe_ratio = np.mean(y_pred_calibrated) / np.std(y_pred_calibrated) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Calculate CAGR (assuming y are returns)
            cumulative_returns = np.cumprod(1 + y_pred_calibrated)
            if len(cumulative_returns) > 1:
                cagr = (cumulative_returns[-1] / cumulative_returns[0]) ** (252 / len(cumulative_returns)) - 1
            else:
                cagr = 0
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'sharpe_ratio': sharpe_ratio,
                'cagr': cagr
            }
        else:
            # For regression, metrics are the same as before
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # Calculate directional accuracy
            direction_correct = np.sum(np.sign(y) == np.sign(y_pred))
            directional_accuracy = direction_correct / len(y)
            
            # Calculate Sharpe ratio (assuming y are returns)
            if np.std(y_pred) > 0:
                sharpe_ratio = np.mean(y_pred) / np.std(y_pred) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Calculate CAGR (assuming y are returns)
            cumulative_returns = np.cumprod(1 + y_pred)
            if len(cumulative_returns) > 1:
                cagr = (cumulative_returns[-1] / cumulative_returns[0]) ** (252 / len(cumulative_returns)) - 1
            else:
                cagr = 0
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'sharpe_ratio': sharpe_ratio,
                'cagr': cagr
            }
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save the trained ensemble model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Cannot save untrained model.")
        
        model_data = {
            'base_models': self.base_models,
            'ensemble_model': self.ensemble_model,
            'best_params': self.best_params,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained ensemble model"""
        model_data = joblib.load(filepath)
        
        self.base_models = model_data['base_models']
        self.ensemble_model = model_data['ensemble_model']
        self.best_params = model_data['best_params']
        self.scalers = model_data['scalers']
        self.feature_importance = model_data['feature_importance']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = {
            'n_base_models': len(self.base_models),
            'ensemble_method': type(self.ensemble_model).__name__ if self.ensemble_model != 'Blending' else 'Blending',
            'is_fitted': self.is_fitted,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance
        }
        
        return summary
    
    def optimize_ensemble_weights(self, X: np.ndarray, y: np.ndarray,
                                 method: str = 'sharpe', 
                                 cost_per_trade: float = 0.001,
                                 slippage: float = 0.0005) -> Dict[str, Any]:
        """
        Optimize ensemble weights to maximize trading performance
        
        Args:
            X: Feature matrix
            y: Target values (returns)
            method: Optimization method ('sharpe', 'cagr', 'sharpe_cagr')
            cost_per_trade: Transaction cost per trade
            slippage: Slippage cost per trade
        
        Returns:
            Dictionary with optimal weights and performance metrics
        """
        if self.task_type != 'classification':
            logger.warning("Weight optimization only available for classification tasks")
            return {}
        
        logger.info(f"Optimizing ensemble weights using {method} method...")
        
        # Get OOF probabilities from all models
        oof_probas = self._get_oof_probabilities(X, y)
        
        if not oof_probas:
            logger.error("Could not generate OOF probabilities")
            return {}
        
        # Convert to numpy array
        proba_matrix = np.column_stack(list(oof_probas.values()))
        model_names = list(oof_probas.keys())
        
        # Store OOF probability matrix for diversity control
        self._oof_proba_matrix = proba_matrix
        
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
    
    def _get_oof_probabilities(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate out-of-fold probabilities for all models"""
        logger.info("Generating out-of-fold probabilities...")
        
        # Use TimeSeriesSplit for OOF predictions
        tscv = TimeSeriesSplit(n_splits=5)
        oof_probas = {name: np.zeros(len(y)) for name in self.base_models.keys()}
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train models on this fold
            for name, model in self.base_models.items():
                try:
                    if name in self.scalers:
                        X_train_scaled = self.scalers[name].transform(X_train)
                        X_val_scaled = self.scalers[name].transform(X_val)
                    else:
                        X_train_scaled, X_val_scaled = X_train, X_val
                    
                    # Fit model on training data
                    model.fit(X_train_scaled, y_train)
                    
                    # Get probabilities on validation data
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_val_scaled)
                        if proba.shape[1] == 2:
                            proba = proba[:, 1]  # Probability of positive class
                        else:
                            proba = proba[:, 0]  # Single probability
                    else:
                        # Fallback for models without predict_proba
                        pred = model.predict(X_val_scaled)
                        proba = (pred > 0).astype(float)
                    
                    # Store OOF probabilities
                    oof_probas[name][val_idx] = proba
                    
                except Exception as e:
                    logger.warning(f"Error generating OOF probabilities for {name}: {e}")
        
        logger.info("OOF probabilities generated successfully")
        return oof_probas
    
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
    
    def apply_diversity_control(self, correlation_threshold: float = 0.95) -> Dict[str, float]:
        """
        Apply diversity control by down-weighting highly correlated models
        
        Args:
            correlation_threshold: Threshold above which models are down-weighted
        
        Returns:
            Dictionary of adjusted weights
        """
        if not hasattr(self, 'optimal_weights') or not self.optimal_weights:
            logger.warning("No optimal weights available. Run optimize_ensemble_weights() first.")
            return {}
        
        logger.info(f"Applying diversity control with correlation threshold: {correlation_threshold}")
        
        # Get OOF probabilities for correlation calculation
        if not hasattr(self, '_oof_proba_matrix'):
            logger.warning("No OOF probability matrix available. Run optimize_ensemble_weights() first.")
            return self.optimal_weights
        
        # Calculate pairwise Spearman correlations
        model_names = list(self.optimal_weights.keys())
        n_models = len(model_names)
        correlations = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                corr, _ = spearmanr(
                    self._oof_proba_matrix[:, i], 
                    self._oof_proba_matrix[:, j]
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
        if not hasattr(self, '_oof_proba_matrix'):
            return {}
        
        model_names = list(self.optimal_weights.keys())
        n_models = len(model_names)
        
        # Calculate pairwise correlations
        correlations = {}
        for i in range(n_models):
            for j in range(i+1, n_models):
                corr, _ = spearmanr(
                    self._oof_proba_matrix[:, i], 
                    self._oof_proba_matrix[:, j]
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
    
    def _optimize_threshold(self, ensemble_proba: np.ndarray, y: np.ndarray,
                           cost_per_trade: float, slippage: float) -> float:
        """Find optimal threshold for trading decisions"""
        logger.info("Optimizing trading threshold...")
        
        # Grid search for threshold
        thresholds = np.arange(0.50, 0.71, 0.01)
        best_sharpe = -np.inf
        optimal_threshold = 0.5
        
        for threshold in thresholds:
            returns = self._calculate_trading_returns(ensemble_proba, y, threshold, cost_per_trade, slippage)
            
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    optimal_threshold = threshold
        
        logger.info(f"Optimal threshold: {optimal_threshold:.3f} (Sharpe: {best_sharpe:.3f})")
        return optimal_threshold
    
    def _calculate_trading_returns(self, ensemble_proba: np.ndarray, y: np.ndarray,
                                 threshold: float, cost_per_trade: float, slippage: float) -> np.ndarray:
        """Calculate trading returns based on probability threshold strategy"""
        # Trading signal: long if p > threshold
        signals = (ensemble_proba > threshold).astype(float)
        
        # Position size: (p - threshold) / (1 - threshold), clipped to [0, 1]
        position_sizes = np.clip((ensemble_proba - threshold) / (1 - threshold), 0, 1)
        
        # Apply position sizes to signals
        weighted_signals = signals * position_sizes
        
        # Calculate strategy returns
        strategy_returns = weighted_signals * y
        
        # Apply transaction costs when signals change
        signal_changes = np.diff(weighted_signals, prepend=0)
        transaction_costs = np.abs(signal_changes) * (cost_per_trade + slippage)
        
        # Net strategy returns
        net_returns = strategy_returns - transaction_costs
        
        return net_returns
    
    def _calculate_trading_metrics(self, ensemble_proba: np.ndarray, y: np.ndarray,
                                 threshold: float, cost_per_trade: float, slippage: float) -> Dict[str, float]:
        """Calculate comprehensive trading metrics"""
        returns = self._calculate_trading_returns(ensemble_proba, y, threshold, cost_per_trade, slippage)
        
        # Basic metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Trading metrics
        trades = np.sum(np.abs(np.diff(returns > 0, prepend=False)))
        turnover = np.sum(np.abs(np.diff(returns, prepend=0)))
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trade_count': trades,
            'turnover': turnover
        }
