"""
Advanced Model Ensemble for the Walk-Forward ML Trading System
Includes multiple base models, ensemble methods, and grid search optimization
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    VotingRegressor, StackingRegressor
)
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
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
    """
    
    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.base_models = {}
        self.ensemble_model = None
        self.best_params = {}
        self.feature_importance = {}
        self.scalers = {}
        self.is_fitted = False
        
        logger.info(f"ModelEnsemble initialized with {self.n_jobs} jobs")
        
        # Initialize base models
        self._initialize_base_models()
    
    def _initialize_base_models(self):
        """Initialize all base models with default parameters"""
        
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
                verbose=-1
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
            max_iter=2000,
            random_state=42
        )
        
        logger.info(f"Initialized {len(self.base_models)} base models")
    
    def get_grid_search_params(self) -> Dict[str, Dict[str, List]]:
        """Get comprehensive grid search parameters for all models"""
        
        grid_params = {}
        
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
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [4, 6, 8],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
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
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
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
        
        def optimize_model_wrapper(model_name, base_models, grid_params, X, y, tscv):
            """Optimize a single model"""
            try:
                logger.info(f"Optimizing {model_name}...")
                
                model = base_models[model_name]
                params = grid_params[model_name]
                
                # Create scaler for this model
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Grid search
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
            
            self.ensemble_model = VotingRegressor(
                estimators=estimators,
                weights=weights,
                n_jobs=self.n_jobs
            )
            
        elif ensemble_method == 'Stacking':
            # Stacking ensemble
            estimators = [(name, model) for name, model in self.base_models.items()]
            
            self.ensemble_model = StackingRegressor(
                estimators=estimators,
                final_estimator=ElasticNet(),
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
