"""
Configuration file for the Walk-Forward Ensemble ML Trading System
Optimized for M1 chip with multi-threading capabilities
Enhanced with auto-detection and smart parameter selection
"""

import os
from pathlib import Path

# System Configuration
SYSTEM_CONFIG = {
    'max_workers': os.cpu_count(),  # Utilize all M1 cores
    'chunk_size': 1000,  # Data processing chunk size
    'memory_limit': '8GB',  # Memory limit for processing
    'use_numba': False,  # Disable Numba JIT compilation to avoid threading conflicts
    'precision': 'float64',  # Precision for calculations
    'graceful_shutdown': True,  # Enable graceful shutdown on Ctrl-C
    'auto_save_interval': 300,  # Auto-save every 5 minutes
    'progress_bar': True  # Show progress bars
}

# Data Configuration
DATA_CONFIG = {
    'data_path': 'data/',
    'start_date': '2009-01-01',
    'end_date': '2024-12-31',
    'lookback_period': 252,  # 1 year of trading days
    'forecast_horizon': 21,  # 1 month forward
    'null_threshold': 0.1,  # Maximum null ratio for features
    'min_data_points': 1000,  # Minimum data points required
    'auto_detect_splits': True,  # Auto-detect optimal train/test splits
    'min_split_size': 0.1,  # Minimum split size (10%)
    'max_split_size': 0.8,  # Maximum split size (80%)
    'split_validation_method': 'time_series_cv',  # Split validation method
    'auto_feature_selection': True,  # Auto-select optimal features
    'max_features': 200,  # Maximum number of features to keep
    'feature_selection_method': 'mutual_info'  # Feature selection method
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'technical_indicators': [
        'SMA', 'EMA', 'RSI', 'MACD', 'BB_upper', 'BB_lower',
        'ATR', 'Stochastic', 'Williams_R', 'CCI', 'ADX'
    ],
    'rolling_windows': [5, 10, 20, 50, 100, 200],
    'price_features': ['open', 'high', 'low', 'close', 'volume'],
    'volatility_features': ['returns', 'log_returns', 'volatility'],
    'correlation_features': True,
    'pca_components': 50,  # Reduce dimensionality
    'auto_window_selection': True,  # Auto-select optimal rolling windows
    'max_correlation_threshold': 0.95,  # Maximum correlation between features
    'auto_feature_engineering': True,  # Auto-engineer features based on data
    'feature_importance_threshold': 0.01  # Minimum feature importance to keep
}

# Model Configuration
MODEL_CONFIG = {
    'base_models': [
        'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost',
        'SVM', 'NeuralNetwork', 'ElasticNet', 'GradientBoosting'
    ],
    'ensemble_methods': ['Voting', 'Stacking', 'Blending'],
    'cross_validation_folds': 5,
    'test_size': 0.2,
    'random_state': 42,
    'auto_model_selection': True,  # Auto-select best performing models
    'auto_hyperparameter_tuning': True,  # Auto-tune hyperparameters
    'model_performance_threshold': 0.6,  # Minimum R² score to include in ensemble
    'ensemble_optimization_method': 'genetic_algorithm',  # Ensemble optimization method
    'max_ensemble_size': 5  # Maximum number of models in ensemble
}

# Walk-Forward Configuration
WALKFORWARD_CONFIG = {
    'initial_train_size': 1260,  # 5 years initial training
    'step_size': 63,  # 3 months step size
    'min_train_size': 504,  # 2 years minimum training
    'validation_split': 0.3,
    'retrain_frequency': 'monthly',
    'auto_window_sizing': True,  # Auto-detect optimal window sizes
    'min_window_size': 252,  # Minimum window size (1 year)
    'max_window_size': 2520,  # Maximum window size (10 years)
    'adaptive_step_size': True,  # Adapt step size based on data volatility
    'early_stopping': True,  # Enable early stopping for poor performance
    'performance_threshold': 0.5  # Minimum performance to continue
}

# Grid Search Configuration
GRIDSEARCH_CONFIG = {
    'cv_folds': 5,
    'n_jobs': -1,  # Use all available cores
    'scoring': ['sharpe_ratio', 'cagr', 'max_drawdown'],
    'refit': 'sharpe_ratio',  # Optimize for Sharpe ratio
    'verbose': 1,
    'auto_parameter_ranges': True,  # Auto-detect parameter ranges
    'parameter_search_method': 'bayesian_optimization',  # Use Bayesian optimization
    'max_iterations': 100,  # Maximum optimization iterations
    'convergence_threshold': 0.001,  # Convergence threshold
    'fast_lightgbm_optimization': True,  # Use fast LightGBM optimization
    'lightgbm_cv_folds': 3,  # Reduced CV folds for LightGBM
    'lightgbm_early_stopping': True,  # Enable early stopping for LightGBM
    'lightgbm_random_search_iterations': 20,  # Number of random search iterations
    'fast_neural_network_optimization': True,  # Use fast neural network optimization
    'neural_network_cv_folds': 3,  # Reduced CV folds for neural network
    'neural_network_early_stopping': True,  # Enable early stopping for neural network
    'neural_network_max_iter': 1000,  # Reduced max iterations for neural network
    'skip_neural_network_optimization': False,  # Skip neural network optimization entirely if True
    'neural_network_verbose': 0  # Verbosity level for neural network training
}

# Performance Targets
PERFORMANCE_TARGETS = {
    'min_sharpe': 3.0,
    'min_cagr': 0.25,  # 25%
    'max_drawdown': -0.15,  # -15%
    'min_win_rate': 0.6,  # 60%
    'max_volatility': 0.25,  # 25%
    'auto_target_adjustment': True,  # Auto-adjust targets based on market conditions
    'market_regime_detection': True,  # Detect market regimes
    'regime_specific_targets': True  # Use regime-specific performance targets
}

# Risk Management
RISK_CONFIG = {
    'position_sizing': 'kelly_criterion',
    'max_position_size': 0.1,  # 10% max position
    'stop_loss': 0.05,  # 5% stop loss
    'take_profit': 0.15,  # 15% take profit
    'max_correlation': 0.7,
    'auto_risk_adjustment': True,  # Auto-adjust risk based on volatility
    'volatility_scaling': True,  # Scale positions by volatility
    'dynamic_stop_loss': True,  # Dynamic stop-loss based on volatility
    'max_portfolio_risk': 0.02  # Maximum portfolio risk per trade
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'auto_log_level': True,  # Auto-adjust log level based on system load
    'log_to_file': True,  # Log to file
    'log_to_console': True,  # Log to console
    'log_rotation': True,  # Enable log rotation
    'max_log_size': '100MB'  # Maximum log file size
}

# Auto-Detection Configuration
AUTO_DETECT_CONFIG = {
    'enabled': True,  # Enable auto-detection
    'auto_clean_data': True,  # Auto-clean data
    'auto_feature_engineering': True,  # Auto-engineer features
    'auto_model_selection': True,  # Auto-select models
    'auto_split_detection': True,  # Auto-detect train/test splits
    'auto_window_sizing': True,  # Auto-detect window sizes
    'auto_hyperparameter_tuning': True,  # Auto-tune hyperparameters
    'chunk_size_range': [100, 200, 500, 1000],  # Chunk sizes to test
    'use_numba_range': [False],  # Numba settings to test (disabled for threading safety)
    'max_features_range': [50, 100, 200],  # Max features to test
    'pca_components_range': [20, 30, 50]  # PCA components to test
}

# Paths
PATHS = {
    'data': Path('data/'),
    'models': Path('models/'),
    'results': Path('results/'),
    'logs': Path('logs/'),
    'reports': Path('reports/'),
    'cache': Path('cache/'),
    'temp': Path('cache/temp/'),  # Temporary files
    'backup': Path('cache/backup/')  # Backup files
}

# Create directories if they don't exist
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)
