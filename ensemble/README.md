# Enhanced Trading Ensemble System

A comprehensive 9-model soft-voting ensemble optimized for trading Sharpe ratio and CAGR with no data leakage.

## Features

### Core Ensemble
- **9 Models**: RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost, LogisticRegression (ElasticNet), RidgeClassifier, SVC (probability=True), MLP
- **Proper Pipelines**: sklearn Pipelines with StandardScaler for SVM/linear/MLP models
- **TimeSeriesSplit**: Walk-forward validation to prevent data leakage

### Probability Calibration
- **OOF-based Calibration**: Uses only out-of-fold data (no leakage)
- **Multiple Methods**: Isotonic and sigmoid calibration
- **Quality Metrics**: Brier score and calibration MAE

### Weight Optimization
- **Trading-focused**: Optimizes for Sharpe ratio, CAGR, or combined score
- **SLSQP Optimization**: Respects bounds [0,1] and sum-to-1 constraint
- **Cost-aware**: Includes transaction costs and slippage

### Threshold Optimization
- **Grid Search**: τ in [0.50, 0.70] with 0.01 step
- **Sharpe Maximization**: Finds optimal threshold for trading decisions
- **Position Sizing**: (p_ens - τ)/(1-τ) clipped to [0,1]

### Diversity Control
- **Correlation Analysis**: Pairwise Spearman correlation of OOF probabilities
- **Penalty Application**: Down-weights models with corr > 0.95
- **Diversity Factor**: (1 - |corr|) applied to weights

### Regime Awareness (Optional)
- **Market Regimes**: VIX/VVIX high vs low, ADX trend vs chop
- **Regime-specific Weights**: Separate weight vectors per regime
- **Fallback**: Defaults to global weights if regime detection fails

### Comprehensive Metrics
- **Per-model**: OOF Sharpe, AUC, Brier, Calibration MAE
- **Ensemble**: Optimal weights, τ, Sharpe, CAGR, MaxDD, trade count, turnover
- **Diversity**: Pairwise correlations, diversity score

### Reproducibility
- **Fixed Seeds**: Reproducible results across runs
- **Artifact Saving**: OOF P, y, weights, τ, metrics JSON
- **Version Info**: Python, numpy, sklearn versions

## Usage

### Basic Usage

```python
from enhanced_ensemble import EnhancedTradingEnsemble

# Initialize ensemble
ensemble = EnhancedTradingEnsemble(random_state=42, n_splits=5)

# Fit models with TimeSeriesSplit
ensemble.fit_models(X, y)

# Calibrate probabilities
ensemble.calibrate_probabilities(method='isotonic')

# Optimize weights for Sharpe
results = ensemble.optimize_ensemble_weights(
    y=y, method='sharpe', cost_per_trade=0.001, slippage=0.0005
)

# Apply diversity control
diversity_weights = ensemble.apply_diversity_control(correlation_threshold=0.95)

# Make predictions
predictions = ensemble.predict_proba(X_new)
```

### Command Line Interface

```bash
# Basic run
python scripts/run_ensemble.py

# With custom parameters
python scripts/run_ensemble.py \
    --data_path data/processed_data.parquet \
    --output_dir results/ensemble \
    --transaction_cost 0.001 \
    --slippage_bps 0.5 \
    --regime_aware \
    --optimization_method sharpe \
    --n_splits 5
```

### Regime-aware Usage

```python
# Fit regime-aware weights
regime_weights = ensemble.fit_regime_aware_weights(
    X, y, regime_method='vix_adx'
)

# Make regime-aware predictions
predictions = ensemble.predict_proba_regime_aware(
    X_new, regime_method='vix_adx'
)
```

## Configuration

### Key Parameters

- `random_state`: Random seed for reproducibility
- `n_splits`: Number of TimeSeriesSplit folds
- `transaction_cost`: Cost per trade (default: 0.001)
- `slippage_bps`: Slippage in basis points (default: 0.5)
- `correlation_threshold`: Diversity control threshold (default: 0.95)

### Threshold Grid

- **Start**: 0.50 (default)
- **Stop**: 0.71 (exclusive)
- **Step**: 0.01
- **Range**: [0.50, 0.70]

## Testing

### Run All Tests

```bash
python tests/run_tests.py
```

### Individual Test Files

- `test_oof_alignment.py`: OOF alignment and data leakage prevention
- `test_calibration.py`: Probability calibration quality
- `test_weight_opt.py`: Weight optimization constraints and performance
- `test_backtest_contract.py`: Backtest contract compliance
- `test_diversity_penalty.py`: Diversity control and penalty

### Test Coverage

- ✅ OOF matrix alignment P (T×M) with y
- ✅ No data leakage in calibration
- ✅ Weight optimization bounds and constraints
- ✅ Next-bar execution and position sizing
- ✅ Diversity penalty for correlated models

## Architecture

### Directory Structure

```
ensemble/
├── __init__.py              # Package initialization
├── enhanced_ensemble.py     # Main ensemble class
├── calibration.py           # Probability calibration
├── weight_opt.py           # Weight optimization
└── README.md               # This file

scripts/
└── run_ensemble.py         # Top-level entry script

tests/
├── test_oof_alignment.py   # OOF alignment tests
├── test_calibration.py     # Calibration tests
├── test_weight_opt.py      # Weight optimization tests
├── test_backtest_contract.py # Backtest contract tests
├── test_diversity_penalty.py # Diversity control tests
└── run_tests.py            # Test runner
```

### Key Classes

- `EnhancedTradingEnsemble`: Main ensemble class
- `ProbabilityCalibrator`: Handles probability calibration
- `WeightOptimizer`: Optimizes ensemble weights

## Performance

### Optimization Methods

1. **Sharpe**: Maximizes Sharpe ratio
2. **CAGR**: Maximizes Compound Annual Growth Rate
3. **Combined**: Maximizes Sharpe + CAGR

### Computational Complexity

- **Model Fitting**: O(n_samples × n_features × n_models)
- **OOF Generation**: O(n_samples × n_features × n_models × n_splits)
- **Weight Optimization**: O(n_iterations × n_models)
- **Threshold Search**: O(n_thresholds × n_samples)

## Best Practices

### Data Preparation

- Use TimeSeriesSplit for temporal validation
- Ensure no future data leakage in features
- Align target returns with feature timestamps

### Model Selection

- Start with all 9 models
- Use diversity control to remove redundant models
- Monitor individual model performance

### Hyperparameter Tuning

- Tune individual models before ensemble
- Use cross-validation with TimeSeriesSplit
- Avoid overfitting to recent data

### Risk Management

- Monitor correlation between models
- Apply diversity penalties for high correlations
- Use regime-aware weights for different market conditions

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce n_splits or use smaller datasets
3. **Optimization Failures**: Check data quality and model convergence
4. **Calibration Errors**: Verify OOF probabilities are valid

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run ensemble with debug logging
ensemble = EnhancedTradingEnsemble(random_state=42, n_splits=3)
```

## Dependencies

- Python 3.8+
- numpy
- pandas
- scikit-learn
- scipy
- xgboost
- lightgbm
- catboost
- joblib

## License

This project is licensed under the MIT License.
