# Enhanced Trading Ensemble System

## Overview

The Enhanced Trading Ensemble System is a sophisticated 9-model soft-voting ensemble optimized for trading performance. It implements proper machine learning pipelines with TimeSeriesSplit to avoid data leakage, probability calibration, weight optimization for Sharpe ratio and CAGR, and comprehensive diversity control.

## 🎯 Key Features

### ✅ **Core Functionality**
- **9-Model Ensemble**: RF, GB, XGB, LGBM, CatBoost, LogisticRegression, RidgeClassifier, SVC, MLP
- **Proper ML Pipelines**: sklearn Pipeline with StandardScaler, no data leakage
- **TimeSeriesSplit**: Walk-forward validation for temporal data
- **Probability Outputs**: Classification with calibrated probability estimates
- **OOF Matrix**: T×M matrix of out-of-fold probabilities for each model

### ✅ **Advanced Optimization**
- **Weight Optimization**: SLSQP-based optimization for Sharpe ratio, CAGR, or combined
- **Threshold Optimization**: Grid search for optimal trading threshold τ ∈ [0.50, 0.70]
- **Probability Calibration**: Platt scaling and isotonic regression using OOF data only
- **Diversity Control**: Spearman correlation-based model down-weighting

### ✅ **Trading Strategy**
- **Position Sizing**: (p_ens - τ)/(1-τ) clipped to [0,1]
- **Transaction Costs**: Integrated cost_per_trade and slippage
- **Next-Bar Execution**: Realistic trading simulation
- **Performance Metrics**: Sharpe, CAGR, MaxDD, turnover, trade count

### ✅ **Comprehensive Metrics**
- **Per-Model**: OOF Sharpe, AUC, Brier score, Calibration MAE
- **Ensemble**: Optimal weights, threshold, Sharpe, CAGR, MaxDD
- **Diversity**: Pairwise correlations, diversity score, high-correlation pairs

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install -r requirements.txt

# Additional packages for enhanced ensemble
pip install scikit-learn scipy xgboost lightgbm catboost
```

### Basic Usage

```python
from enhanced_ensemble import EnhancedTradingEnsemble
import numpy as np

# Initialize ensemble
ensemble = EnhancedTradingEnsemble(random_state=42, n_splits=5)

# Prepare your data (X: features, y: binary targets)
X = np.random.randn(1000, 20)  # 1000 samples, 20 features
y = np.random.randint(0, 2, 1000)  # Binary targets

# Fit models using TimeSeriesSplit
ensemble.fit_models(X, y)

# Calibrate probabilities
ensemble.calibrate_probabilities(method='isotonic')

# Optimize weights for Sharpe ratio
results = ensemble.optimize_ensemble_weights(
    y, method='sharpe', cost_per_trade=0.001, slippage=0.0005
)

# Apply diversity control
diversity_weights = ensemble.apply_diversity_control(correlation_threshold=0.95)

# Make predictions
probabilities = ensemble.predict_proba(X)

# Evaluate individual models
individual_metrics = ensemble.evaluate_individual_models(X, y)
```

### Complete Workflow Example

```python
# Run the complete example
python enhanced_ensemble_example.py
```

## 📊 System Architecture

### Model Pipeline Structure

```
Input Data → StandardScaler → Classifier → Probability Output
     ↓              ↓            ↓              ↓
Feature Matrix → Scaled Features → Model → Raw Probabilities
```

### Ensemble Workflow

```
1. Initialize 9 Models with Pipelines
   ↓
2. Fit Models using TimeSeriesSplit
   ↓
3. Generate OOF Probabilities (T×M matrix)
   ↓
4. Calibrate Probabilities (OOF only)
   ↓
5. Optimize Ensemble Weights (Sharpe/CAGR)
   ↓
6. Optimize Trading Threshold
   ↓
7. Apply Diversity Control
   ↓
8. Generate Final Predictions
```

## 🔧 Configuration

### Ensemble Parameters

```python
ensemble = EnhancedTradingEnsemble(
    random_state=42,      # Reproducibility
    n_splits=5           # TimeSeriesSplit folds
)
```

### Weight Optimization Methods

```python
# Sharpe ratio optimization
results = ensemble.optimize_ensemble_weights(
    y, method='sharpe', cost_per_trade=0.001, slippage=0.0005
)

# CAGR optimization
results = ensemble.optimize_ensemble_weights(
    y, method='cagr', cost_per_trade=0.001, slippage=0.0005
)

# Combined Sharpe + CAGR optimization
results = ensemble.optimize_ensemble_weights(
    y, method='sharpe_cagr', cost_per_trade=0.001, slippage=0.0005
)
```

### Calibration Methods

```python
# Isotonic regression (more flexible)
ensemble.calibrate_probabilities(method='isotonic')

# Platt scaling (parametric)
ensemble.calibrate_probabilities(method='sigmoid')
```

## 📈 Performance Metrics

### Individual Model Metrics

- **AUC**: Area under ROC curve
- **Brier Score**: Probability calibration quality (lower is better)
- **Calibration MAE**: Mean absolute error of calibration
- **OOF Sharpe**: Out-of-fold trading Sharpe ratio

### Ensemble Metrics

- **Optimal Weights**: Model weights that maximize Sharpe/CAGR
- **Optimal Threshold**: Trading threshold τ that maximizes Sharpe
- **Performance Metrics**: Sharpe, CAGR, MaxDD, turnover, trade count

### Diversity Metrics

- **Pairwise Correlations**: Spearman correlation between model probabilities
- **Average Correlation**: Mean correlation across all model pairs
- **Diversity Score**: 1 - average_correlation (higher is better)
- **High Correlation Pairs**: Models with correlation > 0.95

## 🎨 Visualization

The system generates comprehensive visualizations:

- **Ensemble Weights**: Comparison of optimal vs. diversity-adjusted weights
- **Individual Model Performance**: AUC, Brier, Calibration MAE, Sharpe ratios
- **Trading Strategy Performance**: Probabilities over time, cumulative returns
- **Model Correlation Heatmap**: Pairwise correlation matrix

## 💾 Data Persistence

### Save Ensemble

```python
# Save trained ensemble
ensemble.save_ensemble('models/ensemble.pkl')
```

### Load Ensemble

```python
# Load saved ensemble
new_ensemble = EnhancedTradingEnsemble()
new_ensemble.load_ensemble('models/ensemble.pkl')

# Make predictions
probabilities = new_ensemble.predict_proba(X)
```

### Save Results

```python
# Save comprehensive results
save_results(results, ensemble, save_path='results/')
```

Files saved:
- `enhanced_ensemble.pkl`: Trained ensemble
- `ensemble_results.json`: Complete results summary
- `model_performance.csv`: Individual model metrics
- `reports/`: Visualization plots

## 🧪 Testing

Run the comprehensive test suite:

```bash
python -m pytest test_enhanced_ensemble.py -v
```

Or run individual tests:

```bash
python test_enhanced_ensemble.py
```

## 🔍 Advanced Usage

### Custom Model Configuration

```python
# Access individual models
for name, pipeline in ensemble.models.items():
    print(f"{name}: {type(pipeline.named_steps['classifier'])}")
    
    # Modify model parameters
    if name == 'RandomForest':
        pipeline.named_steps['classifier'].n_estimators = 200
```

### Custom Weight Constraints

```python
# The system automatically handles:
# - Weights sum to 1
# - All weights >= 0
# - SLSQP optimization with bounds [0,1]
```

### Regime-Aware Weights (Future Feature)

```python
# Planned feature for regime-specific weight optimization
# - VIX/VVIX-based regime detection
# - Separate weight vectors per regime
# - Dynamic weight selection at inference
```

## 📚 API Reference

### Core Methods

#### `__init__(random_state=42, n_splits=5)`
Initialize the ensemble with specified parameters.

#### `fit_models(X, y)`
Fit all models using TimeSeriesSplit for OOF probability generation.

#### `calibrate_probabilities(method='isotonic')`
Calibrate probabilities using OOF data only.

#### `optimize_ensemble_weights(y, method='sharpe', cost_per_trade=0.001, slippage=0.0005)`
Optimize ensemble weights for specified objective.

#### `apply_diversity_control(correlation_threshold=0.95)`
Apply diversity control by down-weighting highly correlated models.

#### `predict_proba(X)`
Generate probability predictions using optimized weights.

#### `evaluate_individual_models(X, y)`
Evaluate individual model performance metrics.

#### `get_model_diversity_metrics()`
Calculate model diversity and correlation metrics.

### Utility Methods

#### `save_ensemble(filepath)`
Save the trained ensemble to disk.

#### `load_ensemble(filepath)`
Load a saved ensemble from disk.

#### `get_ensemble_summary()`
Get comprehensive ensemble summary.

## 🚨 Important Notes

### Data Leakage Prevention
- **TimeSeriesSplit**: All validation uses temporal splits
- **OOF Probabilities**: Generated using cross-validation only
- **Calibration**: Uses OOF data, never full training data
- **Pipeline Scaling**: StandardScaler fitted per-fold

### Reproducibility
- **Fixed Seeds**: All random operations use fixed seeds
- **Deterministic**: Same input produces same output
- **Version Pinning**: Dependencies are pinned for consistency

### Performance Considerations
- **Memory**: OOF matrix can be large for big datasets
- **Computation**: Weight optimization uses SLSQP (efficient)
- **Scalability**: Designed for medium datasets (1000-10000 samples)

## 🔮 Future Enhancements

### Planned Features
- **Regime Detection**: VIX/VVIX-based market regime identification
- **Dynamic Weights**: Time-varying weight optimization
- **Feature Selection**: Automated feature importance and selection
- **Hyperparameter Tuning**: Bayesian optimization for model parameters
- **Real-time Updates**: Incremental model updates for live trading

### Integration Points
- **Data Sources**: Yahoo Finance, Alpha Vantage, custom APIs
- **Trading Platforms**: Interactive Brokers, Alpaca, custom systems
- **Risk Management**: Position sizing, stop-loss, portfolio constraints
- **Backtesting**: Walk-forward analysis with realistic constraints

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd enhanced-trading-ensemble

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v

# Run linting
flake8 enhanced_ensemble.py
black enhanced_ensemble.py
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Comprehensive docstrings
- Unit tests for all functionality
- Integration tests for workflows

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **scikit-learn**: Core ML framework and pipelines
- **scipy**: Optimization algorithms
- **xgboost/lightgbm/catboost**: Gradient boosting implementations
- **pandas/numpy**: Data manipulation and numerical computing

## 📞 Support

For questions, issues, or contributions:

1. **Issues**: Create a GitHub issue
2. **Discussions**: Use GitHub discussions
3. **Documentation**: Check this README and inline code comments
4. **Examples**: Run `enhanced_ensemble_example.py` for complete workflow

---

**Happy Trading! 🚀📈**

