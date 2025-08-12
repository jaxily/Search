# Walk-Forward Ensemble ML Trading System

A professional, multi-threaded machine learning ensemble system optimized for M1 chips, designed to achieve Sharpe ratio > 3 and CAGR > 25% through advanced walk-forward analysis and grid search optimization.

## 🚀 Features

### Core Capabilities
- **Multi-threaded Processing**: Fully utilizes M1 chip cores for optimal performance
- **Walk-Forward Analysis**: Time series cross-validation for robust backtesting
- **Ensemble Learning**: Combines multiple ML models (Random Forest, XGBoost, LightGBM, CatBoost, SVM, Neural Networks)
- **Grid Search Optimization**: Automated hyperparameter tuning for all models
- **Advanced Feature Engineering**: 450+ features with technical indicators and rolling statistics
- **Null Value Handling**: Intelligent handling of rolling average nulls in first 100 columns

### Performance Targets
- **Sharpe Ratio**: > 3.0
- **CAGR**: > 25%
- **Maximum Drawdown**: < 15%
- **Win Rate**: > 60%

### Technical Features
- **M1 Chip Optimization**: Numba JIT compilation and parallel processing
- **Memory Management**: Efficient handling of large datasets (15+ years, 450+ columns)
- **Feature Selection**: PCA and statistical feature selection for dimensionality reduction
- **Risk Management**: Kelly criterion position sizing and correlation analysis

## 🏗️ Architecture

```
├── config.py              # System configuration and parameters
├── data_processor.py      # Data loading, cleaning, and feature engineering
├── models.py              # Ensemble model creation and optimization
├── walkforward.py         # Walk-forward analysis implementation
├── performance.py         # Performance metrics calculation
├── main.py               # Main execution script
└── requirements.txt      # Python dependencies
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- macOS with M1 chip (optimized) or other systems
- 8GB+ RAM recommended

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Search

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Usage

### Basic Usage
```bash
# Run the complete system
python main.py --data-file your_data.csv --ensemble-method Voting

# Run with different ensemble methods
python main.py --data-file your_data.csv --ensemble-method Stacking
python main.py --data-file your_data.csv --ensemble-method Blending

# Save processed data for future use
python main.py --data-file your_data.csv --save-processed

# Skip walk-forward analysis (only optimize ensemble)
python main.py --data-file your_data.csv --skip-walkforward
```

### Data Format
Your CSV file should have:
- **Index**: Date column (will be parsed automatically)
- **Columns**: 450+ features including price data, technical indicators, etc.
- **First 100 columns**: May contain nulls due to rolling averages (handled automatically)

### Expected Output
The system will generate:
- **Processed data**: Cleaned and feature-engineered dataset
- **Optimized models**: Trained ensemble models
- **Walk-forward results**: Comprehensive backtesting results
- **Performance plots**: Visual analysis of results
- **Final report**: Detailed performance summary and recommendations

## ⚙️ Configuration

### System Configuration (`config.py`)
```python
SYSTEM_CONFIG = {
    'max_workers': os.cpu_count(),  # Utilize all M1 cores
    'chunk_size': 1000,            # Data processing chunk size
    'memory_limit': '8GB',         # Memory limit
    'use_numba': True,             # Enable Numba JIT compilation
    'precision': 'float64'         # Calculation precision
}
```

### Performance Targets
```python
PERFORMANCE_TARGETS = {
    'min_sharpe': 3.0,            # Minimum Sharpe ratio
    'min_cagr': 0.25,             # Minimum CAGR (25%)
    'max_drawdown': -0.15,        # Maximum drawdown (-15%)
    'min_win_rate': 0.6,          # Minimum win rate (60%)
    'max_volatility': 0.25        # Maximum volatility (25%)
}
```

### Walk-Forward Configuration
```python
WALKFORWARD_CONFIG = {
    'initial_train_size': 1260,    # 5 years initial training
    'step_size': 63,              # 3 months step size
    'min_train_size': 504,        # 2 years minimum training
    'validation_split': 0.3,      # Validation set size
    'retrain_frequency': 'monthly'
}
```

## 🔧 Model Ensemble

### Base Models
- **Random Forest**: Robust tree-based ensemble
- **XGBoost**: Gradient boosting with regularization
- **LightGBM**: Light gradient boosting machine
- **CatBoost**: Categorical boosting
- **SVM**: Support Vector Machine
- **Neural Network**: Multi-layer perceptron
- **ElasticNet**: Linear model with L1/L2 regularization
- **Gradient Boosting**: Traditional gradient boosting

### Ensemble Methods
- **Voting**: Weighted average of predictions
- **Stacking**: Meta-learner trained on base model predictions
- **Blending**: Simple averaging of predictions

## 📊 Performance Metrics

### Return Metrics
- Total Return, CAGR, Annualized Mean Return

### Risk Metrics
- Sharpe Ratio, Sortino Ratio, Maximum Drawdown, VaR, CVaR

### Trading Metrics
- Win Rate, Profit Factor, Win/Loss Ratio, Hit Rate

### Advanced Metrics
- Information Ratio, Calmar Ratio, Sterling Ratio, Treynor Ratio, Jensen's Alpha

## 🚀 M1 Chip Optimization

### Multi-threading
- Utilizes all available CPU cores
- Parallel model optimization
- Concurrent walk-forward analysis

### Numba JIT Compilation
- Accelerates numerical computations
- Optimized rolling calculations
- Reduced memory overhead

### Memory Management
- Efficient data chunking
- Smart feature selection
- Optimized data types

## 📈 Walk-Forward Analysis

### Process
1. **Initial Training**: 5 years of data for initial model training
2. **Validation**: 30% of training data for model selection
3. **Testing**: 3-month forward testing period
4. **Rolling**: Move forward 3 months and retrain
5. **Repeat**: Continue until all data is processed

### Benefits
- **No Look-ahead Bias**: Future data never used for training
- **Realistic Performance**: Simulates actual trading conditions
- **Model Stability**: Tests model robustness over time
- **Performance Tracking**: Monitors performance degradation

## 🔍 Feature Engineering

### Technical Indicators
- Moving averages (SMA, EMA)
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility indicators (Bollinger Bands, ATR)
- Trend indicators (ADX, Williams %R, CCI)

### Rolling Features
- Rolling means and standard deviations
- Multiple time windows (5, 10, 20, 50, 100, 200 days)
- Price ratios and correlations

### Dimensionality Reduction
- Statistical feature selection
- Principal Component Analysis (PCA)
- Correlation-based filtering

## 📋 Output Files

### Generated Directories
```
├── data/           # Processed data files
├── models/         # Trained ensemble models
├── results/        # Walk-forward analysis results
├── reports/        # Performance reports and plots
├── logs/          # System execution logs
└── cache/         # Temporary processing files
```

### Key Outputs
- `processed_data.parquet`: Cleaned and feature-engineered dataset
- `optimized_ensemble.pkl`: Trained ensemble model
- `walkforward_results.pkl`: Complete walk-forward analysis results
- `performance_plots.png`: Visual performance analysis
- `final_report.txt`: Comprehensive performance summary

## 🛠️ Customization

### Adding New Models
```python
# In models.py, add to _initialize_base_models()
self.base_models['YourModel'] = YourModelClass(
    param1=value1,
    param2=value2
)
```

### Modifying Feature Engineering
```python
# In data_processor.py, add to engineer_features()
def _create_custom_features(self, data):
    # Your custom feature logic
    return data
```

### Adjusting Performance Targets
```python
# In config.py, modify PERFORMANCE_TARGETS
PERFORMANCE_TARGETS = {
    'min_sharpe': 4.0,    # Increase Sharpe target
    'min_cagr': 0.30,     # Increase CAGR target
    # ... other targets
}
```

## 📊 Monitoring and Debugging

### Logging
- Comprehensive logging throughout the system
- Performance metrics tracking
- Error handling and debugging information

### Progress Tracking
- Progress bars for long-running operations
- Real-time performance updates
- Memory and CPU usage monitoring

### Error Handling
- Graceful failure handling
- Detailed error messages
- Recovery mechanisms

## 🚨 Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce chunk_size in config.py
2. **Slow Performance**: Ensure Numba is enabled and multi-threading is working
3. **Import Errors**: Check all dependencies are installed
4. **Data Format Issues**: Ensure CSV has proper date index and numeric columns

### Performance Tips
- Use SSD storage for large datasets
- Ensure adequate RAM (8GB+ recommended)
- Close other applications during execution
- Monitor system resources during processing

## 📚 Advanced Usage

### Custom Grid Search
```python
# Modify grid search parameters in models.py
def get_grid_search_params(self):
    grid_params = {}
    grid_params['YourModel'] = {
        'param1': [value1, value2, value3],
        'param2': [value1, value2]
    }
    return grid_params
```

### Custom Performance Metrics
```python
# Add custom metrics in performance.py
def _calculate_custom_metric(self, returns):
    # Your custom calculation
    return custom_value
```

### Batch Processing
```python
# Process multiple datasets
for data_file in data_files:
    python main.py --data-file data_file --save-processed
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add type hints
- Include docstrings
- Write comprehensive tests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with scikit-learn, XGBoost, LightGBM, and CatBoost
- Optimized for Apple M1 chip architecture
- Inspired by professional quantitative trading systems

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review the code examples

---

**Note**: This system is designed for educational and research purposes. Always validate results and use appropriate risk management in live trading environments.
