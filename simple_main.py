#!/usr/bin/env python3
"""
Simplified Main Script for the Enhanced Walk-Forward Ensemble ML Trading System
Properly handles data types and provides a working example
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import warnings
import numpy as np
import signal
import time
import psutil
import gc
from datetime import datetime
import pickle
import json
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"simple_trading_system_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_and_process_data(data_file: str, logger):
    """Load and process data with proper type handling"""
    logger.info(f"📥 Loading data from: {data_file}")
    
    try:
        # Load data
        data = pd.read_csv(data_file)
        logger.info(f"✅ Data loaded: {data.shape}")
        
        # Identify column types properly
        numeric_columns = []
        categorical_columns = []
        
        for col in data.columns:
            if col in ['Date', 'Ticker']:
                categorical_columns.append(col)
            elif data[col].dtype in ['object', 'string']:
                # Check if it can be converted to numeric
                try:
                    pd.to_numeric(data[col], errors='coerce')
                    # If more than 50% are numeric, treat as numeric
                    non_null_count = pd.to_numeric(data[col], errors='coerce').notna().sum()
                    if non_null_count > len(data) * 0.5:
                        numeric_columns.append(col)
                    else:
                        categorical_columns.append(col)
                except:
                    categorical_columns.append(col)
            else:
                numeric_columns.append(col)
        
        logger.info(f"🔢 Numeric columns: {len(numeric_columns)}")
        logger.info(f"📝 Categorical columns: {len(categorical_columns)}")
        
        # Process numeric columns
        for col in numeric_columns:
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                # Fill NaN values appropriately
                if col in ['Volume', 'Volume_MA_20', 'Volume_MA_50']:
                    data[col] = data[col].fillna(0)
                else:
                    data[col] = data[col].ffill().bfill()
            except Exception as e:
                logger.warning(f"⚠️  Could not process numeric column {col}: {e}")
        
        # Process categorical columns
        for col in categorical_columns:
            if col == 'Date':
                data[col] = pd.to_datetime(data[col], errors='coerce')
                data = data.dropna(subset=[col])
            elif col == 'Ticker':
                # Keep as is
                pass
            else:
                # Fill NaN values
                data[col] = data[col].fillna('Unknown')
        
        # Create target variable
        if 'Close' in data.columns:
            data['returns'] = data['Close'].pct_change()
            data = data.iloc[1:].reset_index(drop=True)
            logger.info("🎯 Target variable 'returns' created")
        
        # Final cleanup
        data = data.dropna()
        logger.info(f"✅ Final data shape: {data.shape}")
        
        return data
        
    except Exception as e:
        logger.error(f"❌ Error processing data: {e}")
        raise

def create_features(data, logger):
    """Create additional features"""
    logger.info("🔧 Creating additional features...")
    
    try:
        # Technical indicators
        if 'Close' in data.columns:
            # Moving averages
            data['SMA_5'] = data['Close'].rolling(5).mean()
            data['SMA_10'] = data['Close'].rolling(10).mean()
            
            # Volatility
            data['volatility_5'] = data['returns'].rolling(5).std()
            data['volatility_10'] = data['returns'].rolling(10).std()
            
            # Price momentum
            data['price_momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
            data['price_momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
            
            # Volume features
            if 'Volume' in data.columns:
                data['volume_ma_5'] = data['Volume'].rolling(5).mean()
                data['volume_ratio'] = data['Volume'] / data['volume_ma_5']
        
        # Remove rows with NaN values from new features
        data = data.dropna()
        logger.info(f"✅ Features created, final shape: {data.shape}")
        
        return data
        
    except Exception as e:
        logger.error(f"❌ Error creating features: {e}")
        raise

def prepare_features_and_target(data, logger):
    """Prepare features and target for modeling"""
    logger.info("🎯 Preparing features and target...")
    
    try:
        # Select feature columns (exclude non-numeric and target)
        exclude_cols = ['Date', 'Ticker', 'returns']
        feature_cols = [col for col in data.columns if col not in exclude_cols and data[col].dtype in ['float64', 'int64']]
        
        X = data[feature_cols]
        y = data['returns']
        
        logger.info(f"✅ Features shape: {X.shape}")
        logger.info(f"✅ Target shape: {y.shape}")
        logger.info(f"✅ Feature columns: {len(feature_cols)}")
        
        return X, y, feature_cols
        
    except Exception as e:
        logger.error(f"❌ Error preparing features: {e}")
        raise

def run_simple_analysis(data_file: str, ensemble_method: str = "Voting", skip_walkforward: bool = False):
    """Run a simple analysis of the trading system"""
    logger = setup_logging()
    
    try:
        logger.info("🚀 SIMPLIFIED TRADING SYSTEM ANALYSIS")
        logger.info("=" * 50)
        
        # Load and process data
        data = load_and_process_data(data_file, logger)
        
        # Create features
        data = create_features(data, logger)
        
        # Prepare features and target
        X, y, feature_cols = prepare_features_and_target(data, logger)
        
        # Basic analysis
        logger.info("\n📊 BASIC ANALYSIS RESULTS:")
        logger.info(f"   Dataset size: {len(data)} rows")
        logger.info(f"   Features: {len(feature_cols)}")
        logger.info(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")
        logger.info(f"   Average return: {y.mean():.6f}")
        logger.info(f"   Return volatility: {y.std():.6f}")
        logger.info(f"   Sharpe ratio: {y.mean() / y.std():.6f}")
        
        # Feature importance (simple correlation)
        correlations = []
        for col in feature_cols:
            corr = abs(data[col].corr(data['returns']))
            correlations.append((col, corr))
        
        # Sort by correlation
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"\n🔍 TOP 10 FEATURES BY CORRELATION:")
        for i, (col, corr) in enumerate(correlations[:10]):
            logger.info(f"   {i+1:2d}. {col:20s}: {corr:.4f}")
        
        logger.info(f"\n🎉 Analysis completed successfully!")
        logger.info(f"   Data file: {data_file}")
        logger.info(f"   Ensemble method: {ensemble_method}")
        logger.info(f"   Skip walkforward: {skip_walkforward}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simplified Trading System")
    parser.add_argument("--data-file", required=True, help="Path to data file")
    parser.add_argument("--ensemble-method", default="Voting", help="Ensemble method")
    parser.add_argument("--skip-walkforward", action="store_true", help="Skip walkforward analysis")
    
    args = parser.parse_args()
    
    # Import pandas here to avoid issues
    global pd
    import pandas as pd
    
    # Run analysis
    success = run_simple_analysis(
        data_file=args.data_file,
        ensemble_method=args.ensemble_method,
        skip_walkforward=args.skip_walkforward
    )
    
    if success:
        print("\n🎉 Simplified trading system completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Simplified trading system failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

