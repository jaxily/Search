#!/usr/bin/env python3
"""
Stock Data Preparation Script for Enhanced Trading Ensemble

This script helps prepare stock data by:
1. Adding technical indicators
2. Ensuring proper column format
3. Creating binary target from returns
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    logger.info("Adding technical indicators...")
    
    # Ensure we have OHLCV data
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        return df
    
    # Price-based indicators
    df['Daily_Return'] = df['Close'].pct_change()
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Range_Pct'] = df['Price_Range'] / df['Close'].shift(1)
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Volume Weighted Average Price
    df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD_12_26_9'] = exp1 - exp2
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR_14'] = true_range.rolling(14).mean()
    
    # Volatility
    df['Volatility_20'] = df['Close'].rolling(window=20).std()
    
    # Volume indicators
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_MA_50'] = df['Volume'].rolling(window=50).mean()
    
    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # CCI (Commodity Channel Index)
    typical_price = df['Typical_Price']
    sma_tp = typical_price.rolling(window=20).mean()
    mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['CCI_20'] = (typical_price - sma_tp) / (0.015 * mad)
    
    # ADX (Average Directional Index)
    df['ADX_14'] = 50  # Simplified - you can implement full ADX if needed
    
    # MFI (Money Flow Index)
    money_flow = df['Typical_Price'] * df['Volume']
    positive_flow = money_flow.where(df['Typical_Price'] > df['Typical_Price'].shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(df['Typical_Price'] < df['Typical_Price'].shift(1), 0).rolling(14).sum()
    df['MFI_14'] = 100 - (100 / (1 + positive_flow / negative_flow))
    
    # Rate of Change
    df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
    df['ROC_20'] = df['Close'].pct_change(periods=20) * 100
    
    # Target variable (returns)
    df['returns'] = df['Close'].pct_change()
    
    logger.info(f"Added {len(df.columns) - len(required_cols)} technical indicators")
    return df

def prepare_stock_data(input_path, output_path):
    """Prepare stock data for ensemble training"""
    logger.info(f"Loading data from {input_path}")
    
    # Load data
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")
    
    logger.info(f"Original data shape: {df.shape}")
    
    # Check if we need to add technical indicators
    if 'RSI_14' not in df.columns:
        logger.info("Adding technical indicators...")
        df = add_technical_indicators(df)
    else:
        logger.info("Technical indicators already present")
    
    # Ensure we have a target column
    if 'returns' not in df.columns:
        logger.warning("No 'returns' column found. Creating from Close prices...")
        df['returns'] = df['Close'].pct_change()
    
    # Remove rows with NaN values
    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)
    logger.info(f"Removed {initial_rows - final_rows} rows with NaN values")
    
    # Save prepared data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.csv':
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)
    
    logger.info(f"Prepared data saved to {output_path}")
    logger.info(f"Final data shape: {df.shape}")
    
    # Show column info
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Target distribution: {df['returns'].gt(0).mean():.3f} positive, {df['returns'].le(0).mean():.3f} negative")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Prepare stock data for ensemble training')
    parser.add_argument('--input', required=True, help='Input stock data file (CSV or Parquet)')
    parser.add_argument('--output', required=True, help='Output prepared data file')
    parser.add_argument('--format', choices=['csv', 'parquet'], default='csv', 
                       help='Output format (default: csv)')
    
    args = parser.parse_args()
    
    # Ensure output has correct extension
    output_path = args.output
    if not output_path.endswith(f'.{args.format}'):
        output_path = f"{output_path}.{args.format}"
    
    try:
        df = prepare_stock_data(args.input, output_path)
        logger.info("✅ Data preparation completed successfully!")
        
        # Show sample of prepared data
        logger.info("\nSample of prepared data:")
        logger.info(df.head().to_string())
        
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        raise

if __name__ == "__main__":
    main()
