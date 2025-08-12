#!/usr/bin/env python3
"""
Clean Multi-Ticker Dataset

This script cleans the multi-ticker dataset by:
1. Removing infinity values
2. Handling extremely large values
3. Cleaning NaN values
4. Preparing data for ensemble models
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_multi_ticker_data(csv_path, output_path):
    """Clean multi-ticker dataset"""
    logger.info(f"Loading multi-ticker CSV: {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    logger.info(f"Original data shape: {df.shape}")
    
    # Check for infinity values
    inf_mask = df.isin([np.inf, -np.inf])
    inf_count = inf_mask.sum().sum()
    logger.info(f"Found {inf_count} infinity values")
    
    # Replace infinity values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Check for extremely large values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    logger.info(f"Found {len(numeric_cols)} numeric columns")
    
    # Function to handle extreme values
    def handle_extreme_values(series, threshold=1e10):
        """Handle extremely large values by winsorizing"""
        if series.dtype in ['int64', 'float64']:
            # Replace values above threshold with 99th percentile
            upper_limit = series.quantile(0.99)
            if pd.notna(upper_limit):
                series = series.clip(upper=upper_limit)
            
            # Replace values below -threshold with 1st percentile
            lower_limit = series.quantile(0.01)
            if pd.notna(lower_limit):
                series = series.clip(lower=lower_limit)
        
        return series
    
    # Apply extreme value handling to numeric columns
    for col in numeric_cols:
        df[col] = handle_extreme_values(df[col])
    
    # Check for NaN values
    nan_count = df.isna().sum().sum()
    logger.info(f"Found {nan_count} NaN values")
    
    # Sort by Date and Ticker for proper time series handling
    if 'Date' in df.columns and 'Ticker' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date'])
        
        # Forward fill and backward fill within each ticker group
        logger.info("Applying forward/backward fill for time series data...")
        for ticker in df['Ticker'].unique():
            ticker_mask = df['Ticker'] == ticker
            df.loc[ticker_mask, numeric_cols] = df.loc[ticker_mask, numeric_cols].fillna(method='ffill').fillna(method='bfill')
    
    # Fill remaining NaN values with appropriate methods
    logger.info("Filling remaining NaN values...")
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            # For numeric columns, fill with median
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col] = df[col].fillna(median_val)
            else:
                # If median is NaN, fill with 0
                df[col] = df[col].fillna(0)
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
    
    # Verify cleaning
    final_inf_count = df.isin([np.inf, -np.inf]).sum().sum()
    final_nan_count = df.isna().sum().sum()
    
    logger.info(f"After cleaning:")
    logger.info(f"  - Infinity values: {final_inf_count}")
    logger.info(f"  - NaN values: {final_nan_count}")
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to: {output_path}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Clean multi-ticker dataset')
    parser.add_argument('--input_csv', required=True, help='Path to input multi-ticker CSV file')
    parser.add_argument('--output_csv', required=True, help='Path to output cleaned CSV file')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    
    if not input_path.exists():
        logger.error(f"Input file {input_path} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean the data
    cleaned_df = clean_multi_ticker_data(input_path, output_path)
    
    logger.info("✅ Data cleaning completed successfully!")

if __name__ == "__main__":
    main()
