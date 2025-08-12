#!/usr/bin/env python3
"""
Robust data cleaning script for QQQ data
Preserves more data while fixing problematic columns
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def clean_qqq_data_robust(input_file: str, output_file: str = None):
    """
    Clean QQQ data more robustly, preserving more data
    
    Args:
        input_file: Path to input QQQ CSV file
        output_file: Path to output cleaned CSV file (optional)
    """
    print(f"🔧 Robustly cleaning QQQ data from: {input_file}")
    
    # Read the data
    print("📥 Reading data...")
    data = pd.read_csv(input_file)
    print(f"   Original shape: {data.shape}")
    
    # Keep only essential columns for trading
    print("\n🎯 Selecting essential trading columns...")
    essential_columns = [
        'Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Daily_Return', 'Price_Range', 'Price_Range_Pct', 'Typical_Price',
        'VWAP', 'RSI_14', 'MACD_12_26_9', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50',
        'BB_Upper', 'BB_Lower', 'BB_Middle', 'ATR_14', 'Volatility_20'
    ]
    
    # Check which essential columns exist
    available_essential = [col for col in essential_columns if col in data.columns]
    print(f"   Available essential columns: {len(available_essential)}")
    
    # Select columns to keep (essential + some additional useful ones)
    columns_to_keep = available_essential.copy()
    
    # Add some additional useful columns if they exist
    additional_useful = [
        'Volume_MA_20', 'Volume_MA_50', 'OBV', 'CCI_20', 'ADX_14',
        'StochRSI_14_14_3_3', 'Williams_R', 'MFI_14', 'ROC_10', 'ROC_20'
    ]
    
    for col in additional_useful:
        if col in data.columns:
            columns_to_keep.append(col)
    
    # Keep only selected columns
    data = data[columns_to_keep]
    print(f"   Selected columns: {len(columns_to_keep)}")
    
    # Clean numeric columns more carefully
    print("\n🛠️  Cleaning numeric columns...")
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_columns:
        if col == 'Date':  # Skip date column
            continue
            
        try:
            # Convert to numeric, replacing errors with NaN
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Fill NaN values more conservatively
            if col in ['Volume', 'Volume_MA_20', 'Volume_MA_50']:
                # For volume columns, fill with 0
                data[col] = data[col].fillna(0)
            elif col in ['Open', 'High', 'Low', 'Close', 'VWAP', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50']:
                # For price columns, use forward fill then backward fill
                data[col] = data[col].ffill().bfill()
            else:
                # For other columns, use forward fill only
                data[col] = data[col].ffill()
                
        except Exception as e:
            print(f"   ⚠️  Warning: Could not clean column {col}: {e}")
    
    # Handle date column
    if 'Date' in data.columns:
        print("\n📅 Processing date column...")
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        # Remove rows with invalid dates
        data = data.dropna(subset=['Date'])
        data = data.sort_values('Date').reset_index(drop=True)
        print(f"   Date column processed, valid dates: {len(data)}")
    
    # Create target variable (returns)
    print("\n🎯 Creating target variable...")
    if 'Close' in data.columns:
        data['returns'] = data['Close'].pct_change()
        # Remove first row (NaN return)
        data = data.iloc[1:].reset_index(drop=True)
        print(f"   Target variable 'returns' created")
    else:
        print("   ⚠️  Warning: 'Close' column not found, no target variable created")
    
    # Final cleanup - remove rows with too many NaN values
    print("\n🧹 Final cleanup...")
    
    # Calculate percentage of NaN values per row
    nan_percentage = data.isnull().sum(axis=1) / len(data.columns)
    
    # Keep rows where less than 50% of values are NaN
    data = data[nan_percentage < 0.5]
    
    # Fill remaining NaN values with 0 for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(0)
    
    print(f"   Final shape after cleanup: {data.shape}")
    
    # Save cleaned data
    if output_file is None:
        output_file = input_file.replace('.csv', '_cleaned_v2.csv')
    
    data.to_csv(output_file, index=False)
    print(f"💾 Cleaned data saved to: {output_file}")
    
    # Show sample of cleaned data
    print(f"\n📊 Sample of cleaned data:")
    print(data.head(3))
    
    return data

def analyze_cleaned_data(data: pd.DataFrame):
    """Analyze the cleaned data for quality"""
    print(f"\n🔍 Data Quality Analysis:")
    print(f"   Shape: {data.shape}")
    print(f"   Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for null values
    null_counts = data.isnull().sum()
    if null_counts.sum() > 0:
        print(f"   ⚠️  Null values found:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"      {col}: {count}")
    else:
        print(f"   ✅ No null values found")
    
    # Check data types
    print(f"\n📋 Data types:")
    for col, dtype in data.dtypes.items():
        print(f"   {col}: {dtype}")
    
    # Check numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    print(f"\n🔢 Numeric columns ({len(numeric_cols)}):")
    for col in numeric_cols:
        print(f"   {col}")
    
    # Check categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    print(f"\n📝 Categorical columns ({len(categorical_cols)}):")
    for col in categorical_cols:
        print(f"   {col}")
    
    # Check for infinite values
    numeric_data = data.select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_data).sum()
    if inf_counts.sum() > 0:
        print(f"\n⚠️  Infinite values found:")
        for col, count in inf_counts[inf_counts > 0].items():
            print(f"   {col}: {count}")
    else:
        print(f"\n✅ No infinite values found")

def main():
    """Main function"""
    input_file = "QQQ_20250809_155753.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        return
    
    try:
        # Clean the data
        cleaned_data = clean_qqq_data_robust(input_file)
        
        # Analyze the cleaned data
        analyze_cleaned_data(cleaned_data)
        
        print(f"\n🎉 Robust data cleaning completed successfully!")
        print(f"   You can now run the main system with the cleaned data:")
        print(f"   python3 main.py --data-file QQQ_20250809_155753_cleaned_v2.csv --ensemble-method Voting --skip-walkforward")
        
    except Exception as e:
        print(f"❌ Error during data cleaning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

