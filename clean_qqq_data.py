#!/usr/bin/env python3
"""
Data cleaning script for QQQ data
Fixes issues with problematic columns before running the main system
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def clean_qqq_data(input_file: str, output_file: str = None):
    """
    Clean QQQ data by handling problematic columns
    
    Args:
        input_file: Path to input QQQ CSV file
        output_file: Path to output cleaned CSV file (optional)
    """
    print(f"🔧 Cleaning QQQ data from: {input_file}")
    
    # Read the data
    print("📥 Reading data...")
    data = pd.read_csv(input_file)
    print(f"   Original shape: {data.shape}")
    
    # Identify problematic columns
    print("\n🔍 Analyzing columns...")
    
    # Columns that should be numeric but might have issues
    numeric_columns = []
    categorical_columns = []
    problematic_columns = []
    
    for col in data.columns:
        # Skip date and ticker columns
        if col in ['Date', 'Ticker']:
            continue
            
        # Check if column contains mostly numeric data
        try:
            # Try to convert to numeric
            pd.to_numeric(data[col], errors='coerce')
            non_null_count = pd.to_numeric(data[col], errors='coerce').notna().sum()
            
            if non_null_count > len(data) * 0.5:  # More than 50% numeric
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)
                
        except:
            # Column can't be converted to numeric
            categorical_columns.append(col)
    
    print(f"   Numeric columns: {len(numeric_columns)}")
    print(f"   Categorical columns: {len(categorical_columns)}")
    
    # Handle problematic numeric columns
    print("\n🛠️  Cleaning numeric columns...")
    for col in numeric_columns:
        try:
            # Convert to numeric, replacing errors with NaN
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Fill NaN values with appropriate defaults
            if col in ['Volume', 'Dividends', 'Stock Splits']:
                data[col] = data[col].fillna(0)
            else:
                # For other numeric columns, use forward fill then backward fill
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
                
        except Exception as e:
            print(f"   ⚠️  Warning: Could not clean column {col}: {e}")
            problematic_columns.append(col)
    
    # Handle categorical columns
    print("\n🛠️  Cleaning categorical columns...")
    for col in categorical_columns:
        try:
            # Replace empty strings with NaN
            data[col] = data[col].replace('', np.nan)
            
            # Fill NaN values with appropriate defaults
            if col in ['Company_Name', 'Sector', 'Industry', 'Country']:
                data[col] = data[col].fillna('Unknown')
            elif col in ['Currency', 'Exchange']:
                data[col] = data[col].fillna('USD')
            else:
                # For other categorical columns, use forward fill
                data[col] = data[col].fillna(method='ffill')
                
        except Exception as e:
            print(f"   ⚠️  Warning: Could not clean column {col}: {e}")
            problematic_columns.append(col)
    
    # Remove problematic columns if they exist
    if problematic_columns:
        print(f"\n🗑️  Removing problematic columns: {problematic_columns}")
        data = data.drop(columns=problematic_columns)
    
    # Create target variable (returns)
    print("\n🎯 Creating target variable...")
    if 'Close' in data.columns:
        data['returns'] = data['Close'].pct_change()
        # Remove first row (NaN return)
        data = data.iloc[1:].reset_index(drop=True)
        print(f"   Target variable 'returns' created")
    else:
        print("   ⚠️  Warning: 'Close' column not found, no target variable created")
    
    # Final cleanup
    print("\n🧹 Final cleanup...")
    
    # Remove any remaining rows with NaN values
    initial_rows = len(data)
    data = data.dropna()
    final_rows = len(data)
    
    if initial_rows != final_rows:
        print(f"   Removed {initial_rows - final_rows} rows with NaN values")
    
    # Ensure date column is properly formatted
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)
        print(f"   Date column formatted and sorted")
    
    print(f"\n✅ Data cleaning complete!")
    print(f"   Final shape: {data.shape}")
    print(f"   Columns: {len(data.columns)}")
    
    # Save cleaned data
    if output_file is None:
        output_file = input_file.replace('.csv', '_cleaned.csv')
    
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
    for col in numeric_cols[:10]:  # Show first 10
        print(f"   {col}")
    if len(numeric_cols) > 10:
        print(f"   ... and {len(numeric_cols) - 10} more")
    
    # Check categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    print(f"\n📝 Categorical columns ({len(categorical_cols)}):")
    for col in categorical_cols:
        print(f"   {col}")

def main():
    """Main function"""
    input_file = "QQQ_20250809_155753.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        return
    
    try:
        # Clean the data
        cleaned_data = clean_qqq_data(input_file)
        
        # Analyze the cleaned data
        analyze_cleaned_data(cleaned_data)
        
        print(f"\n🎉 Data cleaning completed successfully!")
        print(f"   You can now run the main system with the cleaned data:")
        print(f"   python3 main.py --data-file QQQ_20250809_155753_cleaned.csv --ensemble-method Voting --skip-walkforward")
        
    except Exception as e:
        print(f"❌ Error during data cleaning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
