#!/usr/bin/env python3
"""
Script to clean the dataset by removing infinity values and extreme outliers
"""

import numpy as np
import pandas as pd
import sys
import os

def clean_dataset():
    """Clean the dataset by removing problematic values"""
    print("🧹 Cleaning dataset...")
    
    # Load the dataset
    df = pd.read_csv('multi_ticker_dataset_20250812_161032.csv')
    
    print(f"Original dataset shape: {df.shape}")
    
    # Find feature columns (exclude metadata)
    exclude_cols = ['Date', 'Daily_Return', 'Ticker', 'Company_Name', 'Sector', 'Industry', 'Country', 'Currency', 'Exchange', 'Website', 'Filter_Pass']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Feature columns: {len(feature_cols)}")
    
    # Clean numeric columns
    cleaned_cols = []
    removed_rows = 0
    
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            # Check for infinity and extreme values
            col_data = df[col].values
            
            # Count problematic values
            inf_count = np.sum(np.isinf(col_data))
            nan_count = np.sum(np.isnan(col_data))
            
            if inf_count > 0 or nan_count > 0:
                print(f"  Cleaning {col}: {inf_count} inf, {nan_count} NaN")
                
                # Replace infinity with NaN
                col_data = np.where(np.isinf(col_data), np.nan, col_data)
                
                # Calculate reasonable bounds (excluding NaN)
                valid_data = col_data[~np.isnan(col_data)]
                if len(valid_data) > 0:
                    q1, q99 = np.percentile(valid_data, [1, 99])
                    iqr = np.percentile(valid_data, 75) - np.percentile(valid_data, 25)
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q99 + 3 * iqr
                    
                    # Clip extreme values
                    col_data = np.clip(col_data, lower_bound, upper_bound)
                    
                    # Replace clipped values with NaN
                    col_data = np.where((col_data == lower_bound) | (col_data == upper_bound), np.nan, col_data)
                
                # Update the dataframe
                df[col] = col_data
                
                cleaned_cols.append(col)
    
    print(f"Cleaned {len(cleaned_cols)} columns")
    
    # Remove rows with too many NaN values
    feature_data = df[feature_cols].values.astype(float)
    nan_counts = np.sum(np.isnan(feature_data), axis=1)
    max_nan_per_row = len(feature_cols) * 0.5  # Allow up to 50% NaN per row
    
    valid_rows = nan_counts <= max_nan_per_row
    df_cleaned = df[valid_rows].copy()
    
    removed_rows = len(df) - len(df_cleaned)
    print(f"Removed {removed_rows} rows with too many NaN values")
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    
    # Save cleaned dataset
    output_file = 'multi_ticker_dataset_cleaned.csv'
    df_cleaned.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to: {output_file}")
    
    # Show summary statistics
    print(f"\n📊 Dataset Summary:")
    print(f"   Original samples: {len(df)}")
    print(f"   Cleaned samples: {len(df_cleaned)}")
    print(f"   Removed samples: {removed_rows}")
    print(f"   Years of data: {len(df_cleaned) / 252:.1f}")
    
    # Check for remaining issues
    feature_data_clean = df_cleaned[feature_cols].values
    inf_remaining = np.sum(np.isinf(feature_data_clean))
    nan_remaining = np.sum(np.isnan(feature_data_clean))
    
    print(f"\n🔍 Data Quality Check:")
    print(f"   Infinity values remaining: {inf_remaining}")
    print(f"   NaN values remaining: {nan_remaining}")
    
    if inf_remaining == 0:
        print("   ✅ No infinity values remaining")
    else:
        print("   ❌ Still have infinity values")
    
    # Show feature statistics
    print(f"\n📈 Feature Statistics:")
    numeric_cols = [col for col in feature_cols if df_cleaned[col].dtype in ['int64', 'float64']]
    
    for col in numeric_cols[:10]:  # Show first 10 columns
        col_data = df_cleaned[col].values
        valid_data = col_data[~np.isnan(col_data)]
        if len(valid_data) > 0:
            print(f"   {col}: mean={valid_data.mean():.6f}, std={valid_data.std():.6f}, range=[{valid_data.min():.6f}, {valid_data.max():.6f}]")
    
    if len(numeric_cols) > 10:
        print(f"   ... and {len(numeric_cols) - 10} more columns")
    
    return df_cleaned, output_file

if __name__ == "__main__":
    cleaned_df, output_file = clean_dataset()
    print(f"\n✅ Dataset cleaning completed!")
    print(f"Use '{output_file}' for your ensemble testing.")
