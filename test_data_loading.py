#!/usr/bin/env python3
"""
Simple test script to verify data loading
"""

import pandas as pd
import numpy as np

def test_data_loading():
    """Test loading the cleaned QQQ data"""
    print("🔍 Testing data loading...")
    
    try:
        # Load the cleaned data
        data = pd.read_csv("QQQ_20250809_155753_cleaned_v2.csv")
        print(f"✅ Data loaded successfully: {data.shape}")
        
        # Check data types
        print(f"\n📋 Data types:")
        for col, dtype in data.dtypes.items():
            print(f"   {col}: {dtype}")
        
        # Check for any problematic values
        print(f"\n🔍 Checking for problematic values...")
        
        # Check Ticker column
        if 'Ticker' in data.columns:
            ticker_values = data['Ticker'].unique()
            print(f"   Ticker unique values: {ticker_values}")
            
            # Check for very long strings
            long_strings = data['Ticker'].str.len() > 10
            if long_strings.any():
                print(f"   ⚠️  Found {long_strings.sum()} rows with very long ticker values")
                print(f"   Example: {data[long_strings]['Ticker'].iloc[0]}")
            else:
                print(f"   ✅ Ticker column looks good")
        
        # Check numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        print(f"\n🔢 Numeric columns ({len(numeric_cols)}):")
        
        for col in numeric_cols[:10]:  # Show first 10
            try:
                # Check for infinite values
                inf_count = np.isinf(data[col]).sum()
                if inf_count > 0:
                    print(f"   ⚠️  {col}: {inf_count} infinite values")
                else:
                    print(f"   ✅ {col}: OK")
            except:
                print(f"   ❌ {col}: Error checking")
        
        # Check for NaN values
        print(f"\n🧹 Checking for NaN values...")
        null_counts = data.isnull().sum()
        if null_counts.sum() > 0:
            print(f"   ⚠️  Found null values:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"      {col}: {count}")
        else:
            print(f"   ✅ No null values found")
        
        # Try to create features
        print(f"\n🎯 Testing feature creation...")
        try:
            # Create some basic features
            if 'Close' in data.columns:
                data['returns'] = data['Close'].pct_change()
                data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
                data['volatility'] = data['returns'].rolling(20).std()
                print(f"   ✅ Basic features created successfully")
            else:
                print(f"   ❌ Close column not found")
        except Exception as e:
            print(f"   ❌ Error creating features: {e}")
        
        print(f"\n🎉 Data loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_loading()

