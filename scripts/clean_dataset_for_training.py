#!/usr/bin/env python3
"""
Clean Dataset for Training - Remove Data Leakage

This script removes target columns that cause data leakage from the multi-ticker dataset.
Target columns contain future information that models shouldn't see during training.

Features to EXCLUDE (data leakage):
- Target_Mean_Price, Target_High_Price, Target_Low_Price
- Signal, Signal_Consensus, Ensemble_Signal_Score
- Any other columns that contain future price information
"""

import pandas as pd
import argparse
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_dataset(input_file, output_file=None):
    """
    Clean dataset by removing target columns that cause data leakage
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
    
    Returns:
        str: Path to cleaned output file
    """
    
    logger.info(f"Loading dataset: {input_file}")
    df = pd.read_csv(input_file)
    
    logger.info(f"Original dataset shape: {df.shape}")
    logger.info(f"Original columns: {len(df.columns)}")
    
    # Features to EXCLUDE (they leak future information)
    exclude_features = [
        'Target_Mean_Price', 'Target_High_Price', 'Target_Low_Price',
        'Signal', 'Signal_Consensus', 'Ensemble_Signal_Score',
        'TSI_Signal', 'TRIX_Signal', 'KST_Signal'  # Additional signal columns
    ]
    
    # Features to KEEP (needed for processing)
    keep_features = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # Find columns that actually exist in the dataset
    existing_exclude_features = [col for col in exclude_features if col in df.columns]
    
    logger.info(f"Features to exclude (data leakage): {existing_exclude_features}")
    
    # Keep only legitimate features (exclude target columns but keep essential columns)
    legitimate_features = [col for col in df.columns if col not in existing_exclude_features]
    
    # Ensure essential columns are included
    for feature in keep_features:
        if feature in df.columns and feature not in legitimate_features:
            legitimate_features.append(feature)
            logger.info(f"Added back essential feature: {feature}")
    
    logger.info(f"Legitimate features remaining: {len(legitimate_features)}")
    
    # Create cleaned dataset
    df_clean = df[legitimate_features].copy()
    
    # Create a proper target variable (future returns) if we have price data
    if all(col in df_clean.columns for col in ['Close', 'Date']):
        logger.info("Creating target variable from price data...")
        
        # Sort by date and ticker
        if 'Ticker' in df_clean.columns:
            df_clean = df_clean.sort_values(['Ticker', 'Date'])
        
        # Calculate future returns (next day's return)
        df_clean['Target'] = df_clean.groupby('Ticker')['Close'].shift(-1) / df_clean['Close'] - 1
        
        # Create binary target (1 if positive return, 0 if negative)
        df_clean['Signal'] = (df_clean['Target'] > 0).astype(int)
        
        # Remove rows with NaN targets (last day of each ticker)
        df_clean = df_clean.dropna(subset=['Target', 'Signal'])
        
        logger.info(f"Target variable created. Final shape: {df_clean.shape}")
        logger.info(f"Target distribution: {df_clean['Signal'].value_counts().to_dict()}")
    
    # Generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{input_path.stem}_cleaned_{timestamp}.csv"
    
    # Save cleaned dataset
    df_clean.to_csv(output_file, index=False)
    
    logger.info(f"Cleaned dataset saved: {output_file}")
    logger.info(f"Cleaned dataset shape: {df_clean.shape}")
    
    # Show sample of remaining columns
    logger.info(f"Sample remaining columns: {df_clean.columns[:10].tolist()}")
    
    # Check for any remaining target-like columns
    remaining_target_like = [col for col in df_clean.columns if any(word in col.lower() for word in ['target', 'signal', 'consensus'])]
    if remaining_target_like:
        logger.warning(f"Remaining columns that might be targets: {remaining_target_like}")
    else:
        logger.info("✅ All target columns successfully removed")
    
    return output_file

def main():
    """Main function to handle command line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Clean dataset by removing target columns that cause data leakage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/clean_dataset_for_training.py multi_ticker_dataset_20250812_173111.csv
    python scripts/clean_dataset_for_training.py --input multi_ticker_dataset.csv --output clean_dataset.csv
        """
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Input CSV file path (default: latest multi_ticker_dataset)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine input file
    if args.input_file:
        input_file = args.input_file
    else:
        # Find the latest multi_ticker_dataset file
        dataset_files = list(Path(".").glob("multi_ticker_dataset*.csv"))
        if not dataset_files:
            logger.error("No multi_ticker_dataset files found in current directory")
            return
        
        # Get the most recent file
        input_file = max(dataset_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using latest dataset: {input_file}")
    
    # Check if input file exists
    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    # Clean the dataset
    try:
        output_file = clean_dataset(input_file, args.output)
        logger.info("🎉 Dataset cleaning completed successfully!")
        logger.info(f"📁 Cleaned file: {output_file}")
        logger.info("🚀 Ready for training without data leakage!")
        
    except Exception as e:
        logger.error(f"❌ Error cleaning dataset: {e}")
        return

if __name__ == "__main__":
    main()
