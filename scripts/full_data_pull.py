#!/usr/bin/env python3
"""
Full Data Pull Script for Enhanced Trading Ensemble

This script performs the initial full data pull for all tickers in the system.
It runs the stock history script with optimized parameters for the ensemble model.

Usage:
    python scripts/full_data_pull.py [--ticker-file TICKER_FILE] [--years YEARS] [--lookforward DAYS]

Default behavior:
    - Uses TickersLiveList.txt from the root directory
    - Pulls 15 years of data
    - Sets lookforward to 12 days
    - Applies technical filters
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/full_data_pull_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_full_data_pull(ticker_file=None, years=15, lookforward=12, filters=True):
    """
    Run the full data pull using the stock history script
    
    Args:
        ticker_file (str): Path to ticker file (default: TickersLiveList.txt)
        years (int): Number of years to fetch (default: 15)
        lookforward (int): Number of days to look forward (default: 12)
        filters (bool): Whether to apply technical filters (default: True)
    """
    
    # Determine ticker file path
    if ticker_file is None:
        ticker_file = "TickersLiveList.txt"
    
    # Ensure ticker file exists
    ticker_path = Path(ticker_file)
    if not ticker_path.exists():
        logger.error(f"Ticker file not found: {ticker_file}")
        logger.info("Available ticker files:")
        for file in Path(".").glob("*.txt"):
            if "ticker" in file.name.lower() or "live" in file.name.lower():
                logger.info(f"  - {file}")
        return False
    
    # Build command
    cmd = [
        "python3", "1_stock_history_script.py",
        "--years", str(years),
        "--lookforward", str(lookforward),
        "--ticker-file", str(ticker_path)
    ]
    
    if filters:
        cmd.append("--filters")
    
    # Add additional optimizations
    cmd.extend([
        "--verbose",  # Enable verbose output
        "--keep-nulls"  # Keep nulls for ML frameworks
    ])
    
    logger.info(f"Starting full data pull...")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Ticker file: {ticker_path}")
    logger.info(f"Years: {years}")
    logger.info(f"Lookforward: {lookforward}")
    logger.info(f"Filters: {filters}")
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(".").resolve()
        )
        
        if result.returncode == 0:
            logger.info("✅ Full data pull completed successfully!")
            if result.stdout:
                logger.info("Output:")
                logger.info(result.stdout)
            return True
        else:
            logger.error(f"❌ Full data pull failed with return code {result.returncode}")
            if result.stderr:
                logger.error("Error output:")
                logger.error(result.stderr)
            if result.stdout:
                logger.info("Standard output:")
                logger.info(result.stdout)
            return False
            
    except Exception as e:
        logger.error(f"❌ Exception during full data pull: {e}")
        return False

def main():
    """Main function to handle command line arguments and run the data pull"""
    
    parser = argparse.ArgumentParser(
        description='Run full data pull for the trading ensemble system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/full_data_pull.py
    python scripts/full_data_pull.py --ticker-file custom_tickers.txt
    python scripts/full_data_pull.py --years 20 --lookforward 21
    python scripts/full_data_pull.py --no-filters
        """
    )
    
    parser.add_argument(
        '--ticker-file',
        type=str,
        help='Path to ticker file (default: TickersLiveList.txt)'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        default=15,
        help='Number of years to fetch (default: 15)'
    )
    
    parser.add_argument(
        '--lookforward',
        type=int,
        default=12,
        help='Number of days to look forward (default: 12)'
    )
    
    parser.add_argument(
        '--no-filters',
        action='store_true',
        help='Disable technical filters (default: filters enabled)'
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
    
    logger.info("🚀 Starting Full Data Pull for Trading Ensemble System")
    logger.info("=" * 60)
    
    # Validate arguments
    if args.years < 1 or args.years > 30:
        logger.error("Years must be between 1 and 30")
        sys.exit(1)
    
    if args.lookforward < 1 or args.lookforward > 30:
        logger.error("Lookforward days must be between 1 and 30")
        sys.exit(1)
    
    # Run the data pull
    success = run_full_data_pull(
        ticker_file=args.ticker_file,
        years=args.years,
        lookforward=args.lookforward,
        filters=not args.no_filters
    )
    
    if success:
        logger.info("🎉 Full data pull completed successfully!")
        logger.info("Next steps:")
        logger.info("  1. Check the data/ directory for processed data files")
        logger.info("  2. Run daily data pull script for ongoing updates")
        logger.info("  3. Train your ensemble models")
        sys.exit(0)
    else:
        logger.error("💥 Full data pull failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
