#!/usr/bin/env python3
"""
Daily Data Pull Script for Enhanced Trading Ensemble

This script performs daily data updates for all tickers in the system.
It's designed to be run as a cron job or scheduled task to keep data current.

Usage:
    python scripts/daily_data_pull.py [--ticker-file TICKER_FILE] [--lookforward DAYS]

Default behavior:
    - Uses TickersLiveList.txt from the root directory
    - Pulls 1 year of data (sufficient for daily updates)
    - Sets lookforward to 12 days
    - Applies technical filters
    - Only updates tickers that need new data
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pandas as pd
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/daily_data_pull_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_data_freshness(ticker, data_dir="data"):
    """
    Check if the data for a ticker is fresh (less than 1 day old)
    
    Args:
        ticker (str): Ticker symbol
        data_dir (str): Data directory path
    
    Returns:
        bool: True if data is fresh, False if needs update
    """
    data_path = Path(data_dir)
    
    # Look for CSV files for this ticker
    csv_files = list(data_path.glob(f"*{ticker}*.csv"))
    
    if not csv_files:
        logger.info(f"No data files found for {ticker}, needs initial pull")
        return False
    
    # Check the most recent file
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    file_age = datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)
    
    # If file is older than 1 day, it needs updating
    if file_age > timedelta(days=1):
        logger.info(f"Data for {ticker} is {file_age.days} days old, needs update")
        return False
    
    logger.info(f"Data for {ticker} is fresh ({file_age.days} days old)")
    return True

def get_tickers_needing_update(ticker_file="TickersLiveList.txt", data_dir="data"):
    """
    Get list of tickers that need data updates
    
    Args:
        ticker_file (str): Path to ticker file
        data_dir (str): Data directory path
    
    Returns:
        list: Tickers that need updates
    """
    # Read tickers from file
    try:
        with open(ticker_file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"Ticker file not found: {ticker_file}")
        return []
    
    # Check which tickers need updates
    tickers_needing_update = []
    for ticker in tickers:
        if not check_data_freshness(ticker, data_dir):
            tickers_needing_update.append(ticker)
    
    return tickers_needing_update

def run_daily_data_pull(ticker_file=None, lookforward=12, filters=True, force_all=False):
    """
    Run the daily data pull for tickers that need updates
    
    Args:
        ticker_file (str): Path to ticker file (default: TickersLiveList.txt)
        lookforward (int): Number of days to look forward (default: 12)
        filters (bool): Whether to apply technical filters (default: True)
        force_all (bool): Force update all tickers regardless of freshness
    """
    
    # Determine ticker file path
    if ticker_file is None:
        ticker_file = "TickersLiveList.txt"
    
    # Ensure ticker file exists
    ticker_path = Path(ticker_file)
    if not ticker_path.exists():
        logger.error(f"Ticker file not found: {ticker_file}")
        return False
    
    if force_all:
        # Read all tickers from file
        with open(ticker_path, 'r') as f:
            tickers_to_update = [line.strip() for line in f if line.strip()]
        logger.info(f"Force updating all {len(tickers_to_update)} tickers")
    else:
        # Get only tickers that need updates
        tickers_to_update = get_tickers_needing_update(ticker_file, "data")
        if not tickers_to_update:
            logger.info("✅ All tickers have fresh data, no updates needed!")
            return True
        
        logger.info(f"Found {len(tickers_to_update)} tickers needing updates: {', '.join(tickers_to_update)}")
    
    # For daily updates, we only need 1 year of data
    years = 1
    
    # Build command
    cmd = [
        "python3", "1_stock_history_script.py",
        "--years", str(years),
        "--lookforward", str(lookforward),
        "--ticker-file", str(ticker_path)
    ]
    
    if filters:
        cmd.append("--filters")
    
    # Add additional optimizations for daily updates
    cmd.extend([
        "--verbose",  # Enable verbose output
        "--keep-nulls"  # Keep nulls for ML frameworks
    ])
    
    logger.info(f"Starting daily data pull...")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Ticker file: {ticker_path}")
    logger.info(f"Years: {years}")
    logger.info(f"Lookforward: {lookforward}")
    logger.info(f"Filters: {filters}")
    logger.info(f"Tickers to update: {len(tickers_to_update)}")
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(".").resolve()
        )
        
        if result.returncode == 0:
            logger.info("✅ Daily data pull completed successfully!")
            if result.stdout:
                logger.info("Output:")
                logger.info(result.stdout)
            return True
        else:
            logger.error(f"❌ Daily data pull failed with return code {result.returncode}")
            if result.stderr:
                logger.error("Error output:")
                logger.error(result.stderr)
            if result.stdout:
                logger.info("Standard output:")
                logger.info(result.stdout)
            return False
            
    except Exception as e:
        logger.error(f"❌ Exception during daily data pull: {e}")
        return False

def create_cron_script():
    """
    Create a cron script that can be used to schedule daily data pulls
    """
    cron_script = """#!/bin/bash
# Daily Data Pull Cron Script for Trading Ensemble
# Add this to your crontab with: crontab -e
# Example: 0 9 * * 1-5 /path/to/your/project/scripts/run_daily_pull.sh

cd "$(dirname "$0")/.."
source venv/bin/activate
python scripts/daily_data_pull.py --verbose

# Optional: Send notification on completion
# echo "Daily data pull completed at $(date)" | mail -s "Data Pull Status" your@email.com
"""
    
    cron_path = Path("scripts/run_daily_pull.sh")
    with open(cron_path, 'w') as f:
        f.write(cron_script)
    
    # Make executable
    os.chmod(cron_path, 0o755)
    logger.info(f"Created cron script: {cron_path}")
    logger.info("Add to crontab with: crontab -e")
    logger.info("Example: 0 9 * * 1-5 /path/to/your/project/scripts/run_daily_pull.sh")

def main():
    """Main function to handle command line arguments and run the daily data pull"""
    
    parser = argparse.ArgumentParser(
        description='Run daily data pull for the trading ensemble system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/daily_data_pull.py
    python scripts/daily_data_pull.py --ticker-file custom_tickers.txt
    python scripts/daily_data_pull.py --lookforward 21
    python scripts/daily_data_pull.py --force-all
    python scripts/daily_data_pull.py --create-cron

Cron Setup:
    To set up automatic daily updates, use --create-cron and then add to crontab:
    crontab -e
    0 9 * * 1-5 /path/to/your/project/scripts/run_daily_pull.sh
        """
    )
    
    parser.add_argument(
        '--ticker-file',
        type=str,
        help='Path to ticker file (default: TickersLiveList.txt)'
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
        '--force-all',
        action='store_true',
        help='Force update all tickers regardless of freshness'
    )
    
    parser.add_argument(
        '--create-cron',
        action='store_true',
        help='Create a cron script for automated daily updates'
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
    
    # Handle cron script creation
    if args.create_cron:
        create_cron_script()
        return
    
    logger.info("🔄 Starting Daily Data Pull for Trading Ensemble System")
    logger.info("=" * 60)
    
    # Validate arguments
    if args.lookforward < 1 or args.lookforward > 30:
        logger.error("Lookforward days must be between 1 and 30")
        sys.exit(1)
    
    # Run the daily data pull
    success = run_daily_data_pull(
        ticker_file=args.ticker_file,
        lookforward=args.lookforward,
        filters=not args.no_filters,
        force_all=args.force_all
    )
    
    if success:
        logger.info("🎉 Daily data pull completed successfully!")
        logger.info("Next steps:")
        logger.info("  1. Check the data/ directory for updated files")
        logger.info("  2. Retrain your ensemble models if needed")
        logger.info("  3. Run backtests with fresh data")
        sys.exit(0)
    else:
        logger.error("💥 Daily data pull failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
