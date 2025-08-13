#!/usr/bin/env python3
"""
Data Management Script for Enhanced Trading Ensemble

This script provides a comprehensive interface for managing stock data:
- Full data pull for initial setup
- Daily incremental updates
- Data validation and health checks
- Status reporting
- Data cleanup and maintenance

Usage:
    python scripts/manage_data.py [command] [options]

Commands:
    full-pull      Run full data pull for all tickers
    daily-update   Run daily data update for tickers needing updates
    status         Show data status for all tickers
    validate       Validate data quality and completeness
    cleanup        Clean up old or corrupted data files
    setup-cron     Set up automated daily updates
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
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/data_management_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataManager:
    """Main class for managing stock data operations"""
    
    def __init__(self, ticker_file="TickersLiveList.txt", data_dir="data"):
        self.ticker_file = Path(ticker_file)
        self.data_dir = Path(data_dir)
        self.tickers = self._load_tickers()
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
    
    def _load_tickers(self):
        """Load tickers from the ticker file"""
        try:
            with open(self.ticker_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"Ticker file not found: {self.ticker_file}")
            return []
    
    def get_data_status(self):
        """Get comprehensive status of all ticker data"""
        status = {}
        
        for ticker in self.tickers:
            ticker_status = self._get_ticker_status(ticker)
            status[ticker] = ticker_status
        
        return status
    
    def _get_ticker_status(self, ticker):
        """Get status for a specific ticker"""
        # Look for both individual ticker files and combined dataset files
        csv_files = list(self.data_dir.glob(f"*{ticker}*.csv"))
        combined_files = list(Path(".").glob("multi_ticker_dataset*.csv"))
        
        if not csv_files and not combined_files:
            return {
                'status': 'missing',
                'files': [],
                'latest_file': None,
                'age_days': None,
                'size_mb': None,
                'rows': None
            }
        
        # Get the most recent file (either individual or combined)
        all_files = csv_files + combined_files
        latest_file = max(all_files, key=lambda x: x.stat().st_mtime)
        file_age = datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)
        
        # Get file size
        size_mb = latest_file.stat().st_size / (1024 * 1024)
        
        # Get row count and check if ticker exists in combined dataset
        try:
            df = pd.read_csv(latest_file, nrows=1)
            if 'Ticker' in df.columns:
                # Combined dataset - check if this ticker exists
                full_df = pd.read_csv(latest_file)
                ticker_rows = len(full_df[full_df['Ticker'] == ticker]) if ticker in full_df['Ticker'].values else 0
                rows = ticker_rows
            else:
                # Individual ticker file
                rows = len(pd.read_csv(latest_file))
        except Exception:
            rows = None
        
        # Determine status
        if file_age.days <= 1:
            status = 'fresh'
        elif file_age.days <= 7:
            status = 'recent'
        elif file_age.days <= 30:
            status = 'stale'
        else:
            status = 'outdated'
        
        return {
            'status': status,
            'files': [f.name for f in all_files],
            'latest_file': latest_file.name,
            'age_days': file_age.days,
            'size_mb': round(size_mb, 2),
            'rows': rows
        }
    
    def run_full_pull(self, years=15, lookforward=12, filters=True):
        """Run full data pull for all tickers"""
        logger.info("🚀 Starting full data pull...")
        
        cmd = [
            "python3", "1_stock_history_script.py",
            "--years", str(years),
            "--lookforward", str(lookforward),
            "--ticker-file", str(self.ticker_file),
            "--filters",
            "--keep-nulls"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(".").resolve()
            )
            
            if result.returncode == 0:
                logger.info("✅ Full data pull completed successfully!")
                return True
            else:
                logger.error(f"❌ Full data pull failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Exception during full data pull: {e}")
            return False
    
    def run_daily_update(self, lookforward=12, filters=True, force_all=False):
        """Run daily data update for tickers needing updates"""
        logger.info("🔄 Starting daily data update...")
        
        if not force_all:
            # Check which tickers need updates
            status = self.get_data_status()
            tickers_needing_update = [
                ticker for ticker, ticker_status in status.items()
                if ticker_status['status'] in ['stale', 'outdated', 'missing']
            ]
            
            if not tickers_needing_update:
                logger.info("✅ All tickers have fresh data, no updates needed!")
                return True
            
            logger.info(f"Found {len(tickers_needing_update)} tickers needing updates")
        
        # For daily updates, use 1 year
        years = 1
        
        cmd = [
            "python3", "1_stock_history_script.py",
            "--years", str(years),
            "--lookforward", str(lookforward),
            "--ticker-file", str(self.ticker_file),
            "--filters",
            "--keep-nulls"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(".").resolve()
            )
            
            if result.returncode == 0:
                logger.info("✅ Daily data update completed successfully!")
                return True
            else:
                logger.error(f"❌ Daily data update failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Exception during daily data update: {e}")
            return False
    
    def validate_data(self):
        """Validate data quality and completeness"""
        logger.info("🔍 Validating data quality...")
        
        status = self.get_data_status()
        validation_results = {}
        
        for ticker, ticker_status in status.items():
            validation_results[ticker] = self._validate_ticker_data(ticker, ticker_status)
        
        return validation_results
    
    def _validate_ticker_data(self, ticker, ticker_status):
        """Validate data for a specific ticker"""
        if ticker_status['status'] == 'missing':
            return {'valid': False, 'issues': ['No data files found']}
        
        latest_file = ticker_status['latest_file']
        # Check if it's a combined dataset file (in root) or individual file (in data dir)
        if latest_file.startswith('multi_ticker_dataset'):
            file_path = Path(latest_file)
        else:
            file_path = self.data_dir / latest_file
        
        issues = []
        
        try:
            # Check file size
            if ticker_status['size_mb'] < 0.1:  # Less than 100KB
                issues.append('File size too small')
            
            # Check row count
            if ticker_status['rows'] is not None and ticker_status['rows'] < 100:
                issues.append('Insufficient data rows')
            
            # Check data quality
            df = pd.read_csv(file_path)
            
            # Check if this is a combined dataset
            if 'Ticker' in df.columns:
                # Combined dataset - check if ticker exists
                if ticker not in df['Ticker'].values:
                    issues.append(f'Ticker {ticker} not found in combined dataset')
                else:
                    # Get data for this specific ticker
                    ticker_df = df[df['Ticker'] == ticker]
                    if len(ticker_df) < 100:
                        issues.append(f'Insufficient data rows for {ticker}: {len(ticker_df)}')
                    
                    # Check for required columns (basic OHLCV data)
                    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    missing_cols = [col for col in required_cols if col not in ticker_df.columns]
                    if missing_cols:
                        issues.append(f'Missing required columns: {missing_cols}')
                    
                    # Check for comprehensive feature set (should have 400+ columns)
                    if len(ticker_df.columns) < 400:
                        issues.append(f'Insufficient features: {len(ticker_df.columns)} columns (expected 400+)')
                    else:
                        logger.info(f'{ticker}: {len(ticker_df.columns)} features loaded')
                    
                    # Check for excessive nulls
                    null_ratio = ticker_df.isnull().sum().sum() / (len(ticker_df) * len(ticker_df.columns))
                    if null_ratio > 0.5:  # More than 50% nulls
                        issues.append(f'High null ratio: {null_ratio:.2%}')
                    
                    # Check date range
                    if 'Date' in ticker_df.columns:
                        try:
                            ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
                            date_range = (ticker_df['Date'].max() - ticker_df['Date'].min()).days
                            if date_range < 30:  # Less than 30 days
                                issues.append(f'Limited date range: {date_range} days')
                        except:
                            issues.append('Invalid date format')
            else:
                # Individual ticker file
                # Check for required columns (basic OHLCV data)
                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    issues.append(f'Missing required columns: {missing_cols}')
                
                # Check for comprehensive feature set (should have 400+ columns)
                if len(df.columns) < 400:
                    issues.append(f'Insufficient features: {len(df.columns)} columns (expected 400+)')
                else:
                    logger.info(f'{ticker}: {len(df.columns)} features loaded')
                
                # Check for excessive nulls
                null_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
                if null_ratio > 0.5:  # More than 50% nulls
                    issues.append(f'High null ratio: {null_ratio:.2%}')
                
                # Check date range
                if 'Date' in df.columns:
                    try:
                        df['Date'] = pd.to_datetime(df['Date'])
                        date_range = (df['Date'].max() - df['Date'].min()).days
                        if date_range < 30:  # Less than 30 days
                            issues.append(f'Limited date range: {date_range} days')
                    except:
                        issues.append('Invalid date format')
            
        except Exception as e:
            issues.append(f'Error reading file: {str(e)}')
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def cleanup_data(self, max_age_days=30, dry_run=True):
        """Clean up old or corrupted data files"""
        logger.info("🧹 Starting data cleanup...")
        
        if dry_run:
            logger.info("DRY RUN MODE - No files will be deleted")
        
        files_to_delete = []
        total_size_mb = 0
        
        for file_path in self.data_dir.glob("*.csv"):
            file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
            
            if file_age.days > max_age_days:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                files_to_delete.append((file_path, file_age.days, size_mb))
                total_size_mb += size_mb
        
        if not files_to_delete:
            logger.info("No old files found for cleanup")
            return
        
        logger.info(f"Found {len(files_to_delete)} files older than {max_age_days} days")
        logger.info(f"Total size to free: {total_size_mb:.2f} MB")
        
        for file_path, age_days, size_mb in files_to_delete:
            logger.info(f"  {file_path.name} (age: {age_days} days, size: {size_mb:.2f} MB)")
            
            if not dry_run:
                try:
                    file_path.unlink()
                    logger.info(f"  ✅ Deleted {file_path.name}")
                except Exception as e:
                    logger.error(f"  ❌ Failed to delete {file_path.name}: {e}")
        
        if dry_run:
            logger.info("Run with --no-dry-run to actually delete files")
    
    def setup_cron(self):
        """Set up automated daily updates"""
        logger.info("⏰ Setting up automated daily updates...")
        
        cron_script = """#!/bin/bash
# Daily Data Update Cron Script for Trading Ensemble
# Add this to your crontab with: crontab -e
# Example: 0 9 * * 1-5 /path/to/your/project/scripts/run_daily_update.sh

cd "$(dirname "$0")/.."
source venv/bin/activate
python scripts/manage_data.py daily-update --verbose

# Optional: Send notification on completion
# echo "Daily data update completed at $(date)" | mail -s "Data Update Status" your@email.com
"""
        
        cron_path = Path("scripts/run_daily_update.sh")
        with open(cron_path, 'w') as f:
            f.write(cron_script)
        
        # Make executable
        os.chmod(cron_path, 0o755)
        
        logger.info(f"✅ Created cron script: {cron_path}")
        logger.info("📋 To set up automatic daily updates:")
        logger.info("   1. Edit your crontab: crontab -e")
        logger.info("   2. Add this line: 0 9 * * 1-5 /path/to/your/project/scripts/run_daily_update.sh")
        logger.info("   3. Save and exit")
        logger.info("   This will run daily updates at 9 AM on weekdays")
    
    def print_status(self):
        """Print formatted status report"""
        status = self.get_data_status()
        
        print("\n" + "="*80)
        print("📊 DATA STATUS REPORT")
        print("="*80)
        print(f"{'Ticker':<8} {'Status':<10} {'Age (days)':<12} {'Size (MB)':<12} {'Rows':<8} {'Latest File'}")
        print("-"*80)
        
        total_files = 0
        total_size = 0
        total_rows = 0
        
        for ticker, ticker_status in sorted(status.items()):
            age_str = str(ticker_status['age_days']) if ticker_status['age_days'] is not None else 'N/A'
            size_str = f"{ticker_status['size_mb']:.2f}" if ticker_status['size_mb'] is not None else 'N/A'
            rows_str = str(ticker_status['rows']) if ticker_status['rows'] is not None else 'N/A'
            
            # Color code status
            status_emoji = {
                'fresh': '🟢',
                'recent': '🟡',
                'stale': '🟠',
                'outdated': '🔴',
                'missing': '⚫'
            }
            
            status_display = f"{status_emoji.get(ticker_status['status'], '❓')} {ticker_status['status']}"
            
            print(f"{ticker:<8} {status_display:<10} {age_str:<12} {size_str:<12} {rows_str:<8} {ticker_status['latest_file'] or 'N/A'}")
            
            if ticker_status['files']:
                total_files += len(ticker_status['files'])
                if ticker_status['size_mb']:
                    total_size += ticker_status['size_mb']
                if ticker_status['rows']:
                    total_rows += ticker_status['rows']
        
        print("-"*80)
        print(f"Total Tickers: {len(status)}")
        print(f"Total Files: {total_files}")
        print(f"Total Size: {total_size:.2f} MB")
        print(f"Total Rows: {total_rows:,}")
        print("="*80)

def main():
    """Main function to handle command line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Comprehensive data management for the trading ensemble system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/manage_data.py status
    python scripts/manage_data.py full-pull
    python scripts/manage_data.py daily-update
    python scripts/manage_data.py validate
    python scripts/manage_data.py cleanup --no-dry-run
    python scripts/manage_data.py setup-cron

Cron Setup:
    After running setup-cron, add to crontab:
    crontab -e
    0 9 * * 1-5 /path/to/your/project/scripts/run_daily_update.sh
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Full pull command
    full_parser = subparsers.add_parser('full-pull', help='Run full data pull for all tickers')
    full_parser.add_argument('--years', type=int, default=15, help='Years of data to fetch')
    full_parser.add_argument('--lookforward', type=int, default=12, help='Days to look forward')
    full_parser.add_argument('--no-filters', action='store_true', help='Disable technical filters')
    full_parser.add_argument('--ticker-file', type=str, help='Ticker file path')
    full_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    # Daily update command
    daily_parser = subparsers.add_parser('daily-update', help='Run daily data update')
    daily_parser.add_argument('--lookforward', type=int, default=12, help='Days to look forward')
    daily_parser.add_argument('--no-filters', action='store_true', help='Disable technical filters')
    daily_parser.add_argument('--force-all', action='store_true', help='Force update all tickers')
    daily_parser.add_argument('--ticker-file', type=str, help='Ticker file path')
    daily_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show data status for all tickers')
    status_parser.add_argument('--ticker-file', type=str, help='Ticker file path')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data quality and completeness')
    validate_parser.add_argument('--ticker-file', type=str, help='Ticker file path')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data files')
    cleanup_parser.add_argument('--max-age', type=int, default=30, help='Maximum age in days')
    cleanup_parser.add_argument('--no-dry-run', action='store_true', help='Actually delete files')
    
    # Setup cron command
    setup_cron_parser = subparsers.add_parser('setup-cron', help='Set up automated daily updates')
    setup_cron_parser.add_argument('--ticker-file', type=str, help='Ticker file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set log level based on verbose flag from subcommands
    verbose = False
    if hasattr(args, 'verbose') and args.verbose:
        verbose = True
    elif hasattr(args, 'years'):  # full-pull command
        verbose = True  # Default to verbose for full-pull
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize data manager
    ticker_file = getattr(args, 'ticker_file', None) or 'TickersLiveList.txt'
    manager = DataManager(ticker_file)
    
    if not manager.tickers:
        logger.error("No tickers loaded. Check your ticker file.")
        sys.exit(1)
    
    # Execute command
    if args.command == 'full-pull':
        success = manager.run_full_pull(
            years=args.years,
            lookforward=args.lookforward,
            filters=not args.no_filters
        )
        sys.exit(0 if success else 1)
    
    elif args.command == 'daily-update':
        success = manager.run_daily_update(
            lookforward=args.lookforward,
            filters=not args.no_filters,
            force_all=args.force_all
        )
        sys.exit(0 if success else 1)
    
    elif args.command == 'status':
        manager.print_status()
    
    elif args.command == 'validate':
        results = manager.validate_data()
        print("\n🔍 DATA VALIDATION RESULTS")
        print("="*50)
        for ticker, result in results.items():
            status = "✅ VALID" if result['valid'] else "❌ INVALID"
            print(f"{ticker}: {status}")
            if not result['valid']:
                for issue in result['issues']:
                    print(f"  - {issue}")
    
    elif args.command == 'cleanup':
        manager.cleanup_data(
            max_age_days=args.max_age,
            dry_run=not args.no_dry_run
        )
    
    elif args.command == 'setup-cron':
        manager.setup_cron()

if __name__ == "__main__":
    main()
