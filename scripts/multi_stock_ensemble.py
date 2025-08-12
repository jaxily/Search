#!/usr/bin/env python3
"""
Multi-Stock Ensemble Runner

This script handles CSV files containing multiple stocks by:
1. Splitting into individual stock files
2. Running ensemble on each stock
3. Aggregating results
"""

import pandas as pd
import numpy as np
import argparse
import subprocess
import sys
from pathlib import Path
import logging
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_multi_stock_csv(csv_path, output_dir):
    """Split multi-stock CSV into individual stock files"""
    logger.info(f"Loading multi-stock CSV: {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Check if we have a Ticker column
    if 'Ticker' not in df.columns:
        logger.error("No 'Ticker' column found. Expected columns: Date, Ticker, Open, High, Low, Close, Volume, ...")
        return None
    
    # Get unique tickers
    tickers = df['Ticker'].unique()
    logger.info(f"Found {len(tickers)} unique tickers: {list(tickers)}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split by ticker
    stock_files = {}
    for ticker in tickers:
        ticker_data = df[df['Ticker'] == ticker].copy()
        
        # Sort by date
        if 'Date' in ticker_data.columns:
            ticker_data['Date'] = pd.to_datetime(ticker_data['Date'])
            ticker_data = ticker_data.sort_values('Date')
        
        # Save individual stock file
        output_file = output_dir / f"{ticker}_data.csv"
        ticker_data.to_csv(output_file, index=False)
        
        stock_files[ticker] = str(output_file)
        logger.info(f"Saved {ticker}: {len(ticker_data)} rows -> {output_file}")
    
    return stock_files

def run_ensemble_on_stock(stock_file, output_base_dir, args):
    """Run ensemble on a single stock file"""
    try:
        # Create output directory for this stock
        stock_name = Path(stock_file).stem.replace('_data', '')
        output_dir = Path(output_base_dir) / f"{stock_name}_ensemble"
        
        # Build command
        cmd = [
            sys.executable, "scripts/run_ensemble.py",
            "--data_path", str(stock_file),
            "--output_dir", str(output_dir),
            "--transaction_cost", str(args.transaction_cost),
            "--slippage_bps", str(args.slippage_bps),
            "--optimization_method", args.optimization_method,
            "--n_splits", str(args.n_splits)
        ]
        
        if args.regime_aware:
            cmd.append("--regime_aware")
        
        logger.info(f"Running ensemble on {stock_name}...")
        start_time = time.time()
        
        # Run the ensemble
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ {stock_name} completed in {elapsed_time:.1f}s")
        
        return {
            'stock': stock_name,
            'status': 'success',
            'output_dir': str(output_dir),
            'elapsed_time': elapsed_time
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {stock_name} failed: {e}")
        return {
            'stock': stock_name,
            'status': 'failed',
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr
        }
    except Exception as e:
        logger.error(f"❌ {stock_name} failed with exception: {e}")
        return {
            'stock': stock_name,
            'status': 'failed',
            'error': str(e)
        }

def aggregate_results(results, output_dir):
    """Aggregate results from all stocks"""
    logger.info("Aggregating results...")
    
    # Collect summary data
    summary_data = []
    
    for result in results:
        if result['status'] == 'success':
            try:
                # Load metrics for this stock
                metrics_file = Path(result['output_dir']) / "metrics.json"
                if metrics_file.exists():
                    import json
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    # Extract key metrics
                    summary_data.append({
                        'Stock': result['stock'],
                        'Status': 'Success',
                        'Processing_Time_s': result['elapsed_time'],
                        'Optimal_Threshold': metrics.get('optimal_threshold', 'N/A'),
                        'Sharpe_Ratio': metrics.get('sharpe_ratio', 'N/A'),
                        'Annualized_Return': metrics.get('annualized_return', 'N/A'),
                        'Max_Drawdown': metrics.get('max_drawdown', 'N/A'),
                        'Trade_Count': metrics.get('trade_count', 'N/A')
                    })
                else:
                    summary_data.append({
                        'Stock': result['stock'],
                        'Status': 'Success',
                        'Processing_Time_s': result['elapsed_time'],
                        'Optimal_Threshold': 'N/A',
                        'Sharpe_Ratio': 'N/A',
                        'Annualized_Return': 'N/A',
                        'Max_Drawdown': 'N/A',
                        'Trade_Count': 'N/A'
                    })
            except Exception as e:
                logger.warning(f"Could not load metrics for {result['stock']}: {e}")
                summary_data.append({
                    'Stock': result['stock'],
                    'Status': 'Success',
                    'Processing_Time_s': result['elapsed_time'],
                    'Optimal_Threshold': 'N/A',
                    'Sharpe_Ratio': 'N/A',
                    'Annualized_Return': 'N/A',
                    'Max_Drawdown': 'N/A',
                    'Trade_Count': 'N/A'
                })
        else:
            summary_data.append({
                'Stock': result['stock'],
                'Status': 'Failed',
                'Processing_Time_s': 'N/A',
                'Optimal_Threshold': 'N/A',
                'Sharpe_Ratio': 'N/A',
                'Annualized_Return': 'N/A',
                'Max_Drawdown': 'N/A',
                'Trade_Count': 'N/A'
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_file = Path(output_dir) / "multi_stock_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    logger.info(f"Summary saved to: {summary_file}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("📊 MULTI-STOCK ENSEMBLE SUMMARY")
    logger.info("="*80)
    logger.info(summary_df.to_string(index=False))
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description='Run ensemble on multi-stock CSV file')
    parser.add_argument('--csv_path', required=True, help='Path to multi-stock CSV file')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--transaction_cost', type=float, default=0.001, help='Transaction cost per trade')
    parser.add_argument('--slippage_bps', type=float, default=0.5, help='Slippage in basis points')
    parser.add_argument('--optimization_method', choices=['sharpe', 'cagr', 'sharpe_cagr'], default='sharpe')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of TimeSeriesSplit folds')
    parser.add_argument('--regime_aware', action='store_true', help='Enable regime-aware weights')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum parallel workers')
    parser.add_argument('--keep_temp', action='store_true', help='Keep temporary split files')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    
    if not csv_path.exists():
        logger.error(f"CSV file {csv_path} does not exist")
        return
    
    # Create temporary directory for split files
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary directory: {temp_dir}")
        
        # Step 1: Split multi-stock CSV
        stock_files = split_multi_stock_csv(csv_path, temp_dir)
        if not stock_files:
            logger.error("Failed to split CSV file")
            return
        
        # Step 2: Run ensemble on each stock
        results = []
        
        if args.max_workers > 1:
            logger.info(f"Processing {len(stock_files)} stocks with {args.max_workers} parallel workers...")
            
            with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                # Submit all jobs
                future_to_stock = {
                    executor.submit(run_ensemble_on_stock, stock_file, output_dir, args): stock_file
                    for stock_file in stock_files.values()
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_stock):
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'success':
                        logger.info(f"✅ {result['stock']}: {result['output_dir']}")
                    else:
                        logger.error(f"❌ {result['stock']}: {result.get('error', 'Unknown error')}")
        else:
            logger.info(f"Processing {len(stock_files)} stocks sequentially...")
            
            for stock_file in stock_files.values():
                result = run_ensemble_on_stock(stock_file, output_dir, args)
                results.append(result)
        
        # Step 3: Aggregate results
        summary_df = aggregate_results(results, output_dir)
        
        # Step 4: Clean up or keep temp files
        if args.keep_temp:
            temp_output = output_dir / "temp_split_files"
            shutil.copytree(temp_dir, temp_output)
            logger.info(f"Temporary files saved to: {temp_output}")
    
    logger.info("✅ Multi-stock ensemble processing completed!")

if __name__ == "__main__":
    main()
