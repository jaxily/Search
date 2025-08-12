#!/usr/bin/env python3
"""
Batch Ensemble Runner for Multiple Stocks

This script runs the enhanced ensemble on multiple stock files
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_ensemble_on_stock(stock_file, output_base_dir, args):
    """Run ensemble on a single stock file"""
    try:
        # Create output directory for this stock
        stock_name = Path(stock_file).stem
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

def main():
    parser = argparse.ArgumentParser(description='Run ensemble on multiple stock files')
    parser.add_argument('--data_dir', required=True, help='Directory containing stock data files')
    parser.add_argument('--output_dir', required=True, help='Base output directory for results')
    parser.add_argument('--file_pattern', default='*.csv', help='File pattern to match (default: *.csv)')
    parser.add_argument('--transaction_cost', type=float, default=0.001, help='Transaction cost per trade')
    parser.add_argument('--slippage_bps', type=float, default=0.5, help='Slippage in basis points')
    parser.add_argument('--optimization_method', choices=['sharpe', 'cagr', 'sharpe_cagr'], default='sharpe')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of TimeSeriesSplit folds')
    parser.add_argument('--regime_aware', action='store_true', help='Enable regime-aware weights')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum parallel workers')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        return
    
    # Find stock files
    stock_files = list(data_dir.glob(args.file_pattern))
    if not stock_files:
        logger.error(f"No files found matching pattern {args.file_pattern} in {data_dir}")
        return
    
    logger.info(f"Found {len(stock_files)} stock files to process")
    for file in stock_files:
        logger.info(f"  - {file.name}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process stocks
    results = []
    
    if args.max_workers > 1:
        logger.info(f"Processing {len(stock_files)} stocks with {args.max_workers} parallel workers...")
        
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all jobs
            future_to_stock = {
                executor.submit(run_ensemble_on_stock, stock_file, output_dir, args): stock_file
                for stock_file in stock_files
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
        
        for stock_file in stock_files:
            result = run_ensemble_on_stock(stock_file, output_dir, args)
            results.append(result)
    
    # Summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    logger.info("="*60)
    logger.info("📊 BATCH PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total stocks: {len(stock_files)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    if successful:
        avg_time = sum(r['elapsed_time'] for r in successful) / len(successful)
        logger.info(f"Average processing time: {avg_time:.1f}s")
        
        logger.info("\nSuccessful stocks:")
        for result in successful:
            logger.info(f"  ✅ {result['stock']}: {result['output_dir']}")
    
    if failed:
        logger.info("\nFailed stocks:")
        for result in failed:
            logger.info(f"  ❌ {result['stock']}: {result.get('error', 'Unknown error')}")
    
    logger.info(f"\nAll results saved to: {output_dir}")

if __name__ == "__main__":
    main()
