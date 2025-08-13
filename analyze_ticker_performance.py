#!/usr/bin/env python3
"""
Analyze Ticker Performance

This script analyzes the actual historical performance of each ticker
to calculate real Sharpe ratios and CAGR values.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_performance_metrics(returns_series, risk_free_rate=0.02):
    """Calculate performance metrics for a series of returns"""
    if len(returns_series) < 2:
        return None
    
    # Remove NaN values
    returns = returns_series.dropna()
    if len(returns) < 2:
        return None
    
    # Calculate metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Calculate max drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    win_rate = (returns > 0).mean()
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'data_points': len(returns)
    }

def analyze_ticker_performance(csv_path, output_path):
    """Analyze performance of all tickers"""
    logger.info(f"Loading multi-ticker CSV: {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get unique tickers
    tickers = df['Ticker'].unique()
    logger.info(f"Found {len(tickers)} unique tickers")
    
    # Analyze each ticker
    results = []
    
    for ticker in tickers:
        ticker_data = df[df['Ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date')
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(ticker_data['Daily_Return'])
        
        if metrics:
            results.append({
                'Ticker': ticker,
                'Data_Points': metrics['data_points'],
                'Total_Return': f"{metrics['total_return']:.4f}",
                'Annualized_Return': f"{metrics['annualized_return']:.4f}",
                'Volatility': f"{metrics['volatility']:.4f}",
                'Sharpe_Ratio': f"{metrics['sharpe_ratio']:.4f}",
                'Max_Drawdown': f"{metrics['max_drawdown']:.4f}",
                'Win_Rate': f"{metrics['win_rate']:.4f}"
            })
            
            logger.info(f"✅ {ticker}: Sharpe={metrics['sharpe_ratio']:.4f}, CAGR={metrics['annualized_return']:.4f}")
        else:
            logger.warning(f"⚠️  {ticker}: Insufficient data")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by Sharpe ratio (descending)
    results_df['Sharpe_Ratio_Num'] = pd.to_numeric(results_df['Sharpe_Ratio'])
    results_df = results_df.sort_values('Sharpe_Ratio_Num', ascending=False)
    results_df = results_df.drop('Sharpe_Ratio_Num', axis=1)
    
    # Save results
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to: {output_path}")
    
    # Print top performers
    logger.info("\n" + "="*80)
    logger.info("🏆 TOP PERFORMERS BY SHARPE RATIO")
    logger.info("="*80)
    logger.info(results_df.head(10).to_string(index=False))
    
    logger.info("\n" + "="*80)
    logger.info("📊 TOP PERFORMERS BY CAGR")
    logger.info("="*80)
    
    # Sort by CAGR
    results_df['CAGR_Num'] = pd.to_numeric(results_df['Annualized_Return'])
    top_cagr = results_df.sort_values('CAGR_Num', ascending=False).head(10)
    top_cagr = top_cagr.drop('CAGR_Num', axis=1)
    logger.info(top_cagr.to_string(index=False))
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Analyze ticker performance')
    parser.add_argument('--input_csv', required=True, help='Path to input multi-ticker CSV file')
    parser.add_argument('--output_csv', required=True, help='Path to output performance CSV file')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    
    if not input_path.exists():
        logger.error(f"Input file {input_path} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Analyze performance
    results_df = analyze_ticker_performance(input_path, output_path)
    
    logger.info("✅ Performance analysis completed successfully!")

if __name__ == "__main__":
    main()
