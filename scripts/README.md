# Data Management Scripts

This directory contains scripts for managing stock data for the Enhanced Trading Ensemble system.

## Scripts Overview

### 1. `manage_data.py` - Main Data Management Interface
**Primary script for all data operations**

```bash
# Show data status for all tickers
python scripts/manage_data.py status

# Run full data pull (15 years, 12 days lookforward, 450+ features)
python scripts/manage_data.py full-pull

# Run daily data update (only tickers needing updates)
python scripts/manage_data.py daily-update

# Validate data quality and feature completeness
python scripts/manage_data.py validate

# Clean up old files (dry run first)
python scripts/manage_data.py cleanup
python scripts/manage_data.py cleanup --no-dry-run

# Set up automated daily updates
python scripts/manage_data.py setup-cron
```

### 2. `full_data_pull.py` - Initial Data Collection
**Use this for the first-time setup or when you need to refresh all data**

```bash
# Default: 15 years, 12 days lookforward, with filters
python scripts/full_data_pull.py

# Custom parameters
python scripts/full_data_pull.py --years 20 --lookforward 21

# Use different ticker file
python scripts/full_data_pull.py --ticker-file custom_tickers.txt

# Disable technical filters
python scripts/full_data_pull.py --no-filters
```

### 3. `daily_data_pull.py` - Ongoing Data Updates
**Use this for daily maintenance or scheduled updates**

```bash
# Check and update tickers needing updates
python scripts/daily_data_pull.py

# Force update all tickers
python scripts/daily_data_pull.py --force-all

# Custom lookforward period
python scripts/daily_data_pull.py --lookforward 21

# Create cron script for automation
python scripts/daily_data_pull.py --create-cron
```

## Quick Start Guide

### 1. Initial Setup (Full Data Pull)
```bash
# Run the full data pull with default settings
python scripts/manage_data.py full-pull

# This will:
# - Pull 15 years of data for all tickers in TickersLiveList.txt
# - Generate 450+ technical indicators and features
# - Apply technical filters for better signal quality
# - Set lookforward to 12 days
# - Save comprehensive data to the data/ directory
```

### 2. Check Data Status
```bash
# See the current status of all ticker data
python scripts/manage_data.py status

# This shows:
# - Which tickers have fresh data
# - Which need updates
# - File sizes and row counts
# - Age of data files
```

### 3. Daily Updates
```bash
# Update only tickers that need fresh data
python scripts/manage_data.py daily-update

# This will:
# - Check which tickers need updates
# - Pull 1 year of data (sufficient for daily updates)
# - Only update tickers with stale/outdated data
```

### 4. Set Up Automation
```bash
# Create cron script for automated daily updates
python scripts/manage_data.py setup-cron

# Then add to your crontab:
crontab -e
# Add this line for 9 AM weekday updates:
0 9 * * 1-5 /path/to/your/project/scripts/run_daily_update.sh
```

## Configuration

### Ticker File
The default ticker file is `TickersLiveList.txt` in the root directory. You can specify a different file with `--ticker-file`.

### Data Directory
All data is saved to the `data/` directory. The scripts will create this directory if it doesn't exist.

### Logging
All scripts log to the `logs/` directory with timestamps. Use `--verbose` for detailed logging.

## Data Quality Features

### Comprehensive Feature Set
Your data pull generates over 450 columns including:
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, Stochastic, Williams %R, CCI, ADX
- **Advanced Moving Averages**: DEMA, HMA, KAMA, MA Envelope, MA Ribbon
- **Trend Indicators**: Alligator, Supertrend, Linear Regression
- **Volume & Money Flow**: VWAP, OBV, MFI, Volume Profile
- **Market Sentiment**: Fear/Greed, Price sentiment, Volatility sentiment
- **Calendar Features**: Day of week, month, year, holidays, earnings seasons
- **Lunar & Seasonal**: Moon phases, market seasonality, trading patterns
- **Institutional Data**: Holder counts, recommendations, earnings dates
- **Option Data**: Put/Call ratios, option volumes, expiration dates

### Technical Filters
When enabled (default), the scripts apply technical filters:
- RSI between 35-75
- MACD > 0
- ADX >= 20

### Data Validation
The validation command checks:
- File sizes and row counts
- Required columns (Date, Open, High, Low, Close, Volume)
- Feature completeness (expects 400+ columns)
- Null value ratios
- Date range completeness

### Smart Updates
The daily update script only updates tickers that need fresh data, saving time and API calls.

## Troubleshooting

### Common Issues

1. **"No tickers loaded"**
   - Check that your ticker file exists and contains valid ticker symbols
   - Ensure ticker symbols are one per line

2. **"Data pull failed"**
   - Check internet connection
   - Yahoo Finance API may be rate-limited (wait a few minutes)
   - Verify ticker symbols are valid

3. **"Permission denied"**
   - Ensure scripts are executable: `chmod +x scripts/*.py`
   - Check write permissions for data and logs directories

### Performance Tips

1. **For initial setup**: Use `full_data_pull.py` with your desired parameters
2. **For ongoing updates**: Use `daily_data_pull.py` which only updates what's needed
3. **For automation**: Use `manage_data.py setup-cron` to create scheduled updates
4. **For monitoring**: Use `manage_data.py status` regularly to check data health

## Advanced Usage

### Custom Ticker Lists
Create your own ticker file:
```bash
# Create custom ticker list
echo "AAPL
MSFT
GOOGL
TSLA" > my_tickers.txt

# Use custom ticker file
python scripts/manage_data.py full-pull --ticker-file my_tickers.txt
```

### Batch Operations
```bash
# Pull data for multiple ticker files
for file in tickers_*.txt; do
    python scripts/manage_data.py full-pull --ticker-file "$file"
done
```

### Data Cleanup
```bash
# See what would be cleaned up
python scripts/manage_data.py cleanup --max-age 60

# Actually clean up files older than 60 days
python scripts/manage_data.py cleanup --max-age 60 --no-dry-run
```

## Integration with Trading System

After running data pulls, you can:

1. **Train models**: Use the fresh data to train your ensemble models
2. **Run backtests**: Test strategies with the latest data
3. **Generate signals**: Use the lookforward data for signal generation
4. **Monitor performance**: Track model performance with current market data

## Support

For issues or questions:
1. Check the logs in the `logs/` directory
2. Run with `--verbose` for detailed output
3. Verify your ticker file format
4. Ensure all required dependencies are installed
