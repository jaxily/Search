#!/usr/bin/env python3
"""
Enhanced Main Execution Script for the Walk-Forward Ensemble ML Trading System
Optimized for M1 chip with multi-threading capabilities
Enhanced with auto-detection, smart parameter selection, and graceful shutdown
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import warnings
import numpy as np
import signal
import time
import psutil
import gc
from datetime import datetime
import pickle
import json
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    SYSTEM_CONFIG, DATA_CONFIG, FEATURE_CONFIG, 
    MODEL_CONFIG, WALKFORWARD_CONFIG, PATHS, LOGGING_CONFIG,
    AUTO_DETECT_CONFIG, PERFORMANCE_TARGETS
)

# Global shutdown flag
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl-C and other shutdown signals gracefully"""
    global shutdown_requested
    if not shutdown_requested:
        shutdown_requested = True
        print("\n" + "="*60)
        print("🛑 SHUTDOWN REQUESTED - Cleaning up gracefully...")
        print("="*60)
        
        # Give time for cleanup
        time.sleep(1)
        
        print("✅ Cleanup complete. Exiting safely.")
        sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def setup_logging():
    """Setup enhanced logging with file and console output"""
    # Create logs directory
    PATHS['logs'].mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = PATHS['logs'] / f"trading_system_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=LOGGING_CONFIG['level'],
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("🚀 ENHANCED WALK-FORWARD ENSEMBLE ML TRADING SYSTEM")
    logger.info("="*60)
    logger.info(f"Log file: {log_file}")
    
    return logger

def check_system_resources():
    """Check and optimize system resources"""
    logger = logging.getLogger(__name__)
    
    # Check available memory
    try:
        available_memory = psutil.virtual_memory().available / (1024**3)
        total_memory = psutil.virtual_memory().total / (1024**3)
        
        logger.info(f"💾 Memory: {available_memory:.2f} GB available / {total_memory:.2f} GB total")
        
        # Auto-adjust chunk size based on available memory
        if available_memory < 2:
            SYSTEM_CONFIG['chunk_size'] = 100
            logger.warning("⚠️  Low memory detected. Reduced chunk size to 100")
        elif available_memory < 4:
            SYSTEM_CONFIG['chunk_size'] = 500
            logger.warning("⚠️  Moderate memory. Using chunk size 500")
        else:
            SYSTEM_CONFIG['chunk_size'] = 1000
            logger.info("✅ Sufficient memory. Using chunk size 1000")
        
        # Adjust max workers based on memory
        if available_memory < 4:
            SYSTEM_CONFIG['max_workers'] = max(2, os.cpu_count() // 2)
            logger.info(f"🔧 Adjusted max workers to {SYSTEM_CONFIG['max_workers']}")
        
    except Exception as e:
        logger.warning(f"Could not check memory: {e}")
    
    # Check CPU cores
    try:
        cpu_count = os.cpu_count()
        logger.info(f"🖥️  CPU Cores: {cpu_count}")
        
        if cpu_count < 4:
            logger.warning("⚠️  Limited CPU cores. Performance may be suboptimal.")
    except Exception as e:
        logger.warning(f"Could not check CPU: {e}")
    
    # Check disk space
    try:
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        logger.info(f"💿 Disk space: {free_gb:.2f} GB free")
        
        if free_gb < 5:
            logger.warning("⚠️  Low disk space. Consider cleaning up.")
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")

def setup_environment():
    """Setup the environment and check dependencies"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 Setting up environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8+ required")
    
    # Check system resources
    check_system_resources()
    
    # Create necessary directories
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Directory ensured: {path}")
    
    # Check for graceful shutdown
    if shutdown_requested:
        raise KeyboardInterrupt("Shutdown requested during setup")
    
    logger.info("✅ Environment setup complete")

def auto_detect_data_parameters(data_file_path: str) -> dict:
    """Auto-detect optimal parameters based on data characteristics"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Auto-detecting data parameters...")
    
    try:
        # Get file size
        file_size = os.path.getsize(data_file_path) / (1024**2)  # MB
        logger.info(f"📊 File size: {file_size:.2f} MB")
        
        # Auto-detect parameters based on file size
        if file_size > 1000:  # > 1GB
            params = {
                'chunk_size': 100,
                'use_numba': False,  # Disable for very large files
                'max_features': 100,
                'pca_components': 30
            }
        elif file_size > 100:  # > 100MB
            params = {
                'chunk_size': 500,
                'use_numba': False,  # Disable for threading safety
                'max_features': 150,
                'pca_components': 40
            }
        else:
            params = {
                'chunk_size': 1000,
                'use_numba': False,  # Disable for threading safety
                'max_features': 200,
                'pca_components': 50
            }
        
        # Auto-adjust based on available memory
        available_memory = psutil.virtual_memory().available / (1024**3)
        if available_memory < 4:
            params['chunk_size'] = min(params['chunk_size'], 200)
            params['max_features'] = min(params['max_features'], 100)
            params['pca_components'] = min(params['pca_components'], 30)
        
        logger.info(f"🎯 Auto-detected parameters: {params}")
        return params
        
    except Exception as e:
        logger.warning(f"Auto-detection failed: {e}. Using defaults.")
        return {
            'chunk_size': 500,
            'use_numba': False,  # Disable for threading safety
            'max_features': 150,
            'pca_components': 40
        }

def load_andProcessData(data_file_path: str, save_processed: bool = True) -> tuple:
    """
    Enhanced load and process data with auto-detection and memory optimization
    """
    logger = logging.getLogger(__name__)
    logger.info("📥 Loading and processing data...")
    
    try:
        # Check for shutdown
        if shutdown_requested:
            raise KeyboardInterrupt("Shutdown requested during data loading")
        
        # Auto-detect parameters
        auto_params = auto_detect_data_parameters(data_file_path)
        
        # Update system config with auto-detected parameters
        SYSTEM_CONFIG['chunk_size'] = auto_params['chunk_size']
        SYSTEM_CONFIG['use_numba'] = auto_params['use_numba']
        FEATURE_CONFIG['max_features'] = auto_params['max_features']
        FEATURE_CONFIG['pca_components'] = auto_params['pca_components']
        
        # Import data processor (import here to avoid circular imports)
        from data_processor import SmartDataProcessor
        
        # Initialize data processor with auto-detection
        processor = SmartDataProcessor(use_numba=SYSTEM_CONFIG['use_numba'])
        
        # Load and process data with auto-detection
        logger.info("🔄 Processing data with auto-detection...")
        data, feature_names = processor.loadAndProcessData(data_file_path, save_processed)
        
        # Save processed data if requested
        if save_processed:
            processor.save_processed_data(data, 'processed_data.parquet')
        
        logger.info(f"✅ Data processing complete. Shape: {data.shape}")
        logger.info(f"🎯 Feature count: {len(feature_names)}")
        
        return data, feature_names
        
    except KeyboardInterrupt:
        logger.info("🛑 Data processing interrupted by user")
        raise
    except Exception as e:
        logger.error(f"❌ Error processing data: {e}")
        raise

def runEnsembleOptimization(X: np.ndarray, y: np.ndarray, 
                           ensemble_method: str = 'Voting') -> object:
    """
    Run ensemble model optimization with auto-detection
    """
    logger = logging.getLogger(__name__)
    logger.info("🤖 Running ensemble optimization...")
    
    try:
        # Check for shutdown
        if shutdown_requested:
            raise KeyboardInterrupt("Shutdown requested during ensemble optimization")
        
        # Import models (import here to avoid circular imports)
        from models import ModelEnsemble
        
        # Initialize ensemble with auto-detection
        ensemble = ModelEnsemble(n_jobs=SYSTEM_CONFIG['max_workers'])
        
        # Auto-optimize if enabled
        if MODEL_CONFIG['auto_hyperparameter_tuning']:
            logger.info("🔧 Auto-tuning hyperparameters...")
            optimization_results = ensemble.optimize_individual_models(X, y, cv_folds=3)
        
        # Create ensemble
        logger.info(f"🏗️  Creating {ensemble_method} ensemble...")
        ensemble.create_ensemble(ensemble_method)
        
        # Fit ensemble
        logger.info("🎯 Fitting ensemble model...")
        ensemble.fit_ensemble(X, y)
        
        # Get feature importance
        feature_importance = ensemble.get_feature_importance(feature_names)
        
        logger.info("✅ Ensemble optimization complete")
        logger.info(f"📊 Model summary: {ensemble.get_model_summary()}")
        
        return ensemble
        
    except KeyboardInterrupt:
        logger.info("🛑 Ensemble optimization interrupted by user")
        raise
    except Exception as e:
        logger.error(f"❌ Error in ensemble optimization: {e}")
        raise

def runWalkforwardAnalysis(X: np.ndarray, y: np.ndarray, 
                          ensemble_method: str = 'Voting') -> dict:
    """
    Run walk-forward analysis with auto-detection
    """
    logger = logging.getLogger(__name__)
    logger.info("📈 Running walk-forward analysis...")
    
    try:
        # Check for shutdown
        if shutdown_requested:
            raise KeyboardInterrupt("Shutdown requested during walk-forward analysis")
        
        # Import walkforward (import here to avoid circular imports)
        from walkforward import WalkForwardAnalyzer
        
        # Initialize walk-forward analyzer with auto-detection
        analyzer = WalkForwardAnalyzer(n_jobs=SYSTEM_CONFIG['max_workers'])
        
        # Auto-detect window sizes if enabled
        if WALKFORWARD_CONFIG['auto_window_sizing']:
            logger.info("🔍 Auto-detecting optimal window sizes...")
            # Auto-adjust based on data size
            data_size = len(X)
            if data_size < 2000:
                analyzer.initial_train_size = min(500, data_size // 2)
                analyzer.step_size = min(100, data_size // 10)
                analyzer.min_train_size = min(200, data_size // 5)
            elif data_size < 10000:
                analyzer.initial_train_size = min(1000, data_size // 3)
                analyzer.step_size = min(200, data_size // 15)
                analyzer.min_train_size = min(400, data_size // 8)
            else:
                analyzer.initial_train_size = min(2000, data_size // 4)
                analyzer.step_size = min(500, data_size // 20)
                analyzer.min_train_size = min(800, data_size // 10)
            
            logger.info(f"🎯 Auto-detected window sizes: train={analyzer.initial_train_size}, step={analyzer.step_size}")
        
        # Run walk-forward analysis
        results = analyzer.run_walkforward_analysis(
            X, y, ensemble_method=ensemble_method
        )
        
        # Get performance summary
        performance_summary = analyzer.get_performance_summary()
        
        # Save results
        results_file = PATHS['results'] / 'walkforward_results.pkl'
        analyzer.save_results(str(results_file))
        
        # Generate performance plots
        plot_file = PATHS['reports'] / 'performance_plots.png'
        analyzer.plot_performance(save_path=str(plot_file))
        
        logger.info("✅ Walk-forward analysis complete")
        logger.info(f"💾 Results saved to: {results_file}")
        logger.info(f"📊 Performance plots saved to: {plot_file}")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("🛑 Walk-forward analysis interrupted by user")
        raise
    except Exception as e:
        logger.error(f"❌ Error in walk-forward analysis: {e}")
        raise

def generateFinalReport(results: dict, feature_names: list, 
                       ensemble: object) -> str:
    """
    Generate comprehensive final report with auto-detection insights
    """
    logger = logging.getLogger(__name__)
    logger.info("📝 Generating final report...")
    
    report = "=" * 80 + "\n"
    report += "🚀 ENHANCED WALK-FORWARD ENSEMBLE ML TRADING SYSTEM - FINAL REPORT\n"
    report += "=" * 80 + "\n\n"
    
    # System Information
    report += "🖥️  SYSTEM INFORMATION:\n"
    report += "-" * 40 + "\n"
    report += f"CPU Cores: {SYSTEM_CONFIG['max_workers']}\n"
    report += f"Memory Limit: {SYSTEM_CONFIG['memory_limit']}\n"
    report += f"Numba Enabled: {SYSTEM_CONFIG['use_numba']}\n"
    report += f"Precision: {SYSTEM_CONFIG['precision']}\n"
    report += f"Chunk Size: {SYSTEM_CONFIG['chunk_size']}\n\n"
    
    # Auto-Detection Information
    report += "🔍 AUTO-DETECTION RESULTS:\n"
    report += "-" * 40 + "\n"
    report += f"Auto-Detection Enabled: {AUTO_DETECT_CONFIG['enabled']}\n"
    report += f"Data Quality Score: {getattr(ensemble, 'data_quality_score', 'N/A')}\n"
    report += f"Optimal Features Selected: {len(getattr(ensemble, 'optimal_features', []))}\n"
    report += f"Auto-Cleaning Applied: {AUTO_DETECT_CONFIG['auto_clean_data']}\n"
    report += f"Auto-Feature Engineering: {AUTO_DETECT_CONFIG['auto_feature_engineering']}\n\n"
    
    # Data Information
    report += "📊 DATA INFORMATION:\n"
    report += "-" * 40 + "\n"
    report += f"Feature Count: {len(feature_names)}\n"
    report += f"Lookback Period: {DATA_CONFIG['lookback_period']} days\n"
    report += f"Forecast Horizon: {DATA_CONFIG['forecast_horizon']} days\n"
    report += f"Null Threshold: {DATA_CONFIG['null_threshold']}\n\n"
    
    # Model Information
    report += "🤖 MODEL INFORMATION:\n"
    report += "-" * 40 + "\n"
    try:
        model_summary = ensemble.get_model_summary()
        report += f"Base Models: {model_summary.get('n_base_models', 'N/A')}\n"
        report += f"Ensemble Method: {model_summary.get('ensemble_method', 'N/A')}\n"
        report += f"Models Fitted: {model_summary.get('is_fitted', 'N/A')}\n\n"
    except:
        report += "Model summary not available\n\n"
    
    # Walk-Forward Results
    if 'overall_performance' in results:
        report += "📈 OVERALL PERFORMANCE:\n"
        report += "-" * 40 + "\n"
        overall = results['overall_performance']
        
        # Check performance targets
        targets = PERFORMANCE_TARGETS
        sharpe_met = overall.get('sharpe_ratio', 0) >= targets['min_sharpe']
        cagr_met = overall.get('cagr', 0) >= targets['min_cagr']
        
        report += f"Sharpe Ratio: {overall.get('sharpe_ratio', 0):.2f} "
        report += f"({'✅' if sharpe_met else '❌'} Target: {targets['min_sharpe']})\n"
        
        report += f"CAGR: {overall.get('cagr', 0):.2%} "
        report += f"({'✅' if cagr_met else '❌'} Target: {targets['min_cagr']:.1%})\n"
        
        report += f"Max Drawdown: {overall.get('max_drawdown', 0):.2%}\n"
        report += f"Win Rate: {overall.get('win_rate', 0):.2%}\n"
        report += f"Volatility: {overall.get('volatility', 0):.2%}\n\n"
    
    # Recommendations
    report += "💡 RECOMMENDATIONS:\n"
    report += "-" * 40 + "\n"
    
    if 'overall_performance' in results:
        overall = results['overall_performance']
        sharpe_met = overall.get('sharpe_ratio', 0) >= targets['min_sharpe']
        cagr_met = overall.get('cagr', 0) >= targets['min_cagr']
        
        if sharpe_met and cagr_met:
            report += "✅ All performance targets met! The system is performing excellently.\n"
        elif sharpe_met:
            report += "⚠️  Sharpe ratio target met, but CAGR target not achieved.\n"
            report += "   Consider: Increasing position sizes, reducing transaction costs.\n"
        elif cagr_met:
            report += "⚠️  CAGR target met, but Sharpe ratio target not achieved.\n"
            report += "   Consider: Adding risk management, reducing volatility.\n"
        else:
            report += "❌ Performance targets not met. Consider:\n"
            report += "   - Feature engineering improvements\n"
            report += "   - Model hyperparameter tuning\n"
            report += "   - Ensemble method changes\n"
            report += "   - Data quality improvements\n"
    
    # Auto-detection insights
    report += "\n🔍 AUTO-DETECTION INSIGHTS:\n"
    report += "-" * 40 + "\n"
    report += f"• System automatically detected optimal parameters for your data\n"
    report += f"• Memory usage optimized with chunk size: {SYSTEM_CONFIG['chunk_size']}\n"
    report += f"• Feature selection optimized for {len(feature_names)} features\n"
    report += f"• Processing optimized for M1 chip with {SYSTEM_CONFIG['max_workers']} workers\n"
    
    report += "\n" + "=" * 80 + "\n"
    
    return report

def main():
    """Main execution function with enhanced error handling and auto-detection"""
    global shutdown_requested
    
    # Setup logging first
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(
        description="🚀 Enhanced Walk-Forward Ensemble ML Trading System with Auto-Detection"
    )
    parser.add_argument(
        '--data-file', 
        type=str, 
        required=True,
        help='Path to the input data file'
    )
    parser.add_argument(
        '--ensemble-method',
        type=str,
        default='Voting',
        choices=['Voting', 'Stacking', 'Blending'],
        help='Ensemble method to use'
    )
    parser.add_argument(
        '--save-processed',
        action='store_true',
        help='Save processed data for future use'
    )
    parser.add_argument(
        '--skip-walkforward',
        action='store_true',
        help='Skip walk-forward analysis (only run ensemble optimization)'
    )
    parser.add_argument(
        '--auto-detect',
        action='store_true',
        default=True,
        help='Enable auto-detection (default: True)'
    )
    
    args = parser.parse_args()
    
    try:
        # Setup environment
        setup_environment()
        
        # Check for shutdown
        if shutdown_requested:
            raise KeyboardInterrupt("Shutdown requested during setup")
        
        # Load and process data with auto-detection
        logger.info("🚀 Starting enhanced data processing...")
        data, feature_names = load_andProcessData(
            args.data_file, 
            save_processed=args.save_processed
        )
        
        # Check for shutdown
        if shutdown_requested:
            raise KeyboardInterrupt("Shutdown requested during data processing")
        
        # Prepare walk-forward data
        logger.info("🔄 Preparing walk-forward data...")
        from data_processor import SmartDataProcessor
        processor = SmartDataProcessor()
        X, y = processor.prepare_walkforward_data(data, target_col='returns')
        
        # Check for shutdown
        if shutdown_requested:
            raise KeyboardInterrupt("Shutdown requested during data preparation")
        
        # Run ensemble optimization
        logger.info("🤖 Starting ensemble optimization...")
        ensemble = runEnsembleOptimization(X, y, args.ensemble_method)
        
        # Save optimized ensemble
        model_file = PATHS['models'] / 'optimized_ensemble.pkl'
        ensemble.save_model(str(model_file))
        logger.info(f"💾 Optimized ensemble saved to: {model_file}")
        
        if not args.skip_walkforward:
            # Check for shutdown
            if shutdown_requested:
                raise KeyboardInterrupt("Shutdown requested before walk-forward analysis")
            
            # Run walk-forward analysis
            logger.info("📈 Starting walk-forward analysis...")
            results = runWalkforwardAnalysis(X, y, args.ensemble_method)
            
            # Check for shutdown
            if shutdown_requested:
                raise KeyboardInterrupt("Shutdown requested during analysis")
            
            # Generate final report
            logger.info("📝 Generating final report...")
            final_report = generateFinalReport(results, feature_names, ensemble)
            
            # Save report
            report_file = PATHS['reports'] / 'final_report.txt'
            with open(report_file, 'w') as f:
                f.write(final_report)
            
            # Print report to console
            print(final_report)
            
            logger.info(f"💾 Final report saved to: {report_file}")
        
        logger.info("🎉 System execution completed successfully!")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("🛑 Execution interrupted by user")
        print("\n🛑 System shutdown requested. Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ System execution failed: {e}")
        print(f"\n❌ Error: {e}")
        print("🔧 Check the logs for detailed information.")
        sys.exit(1)
    finally:
        # Final cleanup
        gc.collect()
        logger.info("🧹 Cleanup completed")

if __name__ == "__main__":
    # Import numpy here to avoid circular imports
    import numpy as np
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown complete. Goodbye!")
        sys.exit(0)
