"""
Enhanced Data Processor for the Walk-Forward Ensemble ML Trading System
Optimized for M1 chip with multi-threading capabilities
Enhanced with auto-detection and smart data processing
"""

import numpy as np
import pandas as pd
import numba as nb
from numba import jit, prange
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import Tuple, List, Optional, Dict, Any
import psutil
import gc
import os
import signal
import sys
from pathlib import Path
import pickle
import json
from datetime import datetime
import time

from config import (
    SYSTEM_CONFIG, DATA_CONFIG, FEATURE_CONFIG, 
    PATHS, LOGGING_CONFIG, AUTO_DETECT_CONFIG
)

# Configure logging
logging.basicConfig(
    level=LOGGING_CONFIG['level'],
    format=LOGGING_CONFIG['format']
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl-C gracefully"""
    global shutdown_requested
    logger.info("Shutdown signal received. Cleaning up...")
    shutdown_requested = True
    sys.exit(0)

# Register signal handlers
if SYSTEM_CONFIG['graceful_shutdown']:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# Numba-optimized functions at module level
@jit(nopython=True, parallel=True, cache=True)
def _numba_rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Numba-optimized rolling mean calculation"""
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in prange(window - 1, n):
        result[i] = np.mean(data[i-window+1:i+1])
    
    return result

@jit(nopython=True, parallel=True, cache=True)
def _numba_rolling_std(data: np.ndarray, window: int) -> np.ndarray:
    """Numba-optimized rolling standard deviation"""
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in prange(window - 1, n):
        result[i] = np.std(data[i-window+1:i+1])
    
    return result

class SmartDataProcessor:
    """
    Enhanced data processor with auto-detection and smart processing
    Optimized for M1 chip with multi-threading
    """
    
    def __init__(self, use_numba: bool = True):
        self.use_numba = use_numba and SYSTEM_CONFIG['use_numba']
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.pca = None
        self.feature_names = []
        self.null_columns = []
        self.data_quality_score = 0.0
        self.optimal_splits = {}
        self.optimal_features = []
        self.processing_history = []
        
        # M1 optimization settings
        self.chunk_size = SYSTEM_CONFIG['chunk_size']
        self.max_workers = min(SYSTEM_CONFIG['max_workers'], mp.cpu_count())
        
        # Auto-detection settings
        self.auto_detect = AUTO_DETECT_CONFIG['enabled']
        self.auto_clean = AUTO_DETECT_CONFIG['auto_clean_data']
        self.auto_feature_engineering = AUTO_DETECT_CONFIG['auto_feature_engineering']
        
        logger.info(f"SmartDataProcessor initialized with {self.max_workers} workers")
        logger.info(f"Auto-detection enabled: {self.auto_detect}")
    
    def _check_shutdown(self):
        """Check if shutdown was requested"""
        if shutdown_requested:
            logger.info("Shutdown requested, stopping processing...")
            raise KeyboardInterrupt("Shutdown requested")
    
    def _auto_save_state(self, stage: str, data: Any = None):
        """Auto-save processing state"""
        if not SYSTEM_CONFIG['auto_save_interval']:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_file = PATHS['temp'] / f"processor_state_{stage}_{timestamp}.pkl"
        
        try:
            state = {
                'stage': stage,
                'timestamp': timestamp,
                'feature_names': self.feature_names,
                'data_quality_score': self.data_quality_score,
                'optimal_splits': self.optimal_splits,
                'optimal_features': self.optimal_features,
                'processing_history': self.processing_history
            }
            
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
            
            logger.debug(f"State saved: {state_file}")
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess overall data quality score"""
        logger.info("Assessing data quality...")
        
        quality_metrics = {}
        
        # Check for null values
        null_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        quality_metrics['null_ratio'] = 1 - null_ratio
        
        # Check for infinite values
        inf_ratio = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        inf_ratio = inf_ratio / (data.shape[0] * data.shape[1])
        quality_metrics['inf_ratio'] = 1 - inf_ratio
        
        # Check data consistency
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            # Check for constant columns
            constant_cols = (numeric_data.std() == 0).sum()
            constant_ratio = constant_cols / len(numeric_data.columns)
            quality_metrics['constant_ratio'] = 1 - constant_ratio
            
            # Check for duplicate rows
            duplicate_ratio = data.duplicated().sum() / len(data)
            quality_metrics['duplicate_ratio'] = 1 - duplicate_ratio
        else:
            quality_metrics['constant_ratio'] = 1.0
            quality_metrics['duplicate_ratio'] = 1.0
        
        # Calculate overall quality score
        overall_score = np.mean(list(quality_metrics.values()))
        
        logger.info(f"Data quality metrics: {quality_metrics}")
        logger.info(f"Overall quality score: {overall_score:.3f}")
        
        return overall_score
    
    def _auto_detect_splits(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Auto-detect optimal train/test splits based on data characteristics"""
        logger.info("Auto-detecting optimal splits...")
        
        n_samples = len(data)
        
        # Detect seasonality patterns
        if 'date' in data.columns or data.index.dtype == 'datetime64[ns]':
            # Time series data - use time-based splits
            splits = self._detect_time_based_splits(data, n_samples)
        else:
            # Non-time series data - use statistical splits
            splits = self._detect_statistical_splits(data, n_samples)
        
        # Validate splits
        validated_splits = self._validate_splits(splits, n_samples)
        
        logger.info(f"Optimal splits detected: {validated_splits}")
        return validated_splits
    
    def _detect_time_based_splits(self, data: pd.DataFrame, n_samples: int) -> Dict[str, Any]:
        """Detect optimal splits for time series data"""
        # Try different split ratios
        split_ratios = [0.7, 0.75, 0.8, 0.85]
        best_split = None
        best_score = -np.inf
        
        for ratio in split_ratios:
            train_size = int(n_samples * ratio)
            test_size = n_samples - train_size
            
            if test_size < DATA_CONFIG['min_data_points']:
                continue
            
            # Use time series cross-validation to score this split
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(data.iloc[:train_size]):
                train_data = data.iloc[train_idx]
                val_data = data.iloc[val_idx]
                
                # Simple scoring based on data distribution similarity
                train_mean = train_data.mean()
                val_mean = val_data.mean()
                
                # Calculate distribution similarity
                similarity = 1 / (1 + np.mean((train_mean - val_mean) ** 2))
                scores.append(similarity)
            
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_split = {
                    'train_ratio': ratio,
                    'train_size': train_size,
                    'test_size': test_size,
                    'split_type': 'time_based',
                    'score': best_score
                }
        
        return best_split or {
            'train_ratio': 0.8,
            'train_size': int(n_samples * 0.8),
            'test_size': n_samples - int(n_samples * 0.8),
            'split_type': 'time_based',
            'score': 0.0
        }
    
    def _detect_statistical_splits(self, data: pd.DataFrame, n_samples: int) -> Dict[str, Any]:
        """Detect optimal splits for non-time series data"""
        # Use stratified sampling if possible
        if 'target' in data.columns or 'returns' in data.columns:
            target_col = 'target' if 'target' in data.columns else 'returns'
            
            # Create bins for stratification
            target_bins = pd.qcut(data[target_col], q=5, duplicates='drop')
            
            # Calculate optimal split ratio
            optimal_ratio = 0.8  # Default
            best_score = 0.0
            
            for ratio in [0.7, 0.75, 0.8, 0.85]:
                train_size = int(n_samples * ratio)
                test_size = n_samples - train_size
                
                if test_size < DATA_CONFIG['min_data_points']:
                    continue
                
                # Check if we can maintain class balance
                train_data = data.iloc[:train_size]
                test_data = data.iloc[train_size:]
                
                train_dist = train_data[target_col].value_counts(normalize=True)
                test_dist = test_data[target_col].value_counts(normalize=True)
                
                # Calculate distribution similarity
                common_classes = set(train_dist.index) & set(test_dist.index)
                if len(common_classes) > 0:
                    similarity = 1 - np.mean([
                        abs(train_dist.get(cls, 0) - test_dist.get(cls, 0))
                        for cls in common_classes
                    ])
                    
                    if similarity > best_score:
                        best_score = similarity
                        optimal_ratio = ratio
            
            return {
                'train_ratio': optimal_ratio,
                'train_size': int(n_samples * optimal_ratio),
                'test_size': n_samples - int(n_samples * optimal_ratio),
                'split_type': 'stratified',
                'score': best_score
            }
        else:
            # Simple random split
            return {
                'train_ratio': 0.8,
                'train_size': int(n_samples * 0.8),
                'test_size': n_samples - int(n_samples * 0.8),
                'split_type': 'random',
                'score': 0.0
            }
    
    def _validate_splits(self, splits: Dict[str, Any], n_samples: int) -> Dict[str, Any]:
        """Validate and adjust splits if necessary"""
        # Ensure minimum sizes
        min_train = max(DATA_CONFIG['min_data_points'], int(n_samples * DATA_CONFIG['min_split_size']))
        min_test = DATA_CONFIG['min_data_points']
        
        if splits['train_size'] < min_train:
            splits['train_size'] = min_train
            splits['test_size'] = n_samples - min_train
            splits['train_ratio'] = min_train / n_samples
        
        if splits['test_size'] < min_test:
            splits['test_size'] = min_test
            splits['train_size'] = n_samples - min_test
            splits['train_ratio'] = splits['train_size'] / n_samples
        
        # Ensure maximum sizes
        max_train = int(n_samples * DATA_CONFIG['max_split_size'])
        if splits['train_size'] > max_train:
            splits['train_size'] = max_train
            splits['test_size'] = n_samples - max_train
            splits['train_ratio'] = max_train / n_samples
        
        return splits
    
    def _auto_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Automatically clean data based on quality assessment"""
        logger.info("Auto-cleaning data...")
        
        initial_shape = data.shape
        cleaning_steps = []
        
        # Step 1: Handle null values
        if data.isnull().sum().sum() > 0:
            data = self._smart_null_handling(data)
            cleaning_steps.append("null_handling")
        
        # Step 2: Handle infinite values
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            inf_mask = np.isinf(numeric_data)
            if inf_mask.sum().sum() > 0:
                data = self._handle_infinite_values(data)
                cleaning_steps.append("infinity_handling")
        
        # Step 3: Remove constant columns
        constant_cols = data.columns[data.nunique() == 1]
        if len(constant_cols) > 0:
            data = data.drop(columns=constant_cols)
            cleaning_steps.append(f"removed_{len(constant_cols)}_constant_columns")
        
        # Step 4: Remove duplicate rows
        initial_rows = len(data)
        data = data.drop_duplicates()
        final_rows = len(data)
        if final_rows < initial_rows:
            cleaning_steps.append(f"removed_{initial_rows - final_rows}_duplicate_rows")
        
        # Step 5: Optimize data types
        data = self._optimize_dtypes(data)
        cleaning_steps.append("dtype_optimization")
        
        # Step 6: Sort index if it's time-based
        if data.index.dtype == 'datetime64[ns]':
            data = data.sort_index()
            cleaning_steps.append("index_sorting")
        
        final_shape = data.shape
        logger.info(f"Auto-cleaning complete. Shape: {initial_shape} -> {final_shape}")
        logger.info(f"Cleaning steps: {cleaning_steps}")
        
        return data
    
    def _smart_null_handling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Smart handling of null values based on column characteristics"""
        for col in data.columns:
            if data[col].isnull().sum() > 0:
                null_ratio = data[col].isnull().sum() / len(data)
                
                if null_ratio > DATA_CONFIG['null_threshold']:
                    # Too many nulls - drop column
                    data = data.drop(columns=[col])
                    logger.debug(f"Dropped column {col} due to high null ratio: {null_ratio:.3f}")
                else:
                    # Handle nulls based on column type
                    if data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        # Numeric column - use interpolation
                        if data[col].isnull().sum() < len(data) * 0.1:
                            # Few nulls - use forward fill then backward fill
                            data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
                        else:
                            # Many nulls - use interpolation
                            data[col] = data[col].interpolate(method='linear')
                    else:
                        # Categorical column - use mode
                        mode_value = data[col].mode()
                        if len(mode_value) > 0:
                            data[col] = data[col].fillna(mode_value[0])
                        else:
                            data[col] = data[col].fillna('Unknown')
        
        return data
    
    def _handle_infinite_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite values in numeric columns"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            inf_mask = np.isinf(data[col])
            if inf_mask.sum() > 0:
                # Replace infinities with column statistics
                col_data = data[col][~inf_mask]
                if len(col_data) > 0:
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    
                    # Replace positive infinities with mean + 3*std
                    data.loc[data[col] == np.inf, col] = mean_val + 3 * std_val
                    # Replace negative infinities with mean - 3*std
                    data.loc[data[col] == -np.inf, col] = mean_val - 3 * std_val
        
        return data
    
    def _auto_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Automatically engineer features based on data characteristics"""
        logger.info("Auto-engineering features...")
        
        if not self.auto_feature_engineering:
            return data
        
        # Analyze data to determine optimal features
        feature_plan = self._analyze_feature_needs(data)
        
        # Create features based on plan
        data = self._create_features_by_plan(data, feature_plan)
        
        # Auto-select optimal features
        data = self._auto_feature_selection(data)
        
        return data
    
    def _analyze_feature_needs(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data to determine what features to create"""
        feature_plan = {
            'price_features': False,
            'technical_indicators': False,
            'rolling_features': False,
            'volatility_features': False,
            'correlation_features': False,
            'custom_features': False
        }
        
        # Check if we have price data
        price_keywords = ['open', 'high', 'low', 'close', 'price', 'adj']
        has_price_data = any(any(keyword in col.lower() for keyword in price_keywords) 
                           for col in data.columns)
        
        if has_price_data:
            feature_plan['price_features'] = True
            feature_plan['technical_indicators'] = True
            feature_plan['rolling_features'] = True
            feature_plan['volatility_features'] = True
        
        # Check if we have volume data
        has_volume = any('volume' in col.lower() for col in data.columns)
        if has_volume:
            feature_plan['volume_features'] = True
        
        # Check data size for correlation features
        if len(data.columns) > 10 and len(data) > 1000:
            feature_plan['correlation_features'] = True
        
        # Check if we need custom features
        if len(data.columns) < 50:
            feature_plan['custom_features'] = True
        
        logger.info(f"Feature plan: {feature_plan}")
        return feature_plan
    
    def _create_features_by_plan(self, data: pd.DataFrame, feature_plan: Dict[str, Any]) -> pd.DataFrame:
        """Create features according to the feature plan"""
        if feature_plan['price_features']:
            data = self._create_price_features(data)
        
        if feature_plan['technical_indicators']:
            data = self._create_technical_indicators(data)
        
        if feature_plan['rolling_features']:
            data = self._create_rolling_features(data)
        
        if feature_plan['volatility_features']:
            data = self._create_volatility_features(data)
        
        if feature_plan['correlation_features']:
            data = self._create_correlation_features(data)
        
        if feature_plan['custom_features']:
            data = self._create_custom_features(data)
        
        return data
    
    def _auto_feature_selection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Automatically select optimal features"""
        logger.info("Auto-selecting optimal features...")
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) <= DATA_CONFIG['max_features']:
            return data
        
        # Use mutual information for feature selection
        try:
            # Create a simple target for feature selection
            if 'returns' in numeric_data.columns:
                target = numeric_data['returns']
            elif 'target' in numeric_data.columns:
                target = numeric_data['target']
            else:
                # Use first column as target
                target = numeric_data.iloc[:, 0]
            
            # Remove target from features
            feature_data = numeric_data.drop(columns=[target.name] if hasattr(target, 'name') else numeric_data.columns[0])
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(feature_data, target, random_state=42)
            
            # Select top features
            feature_scores = list(zip(feature_data.columns, mi_scores))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            top_features = [f[0] for f in feature_scores[:DATA_CONFIG['max_features']]]
            
            # Keep target column and selected features
            selected_cols = [target.name if hasattr(target, 'name') else numeric_data.columns[0]] + top_features
            selected_cols = [col for col in selected_cols if col in data.columns]
            
            data = data[selected_cols]
            self.optimal_features = top_features
            
            logger.info(f"Selected {len(top_features)} optimal features")
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Using all features.")
        
        return data
    
    def _create_custom_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create custom features based on data patterns"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Create polynomial features for important columns
        if len(numeric_data.columns) > 0:
            # Use first few columns for polynomial features
            for i, col in enumerate(numeric_data.columns[:5]):
                if data[col].std() > 0:
                    # Create squared and cubed features
                    data[f'{col}_squared'] = data[col] ** 2
                    data[f'{col}_cubed'] = data[col] ** 3
        
        # Create interaction features
        if len(numeric_data.columns) >= 2:
            for i in range(min(3, len(numeric_data.columns))):
                for j in range(i+1, min(4, len(numeric_data.columns))):
                    col1 = numeric_data.columns[i]
                    col2 = numeric_data.columns[j]
                    data[f'{col1}_{col2}_interaction'] = data[col1] * data[col2]
        
        return data
    
    def loadAndProcessData(self, file_path: str, save_processed: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Enhanced load and process data with auto-detection
        """
        logger.info(f"Loading and processing data from {file_path}")
        
        try:
            # Check for shutdown
            self._check_shutdown()
            
            # Load data
            data = self._load_data(file_path)
            
            # Auto-save state
            self._auto_save_state('data_loaded', data.shape)
            
            # Assess data quality
            self.data_quality_score = self._assess_data_quality(data)
            
            # Auto-detect splits if enabled
            if DATA_CONFIG['auto_detect_splits']:
                self.optimal_splits = self._auto_detect_splits(data)
            
            # Auto-clean data if enabled
            if self.auto_clean:
                data = self._auto_clean_data(data)
                self._auto_save_state('data_cleaned', data.shape)
            
            # Auto-engineer features if enabled
            if self.auto_feature_engineering:
                data = self._auto_feature_engineering(data)
                self._auto_save_state('features_engineered', data.shape)
            
            # Prepare walk-forward data
            X, y = self.prepare_walkforward_data(data)
            
            # Save processed data if requested
            if save_processed:
                self.save_processed_data(data, 'processed_data.parquet')
            
            # Update feature names
            self.feature_names = list(data.columns)
            
            # Final quality check
            final_quality = self._assess_data_quality(data)
            logger.info(f"Final data quality score: {final_quality:.3f}")
            
            return data, self.feature_names
            
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            raise
    
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load data with format detection"""
        logger.info(f"Loading data from {file_path}")
        
        # Check available memory
        available_memory = psutil.virtual_memory().available
        logger.info(f"Available memory: {available_memory / (1024**3):.2f} GB")
        
        # Detect file format and load accordingly
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.csv':
                # Try to detect separator
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                
                separators = [',', ';', '\t', '|']
                detected_sep = ','
                
                for sep in separators:
                    if sep in first_line and first_line.count(sep) > 1:
                        detected_sep = sep
                        break
                
                # Load CSV
                data = pd.read_csv(file_path, sep=detected_sep)
                
                # Handle Date column properly
                if 'Date' in data.columns:
                    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                    data = data.dropna(subset=['Date'])
                    data = data.sort_values('Date').reset_index(drop=True)
                    logger.info(f"Date column processed, valid dates: {len(data)}")
                
            elif file_ext == '.parquet':
                data = pd.read_parquet(file_path)
                
            elif file_ext in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path)
                
            else:
                # Try CSV as default
                data = pd.read_csv(file_path)
            
            logger.info(f"Data loaded successfully: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _fast_rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Numba-optimized rolling mean calculation"""
        if self.use_numba:
            return _numba_rolling_mean(data, window)
        else:
            return self._pandas_rolling_mean(data, window)
    
    def _fast_rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Numba-optimized rolling standard deviation"""
        if self.use_numba:
            return _numba_rolling_std(data, window)
        else:
            return self._pandas_rolling_std(data, window)
    
    def _pandas_rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Pandas rolling mean fallback"""
        series = pd.Series(data)
        return series.rolling(window).mean().values
    
    def _pandas_rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Pandas rolling std fallback"""
        series = pd.Series(data)
        return series.rolling(window).std().values
    
    def _process_chunk(self, chunk: pd.DataFrame, window: int) -> pd.DataFrame:
        """Process a chunk of data with rolling calculations"""
        if self.use_numba:
            # Use Numba-optimized functions
            for col in chunk.columns:
                if chunk[col].dtype in ['float64', 'float32']:
                    chunk[f'{col}_sma_{window}'] = self._fast_rolling_mean(
                        chunk[col].values, window
                    )
                    chunk[f'{col}_std_{window}'] = self._fast_rolling_std(
                        chunk[col].values, window
                    )
        else:
            # Fallback to pandas
            for col in chunk.columns:
                if chunk[col].dtype in ['float64', 'float32']:
                    chunk[f'{col}_sma_{window}'] = chunk[col].rolling(window).mean()
                    chunk[f'{col}_std_{window}'] = chunk[col].rolling(window).std()
        
        return chunk
    
    def _optimize_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        for col in data.columns:
            if data[col].dtype == 'float64':
                # Check if we can downcast to float32
                if data[col].notna().all():
                    min_val = data[col].min()
                    max_val = data[col].max()
                    if min_val >= -3.4e38 and max_val <= 3.4e38:
                        data[col] = data[col].astype('float32')
        
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features with multi-threading for M1 optimization
        """
        logger.info("Engineering features...")
        
        # Create price-based features
        data = self._create_price_features(data)
        
        # Create technical indicators
        data = self._create_technical_indicators(data)
        
        # Create rolling features
        data = self._create_rolling_features(data)
        
        # Create volatility features
        data = self._create_volatility_features(data)
        
        # Create correlation features
        if FEATURE_CONFIG['correlation_features']:
            data = self._create_correlation_features(data)
        
        # Feature selection and dimensionality reduction
        data = self._reduce_dimensionality(data)
        
        logger.info(f"Feature engineering complete: {data.shape}")
        return data
    
    def _create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        price_cols = [col for col in data.columns if any(price in col.lower() 
                       for price in ['open', 'high', 'low', 'close'])]
        
        if len(price_cols) >= 4:
            # Calculate returns
            for col in price_cols:
                if 'close' in col.lower():
                    data[f'{col}_returns'] = data[col].pct_change()
                    data[f'{col}_log_returns'] = np.log(data[col] / data[col].shift(1))
            
            # Calculate price ratios
            if len(price_cols) >= 4:
                data['hl_ratio'] = data[price_cols[1]] / data[price_cols[2]]  # high/low
                data['oc_ratio'] = data[price_cols[0]] / data[price_cols[3]]  # open/close
        
        return data
    
    def _create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators using TA library"""
        try:
            import ta
            
            # RSI
            if any('close' in col.lower() for col in data.columns):
                close_col = [col for col in data.columns if 'close' in col.lower()][0]
                data['rsi'] = ta.momentum.RSIIndicator(data[close_col]).rsi()
            
            # MACD
            if 'close' in data.columns:
                macd = ta.trend.MACD(data['close'])
                data['macd'] = macd.macd()
                data['macd_signal'] = macd.macd_signal()
                data['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            if 'close' in data.columns:
                bb = ta.volatility.BollingerBands(data['close'])
                data['bb_upper'] = bb.bollinger_hband()
                data['bb_lower'] = bb.bollinger_lband()
                data['bb_middle'] = bb.bollinger_mavg()
            
        except ImportError:
            logger.warning("TA library not available, skipping technical indicators")
        
        return data
    
    def _create_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create rolling features with multi-threading"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        def process_rolling_features(col):
            features = {}
            for window in FEATURE_CONFIG['rolling_windows']:
                if self.use_numba:
                    features[f'{col}_sma_{window}'] = self._fast_rolling_mean(
                        data[col].values, window
                    )
                    features[f'{col}_std_{window}'] = self._fast_rolling_std(
                        data[col].values, window
                    )
                else:
                    features[f'{col}_sma_{window}'] = data[col].rolling(window).mean()
                    features[f'{col}_std_{window}'] = data[col].rolling(window).std()
            return features
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_rolling_features, numeric_cols))
        
        # Combine results
        for result in results:
            for feature_name, feature_values in result.items():
                data[feature_name] = feature_values
        
        return data
    
    def _create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volatility features"""
        # Calculate rolling volatility
        for window in [20, 50, 100]:
            if 'returns' in data.columns:
                data[f'volatility_{window}'] = data['returns'].rolling(window).std()
        
        return data
    
    def _create_correlation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create correlation-based features"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Calculate rolling correlations
        for window in [50, 100]:
            corr_features = []
            for i, col1 in enumerate(numeric_cols[:10]):  # Limit to first 10 columns
                for col2 in numeric_cols[i+1:11]:
                    if col1 != col2:
                        corr = data[col1].rolling(window).corr(data[col2])
                        data[f'corr_{col1}_{col2}_{window}'] = corr
                        corr_features.append(f'corr_{col1}_{col2}_{window}')
        
        return data
    
    def _reduce_dimensionality(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reduce dimensionality using PCA and feature selection"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Remove infinite values
        numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
        
        # More aggressive NaN handling
        logger.info(f"NaN values before cleaning: {numeric_data.isnull().sum().sum()}")
        
        # Forward fill, then backward fill, then fill remaining with 0
        numeric_data = numeric_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"NaN values after cleaning: {numeric_data.isnull().sum().sum()}")
        
        # Feature selection
        if len(numeric_data.columns) > FEATURE_CONFIG['pca_components']:
            try:
                selector = SelectKBest(score_func=f_regression, k=FEATURE_CONFIG['pca_components'])
                selected_features = selector.fit_transform(numeric_data, numeric_data.iloc[:, 0])
                selected_columns = numeric_data.columns[selector.get_support()]
                
                # PCA for further reduction
                if len(selected_columns) > 50:
                    self.pca = PCA(n_components=50, random_state=42)
                    pca_features = self.pca.fit_transform(selected_features)
                    
                    # Create new dataframe with PCA features
                    pca_columns = [f'pca_{i}' for i in range(50)]
                    pca_df = pd.DataFrame(pca_features, index=data.index, columns=pca_columns)
                    
                    # Combine with original selected features
                    selected_df = data[selected_columns]
                    data = pd.concat([selected_df, pca_df], axis=1)
                else:
                    data = data[selected_columns]
            except Exception as e:
                logger.warning(f"Feature selection failed: {e}. Using all numeric features.")
                data = data[numeric_data.columns]
        else:
            data = data[numeric_data.columns]
        
        self.feature_names = list(data.columns)
        logger.info(f"Final feature count: {len(self.feature_names)}")
        
        return data
    
    def prepare_walkforward_data(self, data: pd.DataFrame, 
                               target_col: str = 'returns') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for walk-forward analysis
        """
        logger.info("Preparing walk-forward data...")
        
        # Ensure target column exists
        if target_col not in data.columns:
            # Create returns from close price if available
            close_cols = [col for col in data.columns if 'close' in col.lower()]
            if close_cols:
                data[target_col] = data[close_cols[0]].pct_change()
            else:
                # Use first numeric column as target
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                data[target_col] = data[numeric_cols[0]].pct_change()
        
        # Remove rows with null target
        data = data.dropna(subset=[target_col])
        
        # Select only numeric columns for features, excluding the target
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Also exclude date-like columns and other non-feature columns
        exclude_cols = ['Date', 'Ticker', 'date', 'ticker', 'time', 'timestamp']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"Selected {len(feature_cols)} numeric feature columns")
        logger.info(f"Feature columns: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
        
        # Create features (X) and target (y) - only numeric features
        X = data[feature_cols].values
        y = data[target_col].values
        
        # Final NaN check and cleaning
        logger.info(f"NaN values in X before final cleaning: {np.isnan(X).sum()}")
        logger.info(f"NaN values in y before final cleaning: {np.isnan(y).sum()}")
        
        # Replace any remaining NaN values with 0
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Walk-forward data prepared: X={X_scaled.shape}, y={y.shape}")
        logger.info(f"Final NaN check - X: {np.isnan(X_scaled).sum()}, y: {np.isnan(y).sum()}")
        return X_scaled, y
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return {}
        
        feature_importance = dict(zip(feature_names, importance))
        return dict(sorted(feature_importance.items(), 
                          key=lambda x: x[1], reverse=True)[:20])
    
    def save_processed_data(self, data: pd.DataFrame, filename: str):
        """Save processed data efficiently"""
        filepath = PATHS['data'] / filename
        data.to_parquet(filepath, compression='gzip')
        logger.info(f"Processed data saved to {filepath}")
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Load processed data"""
        filepath = PATHS['data'] / filename
        data = pd.read_parquet(filepath)
        logger.info(f"Processed data loaded from {filepath}")
        return data
