"""
Probability Calibration Module
Handles probability calibration using OOF data only to prevent data leakage
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, mean_absolute_error
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class ProbabilityCalibrator:
    """Handles probability calibration for ensemble models"""
    
    def __init__(self, n_splits: int = 5, method: str = 'isotonic'):
        self.n_splits = n_splits
        self.method = method
        self.calibrators = {}
        
    def calibrate_model(self, model, X: np.ndarray, y: np.ndarray, 
                       oof_proba: np.ndarray) -> CalibratedClassifierCV:
        """Calibrate a single model using OOF data"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        calibrated_model = CalibratedClassifierCV(
            model, method=self.method, cv=tscv
        )
        calibrated_model.fit(X, y)
        
        return calibrated_model
    
    def evaluate_calibration(self, proba: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate calibration quality"""
        brier = brier_score_loss(y, proba)
        calibration_error = self._calculate_calibration_error(proba, y)
        
        return {
            'brier_score': brier,
            'calibration_mae': calibration_error
        }
    
    def _calculate_calibration_error(self, proba: np.ndarray, y: np.ndarray) -> float:
        """Calculate calibration error using MAE"""
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(proba, bins) - 1
        
        calibration_error = 0
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_proba = np.mean(proba[mask])
                bin_actual = np.mean(y[mask])
                calibration_error += np.abs(bin_proba - bin_actual)
        
        return calibration_error / (len(bins) - 1)
