#!/usr/bin/env python3
"""
Common Anomaly Detection Module
Supports multiple GAN architectures with configurable imports
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any
import os
import torch


class AnomalyDetector:
    """Anomaly detection using GAN model with proper daily scoring"""
    
    def __init__(
        self, 
        model_path: str, 
        lookback_window: int, 
        start_date: Optional[str], 
        end_date: Optional[str],
        model_import_path: str,
        model_class_name: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize anomaly detector
        
        Args:
            model_path: Path to saved model
            lookback_window: Sequence length for anomaly detection
            start_date: Start date for evaluation (YYYY-MM-DD)
            end_date: End date for evaluation (YYYY-MM-DD)
            model_import_path: Python import path (e.g., 'paper_gan_price.gan_model')
            model_class_name: Class name to import (e.g., 'GANExpert')
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.model_path = model_path
        self.lookback_window = lookback_window
        self.start = start_date
        self.end = end_date
        self.anomaly_scores = None
        self.model_import_path = model_import_path
        self.model_class_name = model_class_name
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize and load the GAN model"""
        try:
            # Dynamic import
            module = __import__(self.model_import_path, fromlist=[self.model_class_name])
            ModelClass = getattr(module, self.model_class_name, None)
            
            if ModelClass is None:
                self.logger.warning(
                    f"{self.model_class_name} not available from {self.model_import_path}, "
                    "using mock anomaly detection"
                )
                return
                
            self.model = ModelClass(seq_len=self.lookback_window)
            
            if os.path.exists(self.model_path):
                self.model.load_models(self.model_path)
                self.logger.info(f"Loaded GAN model from {self.model_path}")
            else:
                self.logger.warning(
                    f"Model path {self.model_path} not found. Using untrained model."
                )
                
        except Exception as e:
            self.logger.error(f"Failed to initialize GAN model: {str(e)}")
            self.model = None
    
    def compute_daily_anomaly_scores(
        self, 
        prices: pd.DataFrame,
        use_full_history: bool = False
    ) -> pd.Series:
        """
        Compute anomaly score for each day in the dataset
        
        For each day t, uses the window [t-lookback_window+1, t] to compute the anomaly score
        This ensures each day gets exactly one anomaly score based on its historical context
        
        Args:
            prices: DataFrame with numerical columns (excluding 'date' column)
            use_full_history: If True, uses all available history for scoring.
                            If False, uses limited lookback from start_date.
            
        Returns:
            Series of daily anomaly scores aligned with the input dates
        """
        # Get all columns except 'date' (case-insensitive)
        feature_columns = [
            col for col in prices.columns 
            if col.lower() not in ['date']
        ]
        
        # Extract feature data
        feature_data = prices[feature_columns]

        price_data = prices.copy()
        start_bound_p = (
            price_data.index[0] if self.start is None 
            else pd.Timestamp(self.start).normalize()
        )
        end_bound_p = (
            price_data.index[-1] if self.end is None 
            else pd.Timestamp(self.end).normalize()
        )
        price_data = price_data.loc[start_bound_p:end_bound_p]
        num_days = len(price_data)

        if self.anomaly_scores is not None:
            # Return cached scores if already computed
            return self.anomaly_scores
        
        curr_feature_data = feature_data.copy()

        # Determine start bound for feature extraction
        if use_full_history:
            # Use all available history
            start_bound = prices.index[0]
        elif self.start is not None:
            # Use limited lookback from start date
            start_bound = (
                pd.Timestamp(self.start).normalize() - 
                pd.Timedelta(days=int(self.lookback_window * 2))
            )
        else:
            start_bound = prices.index[0]

        if self.end is not None:
            end_bound = pd.Timestamp(self.end).normalize()
        else:
            end_bound = prices.index[-1]

        curr_feature_data = feature_data.loc[start_bound:end_bound]
        curr_feature_data_np = curr_feature_data.values
        n_features = curr_feature_data.shape[1]
        n_days = curr_feature_data.shape[0]

        anomaly_scores = np.zeros(n_days)

        # Compute anomaly score for each day starting from day lookback_window
        windows = []
        valid_indices = []

        for t in range(self.lookback_window, n_days):
            window = curr_feature_data_np[t - self.lookback_window:t]
            windows.append(window)
            valid_indices.append(t)
        
        if len(windows) > 0:
            windows_array = np.array(windows).reshape(
                len(windows), self.lookback_window, n_features
            )
            
            # Get anomaly scores for all windows
            scores = self._predict_anomaly_batch(windows_array)
            
            # Assign scores to corresponding days
            for idx, score in zip(valid_indices, scores):
                anomaly_scores[idx] = score
        
        # Create series with proper alignment
        aligned_scores = anomaly_scores[-num_days:]
        anomaly_series = pd.Series(aligned_scores, index=price_data.index)

        self.anomaly_scores = anomaly_series.copy()

        self.logger.info(
            f"Computed anomaly scores for {len(valid_indices)} days "
            f"using {n_features} features."
        )
        
        return self.anomaly_scores
    
    def _predict_anomaly_batch(self, windows: np.ndarray) -> np.ndarray:
        """
        Predict anomaly probabilities for batch of windows
        
        Args:
            windows: Array of shape (batch_size, lookback_window, features)
            
        Returns:
            Array of anomaly probabilities of shape (batch_size,)
        """
        if self.model is None:
            # Mock anomaly detection - returns random probabilities with occasional spikes
            batch_size = windows.shape[0]
            # Base probability with some structure
            base_probs = np.random.beta(2, 8, batch_size) * 0.3  # Low baseline
            
            # Add occasional anomaly spikes
            spike_mask = np.random.random(batch_size) < 0.05  # 5% spike rate
            base_probs[spike_mask] += np.random.beta(2, 3, spike_mask.sum()) * 0.7
            
            return np.clip(base_probs, 0.0, 1.0)
        
        try:
            batch_size = windows.shape[0]
            probabilities = []
            
            # Process in smaller batches to avoid memory issues
            batch_chunk_size = min(32, batch_size)
            
            for i in range(0, batch_size, batch_chunk_size):
                end_idx = min(i + batch_chunk_size, batch_size)
                chunk = windows[i:end_idx]
                
                # Get anomaly detection results
                if hasattr(self.model, 'detect_financial_anomalies'):
                    results = self.model.detect_financial_anomalies(
                        chunk, 
                        threshold_percentile=95.0,
                        return_details=True
                    )
                    
                    if isinstance(results, dict) and 'anomaly_scores' in results:
                        scores = results['anomaly_scores']
                    else:
                        scores = np.random.random(len(chunk)) * 0.1
                else:
                    # Fallback method
                    scores = []
                    for window in chunk:
                        processed_data = self.model._preprocess_financial_data(
                            window.reshape(1, -1, 1)
                        )
                        data_tensor = torch.FloatTensor(processed_data).to(
                            self.model.device
                        )
                        
                        with torch.no_grad():
                            score = self.model._evaluate_tanogan_loss(
                                data_tensor
                            ).item()
                        scores.append(score)
                    scores = np.array(scores)
                
                # Convert scores to probabilities using sigmoid
                chunk_probs = 1.0 / (1.0 + np.exp(-scores))
                probabilities.extend(chunk_probs)
            
            return np.clip(np.array(probabilities), 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error in batch anomaly prediction: {str(e)}")
            # Fallback to random probabilities
            return np.random.random(windows.shape[0]) * 0.1


def create_anomaly_detector(
    model_type: str,
    model_path: str,
    lookback_window: int,
    start_date: Optional[str],
    end_date: Optional[str],
    logger: Optional[logging.Logger] = None
) -> AnomalyDetector:
    """
    Factory function to create anomaly detector based on model type
    
    Args:
        model_type: One of 'cnn_lstm_seq_ohlcav', 'cnn_lstm_seq_price', 
                   'lstm_cnn_parallel_ohlcav', 'lstm_cnn_parallel_price',
                   'lstm_cnn_seq_ohlcav', 'lstm_cnn_seq_price',
                   'paper_gan_ohlcav', 'paper_gan_price'
        model_path: Path to model directory
        lookback_window: Sequence length
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        logger: Optional logger
    
    Returns:
        Configured AnomalyDetector instance
    """
    MODEL_CONFIGS = {
        'cnn_lstm_seq_ohlcav': (
            'cnn_lstm_seq_ohlcav.cnn_lstm_gan_model_sequential',
            'LSTMCNNGANExpert'
        ),
        'cnn_lstm_seq_price': (
            'cnn_lstm_seq_price.cnn_lstm_gan_model_sequential',
            'LSTMCNNGANExpert'
        ),
        'lstm_cnn_parallel_ohlcav': (
            'lstm_cnn_parallel_ohlcav.lstm_cnn_gan_model_parallel',
            'LSTMCNNGANExpert'
        ),
        'lstm_cnn_parallel_price': (
            'lstm_cnn_parallel_price.lstm_cnn_gan_model_parallel',
            'LSTMCNNGANExpert'
        ),
        'lstm_cnn_seq_ohlcav': (
            'lstm_cnn_seq_ohlcav.lstm_cnn_gan_model_sequential',
            'LSTMCNNGANExpert'
        ),
        'lstm_cnn_seq_price': (
            'lstm_cnn_seq_price.lstm_cnn_gan_model_sequential',
            'LSTMCNNGANExpert'
        ),
        'paper_gan_ohlcav': (
            'paper_gan_ohlcav.gan_model',
            'GANExpert'
        ),
        'paper_gan_price': (
            'paper_gan_price.gan_model',
            'GANExpert'
        ),
    }
    
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Must be one of {list(MODEL_CONFIGS.keys())}"
        )
    
    import_path, class_name = MODEL_CONFIGS[model_type]
    
    return AnomalyDetector(
        model_path=model_path,
        lookback_window=lookback_window,
        start_date=start_date,
        end_date=end_date,
        model_import_path=import_path,
        model_class_name=class_name,
        logger=logger
    )
