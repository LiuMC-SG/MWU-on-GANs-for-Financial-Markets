#!/usr/bin/env python3
"""
Financial Anomaly Detection Backtesting System with Optuna Optimization and VectorBT
Designed for SLURM cluster execution with comprehensive logging
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import json
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import date, datetime
import warnings
warnings.filterwarnings('ignore')

# Optuna for hyperparameter optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# VectorBT for efficient backtesting
import vectorbt as vbt

# Try to import MPI for parallel processing
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

# Import GANExpert - Updated import path
try:
    from lstm_cnn_gan_model import LSTMCNNGANExpert
except ImportError:
    LSTMCNNGANExpert = None

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    # Moving average ranges for optimization
    short_ma_min: int = 5
    short_ma_max: int = 30
    long_ma_min: int = 20
    long_ma_max: int = 300
    
    # Current parameter values (will be optimized)
    short_ma_period: int = 10
    long_ma_period: int = 50
    lookback_window: int = 20
    
    # Trading parameters
    initial_capital: float = 100000.0
    transaction_cost: float = 0.005  # 0.5% per transaction
    
    # Time period configuration
    start_date: Optional[str] = None  # Format: 'YYYY-MM-DD'
    end_date: Optional[str] = None    # Format: 'YYYY-MM-DD'
    train_start_date: Optional[str] = None  # For model training period
    train_end_date: Optional[str] = None    # For model training period
    
    # I/O Configuration
    input_file: str = ""
    output_dir: str = "./results"
    log_level: str = "INFO"
    
    # Optuna optimization parameters
    optimize_params: bool = False
    n_trials: int = 100
    
    # Model configuration
    model_path: str = "./models"
    anomaly_threshold_percentile: float = 95.0
    
    # VectorBT configuration
    freq: str = '1D'  # Data frequency
    
    def __post_init__(self):
        # Validate ranges
        if self.short_ma_max >= self.long_ma_min:
            self.long_ma_min = self.short_ma_max + 1

class Logger:
    """Centralized logging configuration"""
    
    @staticmethod
    def setup_logger(name: str, log_file: str, level: str = "INFO") -> logging.Logger:
        """Setup logger with both file and console handlers"""
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger

class AnomalyDetector:
    """Anomaly detection using GAN model with proper daily scoring"""
    
    def __init__(self, model_path: str, lookback_window: int, start_date, end_date, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.model_path = model_path
        self.lookback_window = lookback_window
        self.start = start_date
        self.end = end_date
        self.anomaly_scores = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize and load the GAN model"""
        try:
            if LSTMCNNGANExpert is None:
                self.logger.warning("LSTMCNNGANExpert not available, using mock anomaly detection")
                return
                
            self.model = LSTMCNNGANExpert(seq_len=self.lookback_window)
            
            if os.path.exists(self.model_path):
                self.model.load_models(self.model_path)
                self.logger.info(f"Loaded GAN model from {self.model_path}")
            else:
                self.logger.warning(f"Model path {self.model_path} not found. Using untrained model.")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize GAN model: {str(e)}")
            self.model = None
    
    def compute_daily_anomaly_scores(self, prices: pd.DataFrame) -> pd.Series:
        """
        Compute anomaly score for each day in the dataset
        
        For each day t, uses the window [t-lookback_window+1, t] to compute the anomaly score
        This ensures each day gets exactly one anomaly score based on its historical context
        
        Args:
            price_data: DataFrame with numerical columns (excluding 'date' column)
            
        Returns:
            Series of daily anomaly scores aligned with the input dates
        """
        # Get all columns except 'date' (case-insensitive)
        feature_columns = [col for col in prices.columns 
                        if col.lower() not in ['date']]
        
        # Extract feature data
        feature_data = prices[feature_columns]

        price_data = prices.copy()
        start_bound_p = price_data.index[0] if self.start is None else pd.Timestamp(self.start).normalize()
        end_bound_p = price_data.index[-1] if self.end is None else pd.Timestamp(self.end).normalize()
        price_data = price_data.loc[start_bound_p:end_bound_p]
        num_days = len(price_data)

        if self.anomaly_scores is not None:
            # Return cached scores if already computed
            return self.anomaly_scores
        
        curr_feature_data = feature_data.copy()

        if self.start is not None:
            # If you meant “seq_len calendar days lookback”
            start_bound = pd.Timestamp(self.start).normalize() - pd.Timedelta(days=int(self.lookback_window*2))
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
            windows_array = np.array(windows).reshape(len(windows), self.lookback_window, n_features)
            
            # Get anomaly scores for all windows
            scores = self._predict_anomaly_batch(windows_array)
            
            # Assign scores to corresponding days
            for idx, score in zip(valid_indices, scores):
                anomaly_scores[idx] = score
        
        # Create series with proper alignment
        aligned_scores = anomaly_scores[-num_days:]
        anomaly_series = pd.Series(aligned_scores, index=price_data.index)

        self.anomaly_scores = anomaly_series.copy()

        self.logger.info(f"Computed anomaly scores for {len(valid_indices)} days "
                        f"using {n_features} features.")
        
        return self.anomaly_scores
    
    def _predict_anomaly_batch(self, windows: np.ndarray) -> np.ndarray:
        """
        Predict anomaly probabilities for batch of windows
        
        Args:
            windows: Array of shape (batch_size, lookback_window, features)
            
        Returns:
            Array of anomaly probabilities of shape (batch_size,)
        """
        if self.model is None or LSTMCNNGANExpert is None:
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
                        processed_data = self.model._preprocess_financial_data(window.reshape(1, -1, 1))
                        import torch
                        data_tensor = torch.FloatTensor(processed_data).to(self.model.device)
                        
                        with torch.no_grad():
                            score = self.model._evaluate_tanogan_loss(data_tensor).item()
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

class VectorBTBacktester:
    """Main backtesting engine using VectorBT with proper daily anomaly scoring"""
    
    def __init__(self, config: BacktestConfig, anomaly_detector, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize anomaly detector with lookback window
        self.anomaly_detector = anomaly_detector
        
        self.logger.info(f"Initialized VectorBTBacktester with config: {asdict(config)}")
    
    def generate_anomaly_signals(self, price_data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate anomaly signals with proper daily alignment
        
        Args:
            price_data: DataFrame with 'Adj Close' and 'Date' columns
            
        Returns:
            Tuple of (anomaly_signals, short_ma, long_ma, anomaly_scores)
        """
        self.logger.info("Generating daily anomaly scores...")
        
        # Get daily anomaly scores
        anomaly_scores = self.anomaly_detector.compute_daily_anomaly_scores(price_data)
        
        # Fill NaN values (from first lookback_window-1 days) with 0
        anomaly_scores_filled = anomaly_scores.fillna(0)
        
        # Calculate moving averages for signal generation
        short_ma = anomaly_scores_filled.rolling(
            window=self.config.short_ma_period, 
            min_periods=1
        ).mean()
        
        long_ma = anomaly_scores_filled.rolling(
            window=self.config.long_ma_period,
            min_periods=1
        ).mean()
        
        # Generate signals: anomaly when short MA > long MA
        anomaly_signals = (short_ma > long_ma).astype(int)
        
        # Log statistics
        valid_scores = anomaly_scores_filled[anomaly_scores_filled > 0]
        if len(valid_scores) > 0:
            self.logger.info(f"Anomaly score statistics: mean={valid_scores.mean():.4f}, "
                           f"std={valid_scores.std():.4f}, max={valid_scores.max():.4f}")
        
        self.logger.info(f"Generated {anomaly_signals.sum()} anomaly signals out of {len(anomaly_signals)} days "
                        f"({anomaly_signals.mean()*100:.2f}%)")
        
        return anomaly_signals, short_ma, long_ma, anomaly_scores_filled
    
    def create_trading_signals(self, anomaly_signals: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Create entry and exit signals based on anomaly detection
        
        Trading Logic:
        - Hold position when no anomaly (normal market)
        - Exit position when anomaly detected
        - Re-enter when anomaly ends
        
        Args:
            anomaly_signals: Series of anomaly signals (1 = anomaly, 0 = normal)
            
        Returns:
            Tuple of (entries, exits) boolean series
        """
        self.logger.info("Creating trading signals from anomaly signals...")
        
        # Initialize signals
        entries = pd.Series(False, index=anomaly_signals.index)
        exits = pd.Series(False, index=anomaly_signals.index)
        
        # Detect transitions
        # Shift to compare with previous day
        prev_signal = anomaly_signals.shift(1)
        
        # Exit when anomaly starts (transition from 0 to 1)
        exits = (prev_signal == 0) & (anomaly_signals == 1)
        
        # Enter when anomaly ends (transition from 1 to 0)
        entries = (prev_signal == 1) & (anomaly_signals == 0)
        
        # Handle first day
        if anomaly_signals.iloc[0] == 0:
            entries.iloc[0] = True  # Enter on first day if no anomaly
        
        self.logger.info(f"Generated {entries.sum()} entry signals and {exits.sum()} exit signals")
        
        return entries, exits
    
    def backtest(self, price_data: pd.DataFrame) -> Dict:
        """Run backtest using VectorBT with daily frequency"""
        self.logger.info(f"Starting VectorBT backtest with {len(price_data)} data points")
        
        try:
            # Filter data by time period for testing
            filtered_data = self._filter_data_by_time_period(
                price_data, 
                self.config.start_date, 
                self.config.end_date
            )
            self.logger.info(f"Using {len(filtered_data)} days for backtesting period")
            
            # Generate anomaly signals for the test period
            test_data = price_data.copy()
            test_data.drop(columns=['adj_close'], inplace=True, errors='ignore')
            test_data.set_index('date', inplace=True)
            anomaly_signals, short_ma, long_ma, anomaly_scores = self.generate_anomaly_signals(test_data)
            
            # Create trading signals
            entries, exits = self.create_trading_signals(anomaly_signals)
            
            # Set up VectorBT portfolio
            self.logger.info("Running VectorBT portfolio simulation...")
            
            # Create portfolio with transaction costs
            pf = vbt.Portfolio.from_signals(
                close=filtered_data['adj_close'],
                entries=entries,
                exits=exits,
                init_cash=self.config.initial_capital,
                fees=self.config.transaction_cost,
                freq=self.config.freq
            )
            
            # Calculate performance metrics
            results = self._calculate_vectorbt_metrics(
                pf, filtered_data, anomaly_signals, 
                short_ma, long_ma, anomaly_scores
            )
            
            self.logger.info(f"Backtest completed - Total Return: {results['total_return']:.2f}%")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during backtesting: {str(e)}", exc_info=True)
            raise
    
    def _coerce_date_column(self, df: pd.DataFrame, col: str = 'date') -> pd.DataFrame:
        """
        Coerce df[col] to pandas datetime, supporting:
        1) Epoch (seconds) — numeric dtype or digit-like strings (default assumption).
        2) ISO dates in 'YYYY-MM-DD' string format.

        Returns tz-naive datetime64[ns] in df[col].
        """
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

        ser = df[col]
        out = None

        # Helper: decide epoch unit by magnitude (fallback safety; default is seconds)
        def _epoch_unit_from_magnitude(values: pd.Series) -> str:
            # Use median of absolute values to be robust to outliers/NaNs
            arr = pd.to_numeric(values, errors='coerce').astype('float64')
            if np.all(np.isnan(arr)):
                return 's'
            med = float(np.nanmedian(np.abs(arr)))
            # Typical magnitudes:
            #   seconds:    ~1e9  (1970..2100)
            #   milliseconds: ~1e12
            #   nanoseconds:  ~1e18
            if med >= 1e15:
                return 'ns'
            elif med >= 1e12:
                return 'ms'
            else:
                return 's'

        s = pd.to_numeric(ser, errors="coerce").astype("float64")
        s[~np.isfinite(s)] = np.nan
        mask_num = s.notna()

        unit = _epoch_unit_from_magnitude(s.loc[mask_num])

        dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        if mask_num.any():
            dt.loc[mask_num] = pd.to_datetime(s.loc[mask_num], unit=unit, utc=True).dt.tz_convert(None)

        # ---- optional: explicitly parse remaining string timestamps with a known format ----
        mask_str = ~mask_num
        string_date_format = '%Y-%m-%d'
        if mask_str.any():
            dt2 = pd.to_datetime(
                ser.loc[mask_str],
                format=string_date_format,   # explicit format => no "could not infer" warning
                errors="coerce",
                utc=True,
            )
            dt.loc[mask_str] = dt2.dt.tz_convert(None)

        # Drop rows that could not be parsed
        good = dt.notna()
        if not good.any():
            hint = " Provide --string-date-format if your file mixes epoch and formatted strings." \
                if string_date_format is None else ""
        else:
            hint = ""

        if (~good).sum() > 0:
            self.logger.warning("Dropping %d row(s) with unparseable timestamps in %s.%s",
                                int((~good).sum()), col, hint)

        df = df.loc[good].copy()
        df["date"] = dt.loc[good].astype("datetime64[ns]")

        return df

    def _filter_data_by_time_period(self,
                                    data: pd.DataFrame,
                                    start_date: Optional[str] = None,
                                    end_date: Optional[str] = None) -> pd.DataFrame:
        """Filter data based on configured start and end dates (YYYY-MM-DD), inclusive on both ends."""
        df = data.copy()

        # Ensure Date is datetime64[ns]
        df = self._coerce_date_column(df, 'date')

        # Remove timezone to avoid comparison surprises
        if pd.api.types.is_datetime64tz_dtype(df['date'].dtype):
            df['date'] = df['date'].dt.tz_convert(None)

        # Compare on calendar dates only to avoid time-of-day edge cases
        date_only = df['date'].dt.normalize().dt.date

        def _parse_ymd(s: str) -> date:
            # Strict format parsing prevents accidental reinterpretation of month/day
            return pd.to_datetime(s, format="%Y-%m-%d", errors="raise").date()

        sd = _parse_ymd(start_date) if start_date else None
        ed = _parse_ymd(end_date) if end_date else None

        if sd and ed and sd > ed:
            raise ValueError("start_date must be on or before end_date.")

        self.logger.info("Using %d rows for backtesting before date filtering", len(df))
        if sd:
            self.logger.info("Applying start_date (inclusive): %s", sd)
        if ed:
            self.logger.info("Applying end_date (inclusive): %s", ed)

        # Build mask with inclusive bounds
        mask = pd.Series(True, index=df.index)
        if sd:
            mask &= (date_only >= sd)
        if ed:
            mask &= (date_only <= ed)

        df = df.loc[mask]

        if df.empty:
            raise ValueError("No data available for specified date range.")

        self.logger.info(
            "Post-filter: starts %s, ends %s, rows=%d",
            df['date'].min().date(), df['date'].max().date(), len(df)
        )

        # Ensure we have enough data for the lookback window
        if len(df) < self.config.lookback_window:
            raise ValueError(
                f"Insufficient data: need at least {self.config.lookback_window} days, "
                f"but only have {len(df)} days"
            )

        df.sort_values(by='date', inplace=True)

        return df.reset_index(drop=True)
    
    def _calculate_vectorbt_metrics(self, pf, price_data: pd.DataFrame, 
                                anomaly_signals: pd.Series,
                                short_ma: pd.Series, long_ma: pd.Series, 
                                anomaly_scores: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics"""
        self.logger.info("Calculating performance metrics...")
        
        try:
            # Basic portfolio metrics
            total_return = pf.total_return() * 100
            
            # Buy and hold comparison
            buy_hold_return = ((price_data['adj_close'].iloc[-1] / 
                            price_data['adj_close'].iloc[0]) - 1) * 100
            
            # VectorBT statistics
            stats = pf.stats()
            
            # Risk metrics - handle potential NaN values properly
            sharpe_ratio = pf.sharpe_ratio()
            sharpe_ratio = float(sharpe_ratio) if not pd.isna(sharpe_ratio) else 0.0
            
            max_drawdown = pf.max_drawdown() * 100
            max_drawdown = float(max_drawdown) if not pd.isna(max_drawdown) else 0.0
            
            calmar_ratio = pf.calmar_ratio()
            calmar_ratio = float(calmar_ratio) if not pd.isna(calmar_ratio) else 0.0
            
            # Trade statistics - fixed attribute access
            trades = pf.trades
            try:
                # Check if we have any trades
                if hasattr(trades, 'records_readable'):
                    records = trades.records_readable
                    has_trades = len(records) > 0 if hasattr(records, '__len__') else False
                else:
                    has_trades = False
                    records = None
                
                if has_trades:
                    # Calculate win rate
                    if hasattr(trades, 'win_rate'):
                        win_rate_val = trades.win_rate()
                        win_rate = float(win_rate_val) * 100 if not pd.isna(win_rate_val) else 0
                    else:
                        win_rate = 0
                    
                    # Handle winning trades
                    if hasattr(trades, 'winning'):
                        winning_trades = trades.winning
                        # Check if winning has a count method/property
                        if hasattr(winning_trades, 'count'):
                            win_count = winning_trades.count() if callable(winning_trades.count) else winning_trades.count
                        else:
                            win_count = 0
                        
                        if win_count > 0:
                            avg_win = float(winning_trades.mean()) if hasattr(winning_trades, 'mean') else 0
                        else:
                            avg_win = 0
                    else:
                        avg_win = 0
                        
                    # Handle losing trades
                    if hasattr(trades, 'losing'):
                        losing_trades = trades.losing
                        # Check if losing has a count method/property
                        if hasattr(losing_trades, 'count'):
                            loss_count = losing_trades.count() if callable(losing_trades.count) else losing_trades.count
                        else:
                            loss_count = 0
                            
                        if loss_count > 0:
                            avg_loss = abs(float(losing_trades.mean())) if hasattr(losing_trades, 'mean') else 0
                        else:
                            avg_loss = 0
                    else:
                        avg_loss = 0
                        
                    profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
                    num_trades = len(records)
                    trades_records = records.to_dict('records') if hasattr(records, 'to_dict') else []
                else:
                    win_rate = 0
                    avg_win = 0
                    avg_loss = 0
                    profit_factor = 0
                    num_trades = 0
                    trades_records = []
                    
            except Exception as trade_error:
                self.logger.warning(f"Error processing trade statistics: {trade_error}")
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
                num_trades = 0
                trades_records = []
            
            # Calculate daily returns - handle potential division by zero
            portfolio_values = pf.value()
            if hasattr(portfolio_values, 'values'):
                portfolio_values = portfolio_values.values
            
            # Ensure we have valid portfolio values
            if len(portfolio_values) > 1:
                # Calculate returns avoiding division by zero
                daily_returns = []
                for i in range(1, len(portfolio_values)):
                    if portfolio_values[i-1] != 0:
                        ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                        daily_returns.append(ret)
                    else:
                        daily_returns.append(0.0)
            else:
                daily_returns = []
            
            # Handle date column - check if it exists and is datetime
            if 'date' in price_data.columns:
                if pd.api.types.is_datetime64_any_dtype(price_data['date']):
                    dates = price_data['date'].dt.strftime('%Y-%m-%d').tolist()
                else:
                    # If date is already string or needs conversion
                    dates = price_data['date'].astype(str).tolist()
            else:
                # Use index if no date column
                dates = price_data.index.strftime('%Y-%m-%d').tolist() if hasattr(price_data.index, 'strftime') else list(range(len(price_data)))
            
            # Safely convert stats to dict
            vectorbt_stats = {}
            if hasattr(stats, 'to_dict'):
                stats_dict = stats.to_dict()
                for k, v in stats_dict.items():
                    try:
                        if pd.isna(v):
                            vectorbt_stats[k] = 0.0
                        else:
                            vectorbt_stats[k] = float(v)
                    except (ValueError, TypeError):
                        vectorbt_stats[k] = 0.0
            
            results = {
                'total_return': float(total_return) if not pd.isna(total_return) else 0.0,
                'buy_hold_return': float(buy_hold_return) if not pd.isna(buy_hold_return) else 0.0,
                'excess_return': float(total_return - buy_hold_return) if not (pd.isna(total_return) or pd.isna(buy_hold_return)) else 0.0,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': float(win_rate) if not pd.isna(win_rate) else 0.0,
                'profit_factor': float(profit_factor) if profit_factor != float('inf') else 999.99,
                'num_trades': num_trades,
                'portfolio_values': portfolio_values.tolist() if hasattr(portfolio_values, 'tolist') else list(portfolio_values),
                'anomaly_signals': anomaly_signals.values.tolist() if hasattr(anomaly_signals, 'values') else anomaly_signals.tolist(),
                'short_ma': short_ma.values.tolist() if hasattr(short_ma, 'values') else short_ma.tolist(),
                'long_ma': long_ma.values.tolist() if hasattr(long_ma, 'values') else long_ma.tolist(),
                'anomaly_scores': anomaly_scores.values.tolist() if hasattr(anomaly_scores, 'values') else anomaly_scores.tolist(),
                'dates': dates,
                'prices': price_data['adj_close'].values.tolist(),
                'daily_returns': daily_returns,
                'trades': trades_records,
                'config': asdict(self.config),
                'vectorbt_stats': vectorbt_stats
            }
            
            self.logger.info("Performance metrics calculated successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            # Return a default structure to prevent downstream errors
            return {
                'total_return': 0.0,
                'buy_hold_return': 0.0,
                'excess_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'num_trades': 0,
                'portfolio_values': [],
                'anomaly_signals': [],
                'short_ma': [],
                'long_ma': [],
                'anomaly_scores': [],
                'dates': [],
                'prices': [],
                'daily_returns': [],
                'trades': [],
                'config': {},
                'vectorbt_stats': {}
            }
    
    def save_results(self, results: Dict, output_dir: str):
        """Save results to files with proper formatting"""
        self.logger.info(f"Saving results to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results as JSON
        results_file = os.path.join(output_dir, 'backtest_results.json')
        
        # Convert numpy types for JSON serialization
        json_results = self._prepare_for_json(results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_file}")
        
        # Save trades as CSV
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            trades_file = os.path.join(output_dir, 'trades.csv')
            trades_df.to_csv(trades_file, index=False)
            self.logger.info(f"Trades saved to {trades_file}")
        
        # Save daily data as CSV for analysis
        daily_df = pd.DataFrame({
            'Date': results['dates'],
            'Price': results['prices'],
            'Anomaly_Score': results['anomaly_scores'],
            'Anomaly_Signal': results['anomaly_signals'],
            'Short_MA': results['short_ma'],
            'Long_MA': results['long_ma'],
            'Portfolio_Value': results['portfolio_values']
        })
        daily_file = os.path.join(output_dir, 'daily_data.csv')
        daily_df.to_csv(daily_file, index=False)
        self.logger.info(f"Daily data saved to {daily_file}")
    
    def _prepare_for_json(self, obj):
        """Recursively prepare object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def plot_results(self, results: Dict, output_dir: str):
        """Generate comprehensive visualization of backtest results with a correctly ordered time axis."""
        self.logger.info("Generating plots...")

        try:
            fig, axes = plt.subplots(5, 1, figsize=(15, 20))
            # 1) Parse dates and sort ascending; reorder all equal-length arrays accordingly
            dates = pd.to_datetime(results['dates'])
            n = len(dates)
            sort_idx = np.argsort(dates.values)  # ascending
            dates_sorted = dates.values[sort_idx]

            def _reorder(x):
                """Reorder a 1D array/list if it matches length n, else return as-is."""
                arr = np.asarray(x)
                return arr[sort_idx] if arr.shape[0] == n else arr

            prices = _reorder(results['prices'])
            anomaly_scores = _reorder(results['anomaly_scores'])
            short_ma = _reorder(results['short_ma'])
            long_ma = _reorder(results['long_ma'])
            anomaly_signals = _reorder(results['anomaly_signals']).astype(int)
            portfolio_values = _reorder(results['portfolio_values'])

            # Some arrays like daily_returns might be computed on n-1 deltas; reorder only if lengths match
            daily_returns = results.get('daily_returns', None)
            if daily_returns is not None:
                daily_returns = _reorder(daily_returns)

            # 2) Common date formatting for all subplots
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)

            # Plot 1: Price with anomaly periods highlighted
            ax1 = axes[0]
            ax1.plot(dates_sorted, prices, label='Price', color='black', linewidth=1.5)

            # Shade contiguous anomaly regions (more legible than per-step shading)
            if anomaly_signals.any():
                changes = np.diff(np.concatenate(([0], anomaly_signals.astype(np.int8), [0])))
                starts = np.where(changes == 1)[0]
                ends = np.where(changes == -1)[0]
                for s, e in zip(starts, ends):
                    # Ensure we don't go out of bounds and shade properly
                    start_idx = min(s, len(dates_sorted) - 1)
                    end_idx = min(e, len(dates_sorted))  # e can be equal to len for the last segment
                    if start_idx < len(dates_sorted) and end_idx > 0:
                        # For shading, we want to include the end date, so we use end_idx-1 but ensure it's valid
                        end_date_idx = max(0, min(end_idx - 1, len(dates_sorted) - 1))
                        ax1.axvspan(dates_sorted[start_idx], dates_sorted[end_date_idx], alpha=0.3, color='red')

            ax1.set_title('Stock Price with Anomaly Periods', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_locator(locator)
            ax1.xaxis.set_major_formatter(formatter)

            # Plot 2: Daily anomaly scores with moving averages
            ax2 = axes[1]
            ax2.plot(dates_sorted, anomaly_scores, label='Daily Anomaly Score', alpha=0.5, color='gray', linewidth=0.5)
            ax2.plot(dates_sorted, short_ma, label=f'Short MA ({self.config.short_ma_period})', color='blue', linewidth=1.5)
            ax2.plot(dates_sorted, long_ma, label=f'Long MA ({self.config.long_ma_period})', color='orange', linewidth=1.5)
            ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
            ax2.set_title('Anomaly Detection Components', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Anomaly Score')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_locator(locator)
            ax2.xaxis.set_major_formatter(formatter)

            # Plot 3: Portfolio value comparison
            ax3 = axes[2]
            initial_price = prices[0]
            buy_hold_values = self.config.initial_capital * (prices / initial_price)
            ax3.plot(dates_sorted, portfolio_values, label='Anomaly Strategy', color='green', linewidth=2)
            ax3.plot(dates_sorted, buy_hold_values, label='Buy & Hold', color='blue', linewidth=2, alpha=0.7)
            ax3.set_title('Portfolio Value Comparison', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Portfolio Value ($)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_locator(locator)
            ax3.xaxis.set_major_formatter(formatter)

            # Annotate final values with better positioning
            if len(portfolio_values) > 0 and len(buy_hold_values) > 0:
                ax3.text(dates_sorted[-1], portfolio_values[-1],
                        f'${portfolio_values[-1]:,.0f}',
                        ha='right', va='bottom', fontweight='bold', color='green')
                ax3.text(dates_sorted[-1], buy_hold_values[-1],
                        f'${buy_hold_values[-1]:,.0f}',
                        ha='right', va='top', fontweight='bold', color='blue')

            # Plot 4: Daily returns distribution
            ax4 = axes[3]
            if isinstance(daily_returns, (list, np.ndarray)) and len(daily_returns) > 0:
                dr = np.asarray(daily_returns) * 100.0
                # Filter out any NaN or infinite values
                dr = dr[np.isfinite(dr)]
                if len(dr) > 0:
                    positive_returns = dr[dr >= 0]
                    negative_returns = dr[dr < 0]
                    
                    # Use appropriate number of bins based on data size
                    n_bins = min(30, max(10, len(dr) // 10))
                    
                    if len(positive_returns) > 0:
                        ax4.hist(positive_returns, bins=n_bins, alpha=0.7, color='green',
                                label=f'Positive ({len(positive_returns)} days)')
                    if len(negative_returns) > 0:
                        ax4.hist(negative_returns, bins=n_bins, alpha=0.7, color='red',
                                label=f'Negative ({len(negative_returns)} days)')
                    
                    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
                    ax4.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
                    ax4.set_xlabel('Daily Return (%)')
                    ax4.set_ylabel('Frequency')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'No valid daily returns data', 
                            ha='center', va='center', transform=ax4.transAxes)
            else:
                ax4.text(0.5, 0.5, 'No daily returns data available', 
                        ha='center', va='center', transform=ax4.transAxes)

            # Plot 5: Drawdown analysis
            ax5 = axes[4]
            if len(portfolio_values) > 0:
                peak = np.maximum.accumulate(portfolio_values)
                drawdown = (portfolio_values - peak) / peak * 100.0
                ax5.fill_between(dates_sorted, drawdown, 0, alpha=0.3, color='red')
                ax5.plot(dates_sorted, drawdown, color='red', linewidth=1)

                # Mark maximum drawdown
                max_dd_idx = int(np.argmin(drawdown))
                ax5.plot(dates_sorted[max_dd_idx], drawdown[max_dd_idx], 'ro', markersize=8)
                ax5.text(dates_sorted[max_dd_idx], drawdown[max_dd_idx],
                        f'Max DD: {drawdown[max_dd_idx]:.1f}%',
                        ha='right', va='top', fontweight='bold')

                ax5.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
                ax5.set_ylabel('Drawdown (%)')
                ax5.set_xlabel('Date')
                ax5.grid(True, alpha=0.3)
                ax5.xaxis.set_major_locator(locator)
                ax5.xaxis.set_major_formatter(formatter)
            else:
                ax5.text(0.5, 0.5, 'No portfolio data available', 
                        ha='center', va='center', transform=ax5.transAxes)

            # 3) Keep all subplots aligned on a forward-in-time x-axis
            if len(dates_sorted) > 0:
                xmin, xmax = dates_sorted[0], dates_sorted[-1]
                for ax in axes:
                    ax.set_xlim(xmin, xmax)

            plt.tight_layout()

            # Save plot
            os.makedirs(output_dir, exist_ok=True)
            plot_file = os.path.join(output_dir, 'backtest_plots.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Plots saved to {plot_file}")

        except Exception as e:
            self.logger.error(f"Error generating plots: {str(e)}")
            plt.close('all')

    def print_performance_summary(self, results: Dict):
        """Print performance summary"""
        summary = f"""
{'='*60}
ANOMALY DETECTION BACKTEST PERFORMANCE SUMMARY
{'='*60}
Strategy Return:      {results['total_return']:.2f}%
Buy & Hold Return:    {results['buy_hold_return']:.2f}%
Excess Return:        {results['excess_return']:.2f}%
Sharpe Ratio:         {results['sharpe_ratio']:.3f}
Calmar Ratio:         {results['calmar_ratio']:.3f}
Maximum Drawdown:     {results['max_drawdown']:.2f}%
Win Rate:             {results['win_rate']:.1f}%
Number of Trades:     {results['num_trades']}

Parameters:
Short MA Period:      {self.config.short_ma_period}
Long MA Period:       {self.config.long_ma_period}
Lookback Window:      {self.config.lookback_window}
Transaction Cost:     {self.config.transaction_cost*100:.1f}%
{'='*60}"""
        
        print(summary)
        self.logger.info(f"Performance: Return={results['total_return']:.2f}%, "
                        f"Sharpe={results['sharpe_ratio']:.3f}, "
                        f"Drawdown={results['max_drawdown']:.2f}%")

class OptunaBayesianOptimizer:
    """Bayesian optimization using Optuna for hyperparameter tuning"""
    
    def __init__(self, config: BacktestConfig, anomaly_detector, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.study = None
        self.anomaly_detector = anomaly_detector
        
        # Initialize MPI if available
        if MPI_AVAILABLE:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.logger.info(f"MPI initialized: rank {self.rank}/{self.size}")
        else:
            self.rank = 0
            self.size = 1
            self.logger.info("MPI not available, running in single process mode")
    
    def objective(self, trial, price_data: pd.DataFrame) -> float:
        """Objective function for Optuna optimization"""
        
        # Suggest hyperparameters
        short_ma = trial.suggest_int('short_ma_period', 
                                   self.config.short_ma_min, 
                                   self.config.short_ma_max)
        long_ma = trial.suggest_int('long_ma_period', 
                                  max(short_ma + 1, self.config.long_ma_min), 
                                  self.config.long_ma_max)
        
        # Create temporary config
        temp_config = BacktestConfig(
            short_ma_period=short_ma,
            long_ma_period=long_ma,
            lookback_window=self.config.lookback_window,
            initial_capital=self.config.initial_capital,
            transaction_cost=self.config.transaction_cost,
            model_path=self.config.model_path,
            output_dir=self.config.output_dir,
            log_level=self.config.log_level,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            train_start_date=self.config.train_start_date,
            train_end_date=self.config.train_end_date,
            freq=self.config.freq
        )
        
        try:
            # Run backtest
            backtester = VectorBTBacktester(temp_config, self.anomaly_detector, self.logger)
            results = backtester.backtest(price_data)
            
            # Optimization objective: maximize risk-adjusted return
            sharpe_ratio = results['sharpe_ratio']
            
            objective_score = float(sharpe_ratio) if not pd.isna(sharpe_ratio) else -1e-5

            # Report for pruning
            trial.report(objective_score, step=0)
            
            self.logger.debug(f"Trial {trial.number}: Short={short_ma}, Long={long_ma}, "
                            f"Lookback={temp_config.lookback_window}, Objective={objective_score:.4f}")
            
            return objective_score
            
        except Exception as e:
            self.logger.error(f"Error in trial {trial.number}: {str(e)}")
            return -10.0
    
    def optimize_parameters(self, price_data: pd.DataFrame) -> Dict:
        """Run Bayesian optimization using Optuna"""
        self.logger.info("Starting Bayesian optimization with Optuna...")
        
        # Create study
        study_name = f"anomaly_backtest_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        sampler = TPESampler(
            n_startup_trials=20,
            n_ei_candidates=24,
            seed=42
        )
        
        pruner = MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=3
        )
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )
        
        # Optimize
        n_trials = self.config.n_trials // self.size if MPI_AVAILABLE else self.config.n_trials
        self.logger.info(f"Process {self.rank} running {n_trials} trials")
        
        try:
            self.study.optimize(
                lambda trial: self.objective(trial, price_data),
                n_trials=n_trials,
                timeout=None,
                n_jobs=1,
                show_progress_bar=True if self.rank == 0 else False
            )
            
        except KeyboardInterrupt:
            self.logger.info("Optimization interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            raise
        
        # Gather results from MPI processes
        if MPI_AVAILABLE and self.size > 1:
            all_trials = self.comm.gather(self.study.trials, root=0)
            
            if self.rank == 0:
                combined_study = optuna.create_study(direction='maximize')
                for process_trials in all_trials:
                    for trial in process_trials:
                        if trial.state == optuna.trial.TrialState.COMPLETE:
                            combined_study.add_trial(trial)
                
                best_trial = combined_study.best_trial
                best_params = best_trial.params
                best_value = best_trial.value
            else:
                best_params = None
                best_value = None
                combined_study = None
        else:
            best_params = self.study.best_params
            best_value = self.study.best_value
            combined_study = self.study
        
        if self.rank == 0:
            self.logger.info(f"Optimization completed. Best trial:")
            self.logger.info(f"  Value: {best_value:.4f}")
            self.logger.info(f"  Params: {best_params}")
            
            # Save optimization results
            optimization_results = {
                'best_params': best_params,
                'best_value': best_value,
                'n_trials': len(combined_study.trials),
                'study_name': study_name,
                'optimization_history': [
                    {
                        'trial_number': trial.number,
                        'value': trial.value,
                        'params': trial.params,
                        'state': str(trial.state)
                    } for trial in combined_study.trials
                ]
            }
            
            return optimization_results
        else:
            return {'best_params': None, 'best_value': None, 'optimization_history': []}
    
    def plot_optimization_history(self, output_dir: str):
        """Plot optimization history and parameter importance"""
        if self.study is None or self.rank != 0:
            return
            
        try:
            import optuna.visualization as vis
            
            # Create plots
            fig1 = vis.plot_optimization_history(self.study)
            fig1.write_html(os.path.join(output_dir, 'optimization_history.html'))
            
            fig2 = vis.plot_param_importances(self.study)
            fig2.write_html(os.path.join(output_dir, 'param_importances.html'))
            
            fig3 = vis.plot_parallel_coordinate(self.study)
            fig3.write_html(os.path.join(output_dir, 'parallel_coordinate.html'))
            
            self.logger.info(f"Optimization plots saved to {output_dir}")
            
        except ImportError:
            self.logger.warning("Plotly not available for optimization visualization")
        except Exception as e:
            self.logger.error(f"Error creating optimization plots: {str(e)}")

def load_config(config_file: str) -> BacktestConfig:
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    return BacktestConfig(**config_dict)

def save_config(config: BacktestConfig, config_file: str):
    """Save configuration to JSON file"""
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)

def _coerce_date_column(df: pd.DataFrame, col: str = 'date') -> pd.DataFrame:
    """Coerce date column to datetime with various format handling"""
    ser = df[col]
    if np.issubdtype(ser.dtype, np.number):
        s = ser.astype('int64').abs().median()
        # Infer epoch unit by magnitude
        if s >= 1e15:
            unit = 'ns'
        elif s >= 1e12:
            unit = 'ms'
        else:
            unit = 's'
        df[col] = pd.to_datetime(df[col].astype('int64'), unit=unit, origin='unix', utc=True).dt.tz_localize(None)
    else:
        # Try fast parse; if many NaT, retry with day-first
        parsed = pd.to_datetime(ser, errors='coerce', utc=True, infer_datetime_format=True)
        if parsed.isna().mean() > 0.5:
            parsed = pd.to_datetime(ser, errors='coerce', utc=True, dayfirst=True)
        df[col] = parsed.dt.tz_localize(None)
    return df

def main():
    """Main execution function with Optuna integration and VectorBT backtesting"""
    
    parser = argparse.ArgumentParser(description='Financial Anomaly Detection Backtesting with Optuna and VectorBT')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--input', type=str, help='Input data file (CSV)')
    parser.add_argument('--output', type=str, default='./results', help='Output directory')
    parser.add_argument('--model-path', type=str, default='./models', help='GAN model path')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    # Time period parameters
    parser.add_argument('--start-date', type=str, help='Test start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Test end date (YYYY-MM-DD)')
    parser.add_argument('--train-start-date', type=str, help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--train-end-date', type=str, help='Training end date (YYYY-MM-DD)')
    
    # Optimization parameters
    parser.add_argument('--optimize', action='store_true', help='Run Bayesian parameter optimization')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of optimization trials')
    
    # Parameter ranges for optimization
    parser.add_argument('--short-ma-min', type=int, default=2, help='Minimum short MA period')
    parser.add_argument('--short-ma-max', type=int, default=60, help='Maximum short MA period')
    parser.add_argument('--long-ma-min', type=int, default=20, help='Minimum long MA period')
    parser.add_argument('--long-ma-max', type=int, default=300, help='Maximum long MA period')
    
    # Manual parameter override (if not optimizing)
    parser.add_argument('--short-ma', type=int, default=10, help='Short moving average period')
    parser.add_argument('--long-ma', type=int, default=50, help='Long moving average period')
    parser.add_argument('--lookback', type=int, default=20, help='Lookback window for anomaly detection')
    
    # Trading parameters
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--transaction-cost', type=float, default=0.005, help='Transaction cost (0.005 = 0.5%)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.output, f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = Logger.setup_logger('backtester', log_file, args.log_level)
    
    logger.info("="*60)
    logger.info("FINANCIAL ANOMALY DETECTION BACKTESTING WITH OPTUNA & VECTORBT")
    logger.info("="*60)
    logger.info(f"Command line args: {vars(args)}")
    
    try:
        # Load or create configuration
        if args.config and os.path.exists(args.config):
            config = load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            config = BacktestConfig(
                short_ma_period=args.short_ma,
                long_ma_period=args.long_ma,
                lookback_window=args.lookback,
                initial_capital=args.initial_capital,
                transaction_cost=args.transaction_cost,
                start_date=args.start_date,
                end_date=args.end_date,
                train_start_date=args.train_start_date,
                train_end_date=args.train_end_date,
                output_dir=args.output,
                model_path=args.model_path,
                log_level=args.log_level,
                optimize_params=args.optimize,
                n_trials=args.n_trials,
                short_ma_min=args.short_ma_min,
                short_ma_max=args.short_ma_max,
                long_ma_min=args.long_ma_min,
                long_ma_max=args.long_ma_max,
            )
            logger.info("Using default configuration")
        
        # Override config with command line arguments
        config.output_dir = args.output
        config.log_level = args.log_level
        config.model_path = args.model_path
        config.optimize_params = args.optimize
        config.n_trials = args.n_trials
        config.initial_capital = args.initial_capital
        config.transaction_cost = args.transaction_cost
        
        # Override time periods if provided
        if args.start_date:
            config.start_date = args.start_date
        if args.end_date:
            config.end_date = args.end_date
        if args.train_start_date:
            config.train_start_date = args.train_start_date
        if args.train_end_date:
            config.train_end_date = args.train_end_date
            
        if args.input:
            config.input_file = args.input
        
        # Log configuration
        logger.info(f"Configuration: Short MA={config.short_ma_period}, Long MA={config.long_ma_period}, "
                   f"Lookback={config.lookback_window}, TX Cost={config.transaction_cost*100:.1f}%")
        
        if config.start_date or config.end_date:
            logger.info(f"Test period: {config.start_date or 'beginning'} to {config.end_date or 'end'}")
        if config.train_start_date or config.train_end_date:
            logger.info(f"Training period: {config.train_start_date or 'beginning'} to {config.train_end_date or 'auto'}")
        
        # Save current configuration
        config_file = os.path.join(args.output, 'config.json')
        save_config(config, config_file)
        logger.info(f"Configuration saved to {config_file}")

        # Load lookback window
        model_params_path = os.path.join(args.model_path, 'best_params.json')
        with open(model_params_path, "r", encoding="utf-8") as f:
            model_params = json.load(f)
        config.lookback_window = model_params["sequence_length"]
        
        # Load price data
        if not config.input_file:
            logger.error("No input file specified. Use --input")
            sys.exit(1)
        
        if not os.path.exists(config.input_file):
            logger.error(f"Input file not found: {config.input_file}")
            sys.exit(1)
        
        logger.info(f"Loading price data from {config.input_file}")
        price_data = pd.read_csv(config.input_file)
        price_data = _coerce_date_column(price_data)
        price_data = price_data.sort_values('date')
        price_data = price_data[["date", "adj_close", "log_adj_close"]]

        logger.info(f"Loaded {len(price_data)} data points from {price_data['date'].min()} to {price_data['date'].max()}")
        
        # Validate minimum data requirements
        min_required_points = config.lookback_window + config.long_ma_max + 50  # Buffer for analysis
        if len(price_data) < min_required_points:
            logger.warning(f"Data has only {len(price_data)} points, but {min_required_points} recommended for robust optimization")
        
        anomaly_detector = AnomalyDetector(
            config.model_path, 
            config.lookback_window,
            config.start_date,
            config.end_date,
            logger
        )

        if config.optimize_params:
            # Run Bayesian optimization with Optuna
            logger.info("Starting Bayesian optimization with Optuna...")
            optimizer = OptunaBayesianOptimizer(config, anomaly_detector, logger)
            optimization_results = optimizer.optimize_parameters(price_data)
            
            if optimizer.rank == 0:  # Only master process saves results
                # Save optimization results
                opt_file = os.path.join(args.output, 'optimization_results.json')
                with open(opt_file, 'w') as f:
                    json.dump(optimization_results, f, indent=2, default=str)
                logger.info(f"Optimization results saved to {opt_file}")
                
                # Create optimization plots
                optimizer.plot_optimization_history(args.output)
                
                # Update config with best parameters
                if optimization_results['best_params']:
                    best_params = optimization_results['best_params']
                    config.short_ma_period = best_params['short_ma_period']
                    config.long_ma_period = best_params['long_ma_period']
                    
                    logger.info(f"Using optimized parameters:")
                    logger.info(f"  Short MA: {config.short_ma_period}")
                    logger.info(f"  Long MA: {config.long_ma_period}")
        
        # Run main backtest (only on master process if using MPI, or always if not using MPI)
        if not config.optimize_params or (MPI_AVAILABLE and MPI.COMM_WORLD.Get_rank() == 0) or not MPI_AVAILABLE:
            logger.info("Running main backtest with VectorBT...")
            backtester = VectorBTBacktester(config, anomaly_detector, logger)
            results = backtester.backtest(price_data)
            
            # Save results
            backtester.save_results(results, config.output_dir)
            
            # Generate plots
            backtester.plot_results(results, config.output_dir)
            
            # Print summary
            backtester.print_performance_summary(results)
            
            logger.info("Backtest completed successfully")
            logger.info(f"Results saved to {config.output_dir}")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        sys.exit(1)
    
    logger.info("="*60)
    logger.info("BACKTESTING COMPLETED SUCCESSFULLY")
    logger.info("="*60)

if __name__ == "__main__":
    main()
