from dataclasses import asdict
import json
import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import warnings
import logging
import sys
import os
import argparse
from datetime import datetime
import torch
warnings.filterwarnings('ignore')

try:
    from ..paper_gan_price.gan_model import GANExpert
except ImportError:
    GANExpert = None

try:
    from ..lstm_cnn_parallel_price.lstm_cnn_gan_model_parallel import LSTMCNNGANExpert
except ImportError:
    LSTMCNNGANExpert = None

try:
    from ..lstm_cnn_seq_price.lstm_cnn_gan_model_sequential import LSTMCNNGANExpert as LSTMCNNGANSeqExpert
except ImportError:
    LSTMCNNGANSeqExpert = None

try:
    from ..cnn_lstm_seq_price.cnn_lstm_gan_model_sequential import LSTMCNNGANExpert as CNNSLSTMGANSeqExpert
except ImportError:
    CNNSLSTMGANSeqExpert = None

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

class HedgeAlgorithm:
    """Hedge algorithm with multiplicative weight updates and regret tracking."""
    
    def __init__(
        self, 
        experts, 
        short_window: int = 5, 
        long_window: int = 20,
        learning_rate: float = None,
        max_loss: float = 1.0,
        start: str = None,
        end: str = None,
        logger: Logger = None,
        anomaly_scores = None,
        max_iters: int = 50,
        weights: list = None,
        initial_capital: float = 100000
    ):
        """
        Initialize hedge algorithm.
        
        Args:
            experts: List of expert models
            short_window: Window for short moving average
            long_window: Window for long moving average
            learning_rate: Learning rate (eta). If None, uses optimal eta = sqrt(2*ln(N)/T)
            max_loss: Maximum loss per round (for normalization)
            start: Start date for backtesting (YYYY-MM-DD)
            end: End date for backtesting (YYYY-MM-DD)
        """
        self.experts = experts
        self.n_experts = len(experts)
        self.weights = np.ones(self.n_experts) / self.n_experts if weights is None else np.array(weights)
        self.short_window = short_window
        self.long_window = long_window
        self.learning_rate = learning_rate  # Will be set adaptively if None
        self.max_loss = max_loss
        self.start = pd.to_datetime(start) if start else None
        self.end = pd.to_datetime(end) if end else None
        self.logger = logger or logging.getLogger(__name__)
        self.cache = [anomaly_scores] if isinstance(anomaly_scores, pd.Series) else [None for i in range(self.n_experts)]
        self.max_iters = max_iters
        self.initial_capital = initial_capital

        # Tracking for regret calculation
        self.cumulative_loss = 0.0
        self.expert_cumulative_losses = np.zeros(self.n_experts)
        self.weight_history = [self.weights.copy()]
        self.regret_history = []
        self.iteration = 0

        # Time end periods for iterative updates
        self.time_periods = []

        # Track continuous portfolio state
        self.last_period_end = None  # Track last processed date
        self.cumulative_signals = pd.Series(dtype=int)  # All signals across periods
        self.current_position = 0  # Track if we're in a position (0 or 1)
        
        # NEW: Track all diagnostics across periods
        self.all_diagnostics = pd.DataFrame()

    def _split_time_periods(self, _data: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Split the time period into equal portions based on max_iters.

        Args:
            data: Full DataFrame with time index

        Returns:
            List of (start, end) timestamp tuples for each period
        """
        data = _data.copy()
        start_bound = data.index[0] if self.start is None else pd.Timestamp(self.start).normalize()
        end_bound = data.index[-1] if self.end is None else pd.Timestamp(self.end).normalize()
        
        # Filter data to the specified range
        data_in_range = data.loc[start_bound:end_bound]
        
        # Calculate period boundaries
        total_days = len(data_in_range)

        if total_days == 0:
            self.logger.warning("No data in selected range; returning empty periods.")
            return []

        num_periods = min(self.max_iters, total_days)
        days_per_period = total_days // num_periods

        if days_per_period < 1:
            self.logger.warning(f"Not enough days ({total_days}) for {num_periods} iterations. Using 1 day per period.")
            days_per_period = 1

        idx_positions = np.linspace(0, total_days - 1, num_periods, dtype=int)
        periods = data_in_range.index[idx_positions].tolist()
        
        periods = sorted(pd.Index(periods).unique())
        self.logger.info(f"Split {total_days} days into {len(periods)} periods of ~{days_per_period} days each")
        self.time_periods = periods

        return periods

    def _compute_anomaly_scores_full(self, prices: pd.DataFrame) -> None:
        """
        Compute and cache anomaly scores for all experts over the full time period.
        This is done once at the beginning to avoid recomputation.
        
        Args:
            prices: DataFrame with price and feature data
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
        
        for i, expert in enumerate(self.experts):
            cached = self.cache[i]

            if cached is not None:
                self.logger.info(f"[Cache Hit] Expert {i+1}: Using cached anomaly scores")
                continue
            
            seq_len = expert.sequence_length
            self.logger.info(f"Computing anomaly scores for Expert {i+1}/{self.n_experts} with sequence length {seq_len}...")
            
            # Prepare data with lookback for sequence
            curr_feature_data = feature_data.copy()
            
            if self.start is not None:
                start_bound = pd.Timestamp(self.start).normalize() - pd.Timedelta(days=int(2*seq_len))
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
            
            self.logger.info(f"Using {n_days} days of data for Expert {i+1} from {start_bound} to {end_bound}")
            
            anomaly_scores = np.zeros(n_days)
            
            # Prepare windows for batch processing
            windows = []
            valid_indices = []
            for t in range(seq_len, n_days):
                window = curr_feature_data_np[t - seq_len:t]
                windows.append(window)
                valid_indices.append(t)
            
            if len(windows) > 0:
                windows_array = np.array(windows).reshape(len(windows), seq_len, n_features)
                
                # Get anomaly scores for all windows
                scores = self._predict_anomaly_batch(expert, windows_array)
                
                # Assign scores to corresponding days
                for idx, score in zip(valid_indices, scores):
                    anomaly_scores[idx] = score
            
            # Create series aligned with feature_data index
            # Take only the scores that correspond to the original data (without lookback)
            aligned_scores = anomaly_scores[-num_days:]
            anomaly_scores_series = pd.Series(aligned_scores, index=price_data.index)
            
            # Cache the scores
            self.cache[i] = anomaly_scores_series.copy()
            self.logger.info(f"[Cached] Expert {i+1}: Computed and cached {len(anomaly_scores_series)} anomaly scores")

    def _predict_anomaly_batch(self, expert, windows: np.ndarray) -> np.ndarray:
        """
        Predict anomaly probabilities for batch of windows
        
        Args:
            windows: Array of shape (batch_size, lookback_window, features)
            
        Returns:
            Array of anomaly probabilities of shape (batch_size,)
        """
        try:
            batch_size = windows.shape[0]
            probabilities = []

            self.logger.info(f"Going through a total of {batch_size} windows")
            
            # Process in smaller batches to avoid memory issues
            batch_chunk_size = min(1024, batch_size)
            
            for i in range(0, batch_size, batch_chunk_size):
                self.logger.info(f"Processing batch {i // batch_chunk_size + 1} / {int(np.ceil(batch_size / batch_chunk_size))}...")

                end_idx = min(i + batch_chunk_size, batch_size)
                chunk = windows[i:end_idx]
                
                # Get anomaly detection results
                if hasattr(expert, 'detect_financial_anomalies'):
                    results = expert.detect_financial_anomalies(
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
                    self.logger.error("Expert does not have detect_financial_anomalies method, falling back to manual calculation")
                    scores = []
                    for window in chunk:
                        processed_data = expert._preprocess_financial_data(window.reshape(1, -1, 1))
                        data_tensor = torch.FloatTensor(processed_data).to(expert.device)
                        
                        with torch.no_grad():
                            score = expert._evaluate_tanogan_loss(data_tensor).item()
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
        
    def compute_weighted_anomaly_scores(self, period_start: pd.Timestamp, period_end: pd.Timestamp) -> pd.Series:
        """
        Compute weighted anomaly scores from all experts for a specific time period.
        
        Args:
            period_start: Start of the period
            period_end: End of the period
            
        Returns:
            Weighted anomaly scores for the period
        """
        all_scores = pd.DataFrame()
        
        for i in range(self.n_experts):
            cached = self.cache[i]

            if cached is None:
                raise ValueError(f"Anomaly scores not cached for expert {i}. Call _compute_anomaly_scores_full first.")
            
            # Extract scores for this period
            expert_scores = cached.loc[period_start:period_end]
            all_scores[f"expert_{i}"] = expert_scores
        
        # Compute weighted sum using current weights
        # THIS IS WHERE ANOMALY SCORES ARE USED WITH WEIGHTS
        weighted_scores = (all_scores * self.weights).sum(axis=1)
        
        return weighted_scores

    def generate_signals(self, period_start: pd.Timestamp, period_end: pd.Timestamp) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Generate trading signals based on anomaly detection for a specific period.
        
        Args:
            period_start: Start of the period
            period_end: End of the period
            
        Returns:
            Tuple of (signals, diagnostics dataframe)
        """
        # Compute weighted anomaly scores for this period
        # THIS IS THE KEY LINE - weighted anomaly scores are computed here
        anomaly_scores = self.compute_weighted_anomaly_scores(period_start, period_end)
        
        # Calculate moving averages OF THE WEIGHTED ANOMALY SCORES
        # This is correct - we're using the anomaly scores to generate signals
        short_ma = anomaly_scores.rolling(window=self.short_window, min_periods=1).mean()
        long_ma = anomaly_scores.rolling(window=self.long_window, min_periods=1).mean()
        
        # Generate signals: 1 = long position, 0 = no position
        # Exit (0) when short MA > long MA (anomaly detected)
        # Enter (1) when short MA <= long MA (no anomaly)
        signals = (short_ma <= long_ma).astype(int)
        
        # Create diagnostics dataframe
        diagnostics = pd.DataFrame({
            'anomaly_score': anomaly_scores,
            'short_ma': short_ma,
            'long_ma': long_ma,
            'signal': signals
        })
        
        return signals, diagnostics
    
    def backtest_period(
        self, 
        _data: pd.DataFrame,
        period_end: pd.Timestamp,
        trading_cost: float = 0.005,
        initial_capital: float = 100000.0
    ) -> Dict:
        """
        Backtest the strategy for a specific time period.
        Maintains continuous positions by accumulating signals across periods.
        """
        data = _data.copy()
        
        # Determine period boundaries
        if self.last_period_end is None:
            # First period
            period_start = data.index[0] if self.start is None else pd.Timestamp(self.start).normalize()
        else:
            # Continue from where we left off
            period_start = self.last_period_end
        
        # Generate signals for this period only
        period_signals, diagnostics = self.generate_signals(period_start, period_end)
        
        # FIXED: Store diagnostics incrementally
        self.all_diagnostics = pd.concat([self.all_diagnostics, diagnostics])
        # Remove duplicates, keeping the latest
        self.all_diagnostics = self.all_diagnostics[~self.all_diagnostics.index.duplicated(keep='last')]
        self.all_diagnostics.sort_index(inplace=True)
        
        # Merge with cumulative signals
        if len(self.cumulative_signals) > 0:
            # Keep previous signals, append new ones
            self.cumulative_signals = pd.concat([
                self.cumulative_signals[self.cumulative_signals.index < period_start],
                period_signals
            ])
        else:
            self.cumulative_signals = period_signals
        
        # FIXED: Get price data only up to period_end (no future data)
        start_date = data.index[0] if self.start is None else pd.Timestamp(self.start).normalize()
        price_data = data.loc[start_date:period_end].copy()
        
        # Align signals with price data
        aligned_signals = self.cumulative_signals.reindex(price_data.index, method='ffill', fill_value=0)
        
        # Simply create portfolio from cumulative signals
        # VectorBT will handle position continuity automatically
        entries = aligned_signals.astype(bool)
        exits = ~aligned_signals.astype(bool)
        
        portfolio = vbt.Portfolio.from_signals(
            price_data["adj_close"],
            entries=entries,
            exits=exits,
            init_cash=initial_capital,
            fees=trading_cost,
            freq='1D'
        )
        
        # Update position state
        if len(aligned_signals) > 0:
            self.current_position = int(aligned_signals.iloc[-1])
        
        # Store last period end for next iteration
        self.last_period_end = period_end
        
        # Calculate buy and hold benchmark for comparison (full period)
        buy_hold_portfolio = vbt.Portfolio.from_holding(
            price_data["adj_close"],
            init_cash=initial_capital,
            fees=trading_cost,
            freq='1D'
        )
        
        # Get trade statistics
        num_trades = 0
        win_rate = 0
        
        try:
            if hasattr(portfolio, 'trades') and hasattr(portfolio.trades, 'records_readable'):
                trades = portfolio.trades.records_readable
                if trades is not None and len(trades) > 0:
                    num_trades = len(trades)
                    winning_trades = len(trades[trades['PnL'] > 0])
                    win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
        except:
            num_trades = 0
            win_rate = 0

        # Use accumulated diagnostics for metrics
        metric_results = self._calculate_vectorbt_metrics(
            portfolio, price_data, aligned_signals, 
            self.all_diagnostics['short_ma'], 
            self.all_diagnostics['long_ma'], 
            self.all_diagnostics['anomaly_score']
        )

        results = {
            'portfolio': portfolio,
            'buy_hold_portfolio': buy_hold_portfolio,
            'diagnostics': diagnostics,  # This period's diagnostics
            'total_return': portfolio.total_return() * 100,
            'buy_hold_return': buy_hold_portfolio.total_return() * 100,
            'sharpe_ratio': portfolio.sharpe_ratio(),
            'max_drawdown': portfolio.max_drawdown() * 100,
            'total_trades': num_trades,
            'win_rate': win_rate,
            'final_value': portfolio.final_value(),
            'period_start': period_start,
            'period_end': period_end,
            'metrics': metric_results,
            'current_position': self.current_position
        }
        
        return results
    
    def compute_regret_bound(self, T: int) -> float:
        """
        Compute theoretical regret bound for hedge algorithm.
        
        For hedge algorithm with optimal learning rate:
        Regret <= sqrt(2 * T * ln(N))
        
        Args:
            T: Number of rounds (iterations)
            
        Returns:
            Theoretical regret bound
        """
        return np.sqrt(2 * T * np.log(self.n_experts))
    
    def compute_current_regret(self) -> float:
        """
        Compute current regret: difference between algorithm loss and best expert loss.
        
        Returns:
            Current regret value
        """
        best_expert_loss = np.min(self.expert_cumulative_losses)
        regret = self.cumulative_loss - best_expert_loss
        return regret
    
    def update_weights(self, data: pd.DataFrame, period_end: pd.Timestamp, 
                    prev_period_end: pd.Timestamp, trading_cost: float = 0.005) -> Dict:
        """
        Update expert weights using multiplicative weight update rule.
        Returns dictionary with expert returns, losses, and new weights.
        """
        self.iteration += 1

        # Learning rate for this round (eta_t)
        if self.learning_rate is None:
            eta = np.sqrt(2.0 * np.log(self.n_experts) / max(1, self.iteration))
        else:
            eta = float(self.learning_rate)

        expert_returns = []
        expert_drawdowns = []

        # Compute each expert's standalone performance for this period
        for i in range(self.n_experts):
            # Save current state
            saved_last_end = self.last_period_end
            saved_signals = self.cumulative_signals.copy()
            saved_position = self.current_position
            saved_diagnostics = self.all_diagnostics.copy()
            
            # Temporarily set weights to give full weight to this expert
            temp_weights = self.weights.copy()
            self.weights = np.zeros(self.n_experts)
            self.weights[i] = 1.0
            self.last_period_end = None
            self.cumulative_signals = pd.Series(dtype=int)
            self.current_position = 0
            self.all_diagnostics = pd.DataFrame()
            
            # Run backtest for THIS PERIOD continuing from previous state
            res = self.backtest_period(data, period_end, trading_cost)
            
            # Calculate PERIOD return (not cumulative)
            portfolio = res['portfolio']
            price_data = data.loc[:period_end]
            
            # Get portfolio values
            portfolio_values = portfolio.value()
            
            if prev_period_end in portfolio_values.index:
                prev_value = portfolio_values.loc[prev_period_end]
                curr_value = portfolio_values.iloc[-1]
                period_return = ((curr_value - prev_value) / prev_value) * 100
            else:
                # First period
                period_return = res['total_return']
            
            # Restore state
            self.last_period_end = saved_last_end
            self.cumulative_signals = saved_signals
            self.current_position = saved_position
            self.all_diagnostics = saved_diagnostics
            self.weights = temp_weights
            
            expert_returns.append(float(period_return))
            expert_drawdowns.append(float(res.get('max_drawdown', 0.0)))

        expert_returns = np.asarray(expert_returns, dtype=float)
        expert_drawdowns = np.asarray(expert_drawdowns, dtype=float)

        # Robust bounded loss in [0, 1]
        r_min = float(np.min(expert_returns))
        r_max = float(np.max(expert_returns))
        span = max(1e-12, r_max - r_min)
        expert_losses = (r_max - expert_returns) / span
        expert_losses = np.clip(expert_losses, 0.0, 1.0) * float(self.max_loss)

        # Degeneracy handling (all losses equal)
        if np.allclose(expert_losses, expert_losses[0], atol=1e-12, rtol=0.0):
            dd_min, dd_max = float(np.min(expert_drawdowns)), float(np.max(expert_drawdowns))
            dd_span = max(1e-12, dd_max - dd_min)
            dd_loss = (expert_drawdowns - dd_min) / dd_span
            dd_loss = np.clip(dd_loss, 0.0, 1.0) * float(self.max_loss)

            if not np.allclose(dd_loss, dd_loss[0], atol=1e-12, rtol=0.0):
                expert_losses = 0.9 * expert_losses + 0.1 * dd_loss
            else:
                jitter = 1e-6 * (np.arange(self.n_experts) - (self.n_experts - 1) / 2.0)
                expert_losses = expert_losses + jitter

        # Update experts' cumulative losses
        self.expert_cumulative_losses += expert_losses

        # Compute algorithm loss with PRE-UPDATE weights
        algorithm_loss = float(np.dot(self.weights, expert_losses))
        self.cumulative_loss += algorithm_loss

        # Multiplicative weights update
        self.weights = self.weights * np.exp(-eta * expert_losses)
        s = float(self.weights.sum())
        if s <= 0 or not np.isfinite(s):
            self.weights = np.ones_like(self.weights) / len(self.weights)
        else:
            self.weights = self.weights / s

        # Track regret
        current_regret = self.compute_current_regret()
        regret_bound = self.compute_regret_bound(self.iteration)

        # History
        self.weight_history.append(self.weights.copy())
        self.regret_history.append({
            'iteration': self.iteration,
            'regret': current_regret,
            'bound': regret_bound,
            'learning_rate': eta
        })

        self.logger.info(f"eta={eta:.6g}, period_returns={np.array2string(expert_returns, precision=2)}%, "
                        f"losses={np.array2string(expert_losses, precision=6)}, "
                        f"weights={np.array2string(self.weights, precision=8)}")

        return {
            'expert_returns': expert_returns,
            'expert_losses': expert_losses,
            'new_weights': self.weights.copy(),
            'learning_rate': eta,
            'algorithm_loss': algorithm_loss,
            'cumulative_loss': self.cumulative_loss,
            'expert_cumulative_losses': self.expert_cumulative_losses.copy(),
            'regret': current_regret,
            'regret_bound': regret_bound,
            'regret_ratio': (current_regret / regret_bound) if regret_bound > 0 else 0.0
        }
    
    def has_reached_regret_bound(self, threshold: float = 0.95) -> bool:
        """
        Check if current regret has reached the theoretical bound.
        
        Args:
            threshold: Fraction of bound to consider as "reached" (default 0.95 = 95%)
            
        Returns:
            True if regret >= threshold * bound
        """
        if self.iteration == 0:
            return False
        
        current_regret = self.compute_current_regret()
        regret_bound = self.compute_regret_bound(self.iteration)
        
        return current_regret >= threshold * regret_bound
    
    def plot_results(self, prices: pd.Series, results: Dict, figsize=(15, 12)):
        """
        Plot backtest results including regret analysis.
        
        Args:
            prices: Series of price data
            results: Results dictionary from backtest
            figsize: Figure size
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # Plot 1: Portfolio value vs Buy & Hold
        portfolio_value = results['portfolio'].value()
        buy_hold_value = results['buy_hold_portfolio'].value()
        
        axes[0].plot(portfolio_value.index, portfolio_value.values, 
                    label=f"Strategy (Return: {results['total_return']:.2f}%)", 
                    linewidth=2, color='blue')
        axes[0].plot(buy_hold_value.index, buy_hold_value.values, 
                    label=f"Buy & Hold (Return: {results['buy_hold_return']:.2f}%)", 
                    linewidth=2, color='gray', alpha=0.7)
        axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Price and signals
        # FIXED: Use all_diagnostics instead of single period
        diagnostics = self.all_diagnostics
        axes[1].plot(prices.index, prices.values, label='Price', color='black', linewidth=1.5)
        
        # Highlight periods with position
        in_position = (
            diagnostics['signal']
            .reindex(prices.index)
            .eq(1)
            .fillna(False)
            .to_numpy()
        )
        axes[1].fill_between(prices.index, prices.min(), prices.max(), 
                             where=in_position, alpha=0.2, color='green', 
                             label='Long Position')
        
        axes[1].set_title('Price and Trading Signals', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Price')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Anomaly scores and moving averages
        axes[2].plot(diagnostics.index, diagnostics['anomaly_score'], 
                    label='Anomaly Score', color='lightgray', linewidth=1)
        axes[2].plot(diagnostics.index, diagnostics['short_ma'], 
                    label=f'Short MA ({self.short_window})', color='orange', linewidth=2)
        axes[2].plot(diagnostics.index, diagnostics['long_ma'], 
                    label=f'Long MA ({self.long_window})', color='green', linewidth=2)
        axes[2].set_title('Anomaly Scores and Moving Averages', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Anomaly Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Regret vs Theoretical Bound
        if len(self.regret_history) > 0:
            iterations = [h['iteration'] for h in self.regret_history]
            regrets = [h['regret'] for h in self.regret_history]
            bounds = [h['bound'] for h in self.regret_history]
            
            axes[3].plot(iterations, regrets, label='Actual Regret', 
                        color='red', linewidth=2, marker='o', markersize=4)
            axes[3].plot(iterations, bounds, label='Theoretical Bound', 
                        color='blue', linewidth=2, linestyle='--')
            axes[3].fill_between(iterations, 0, bounds, alpha=0.1, color='blue')
            axes[3].set_title('Regret Analysis', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('Iteration')
            axes[3].set_ylabel('Regret')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

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
            
            # FIXED: Align all series to price_data index
            price_index = price_data.index
            
            # Reindex all series to match price_data
            aligned_signals = anomaly_signals.reindex(price_index, method='ffill', fill_value=0)
            aligned_short_ma = short_ma.reindex(price_index, method='ffill', fill_value=0)
            aligned_long_ma = long_ma.reindex(price_index, method='ffill', fill_value=0)
            aligned_scores = anomaly_scores.reindex(price_index, method='ffill', fill_value=0)
            
            # Handle date column - check if it exists and is datetime
            if 'date' in price_data.columns:
                if pd.api.types.is_datetime64_any_dtype(price_data['date']):
                    dates = price_data['date'].dt.strftime('%Y-%m-%d').tolist()
                else:
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
            
            # Align portfolio values to price data length
            pf_values = pf.value().reindex(price_index, method='ffill')
            
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
                'portfolio_values': pf_values.values.tolist() if hasattr(pf_values, 'values') else pf_values.tolist(),
                'anomaly_signals': aligned_signals.values.tolist(),
                'short_ma': aligned_short_ma.values.tolist(),
                'long_ma': aligned_long_ma.values.tolist(),
                'anomaly_scores': aligned_scores.values.tolist(),
                'dates': dates,
                'prices': price_data['adj_close'].values.tolist(),
                'daily_returns': daily_returns,
                'trades': trades_records,
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
                'vectorbt_stats': {}
            }

def _coerce_date_column(df: pd.DataFrame, logger, col: str = 'date') -> pd.DataFrame:
    """
    Coerce df[col] to pandas datetime, supporting:
    1) Epoch (seconds) – numeric dtype or digit-like strings (default assumption).
    2) ISO dates in 'YYYY-MM-DD' string format.

    Returns tz-naive datetime64[ns] in df[col].
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")

    ser = df[col]
    out = None

    # Helper: decide epoch unit by magnitude (fallback safety; default is seconds)
    def _epoch_unit_from_magnitude(values: pd.Series) -> str:
        arr = pd.to_numeric(values, errors='coerce').astype('float64')
        if np.all(np.isnan(arr)):
            return 's'
        med = float(np.nanmedian(np.abs(arr)))
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

    mask_str = ~mask_num
    string_date_format = '%Y-%m-%d'
    if mask_str.any():
        dt2 = pd.to_datetime(
            ser.loc[mask_str],
            format=string_date_format,
            errors="coerce",
            utc=True,
        )
        dt.loc[mask_str] = dt2.dt.tz_convert(None)

    good = dt.notna()
    if not good.any():
        hint = " Provide --string-date-format if your file mixes epoch and formatted strings." \
            if string_date_format is None else ""
    else:
        hint = ""

    if (~good).sum() > 0:
        logger.warning("Dropping %d row(s) with unparseable timestamps in %s.%s",
                            int((~good).sum()), col, hint)

    df = df.loc[good].copy()
    df["date"] = dt.loc[good].astype("datetime64[ns]")

    return df

def _prepare_for_json(obj):
    """Recursively prepare object for JSON serialization"""
    if isinstance(obj, dict):
        return {k: _prepare_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_prepare_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

def main():
    """Main function to demonstrate the hedge algorithm with regret bound."""
    parser = argparse.ArgumentParser(description="Hedge Algorithm with Regret Bound Tracking")
    parser.add_argument('--input', type=str, help='Input CSV file with price data')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--paper-gan-model', type=str, help='Path to pre-trained GAN model')
    parser.add_argument('--lstm-cnn-gan-model', type=str, help='Path to pre-trained LSTM CNN GAN model')
    parser.add_argument('--lstm-cnn-gan-seq-model', type=str, help='Path to pre-trained LSTM CNN GAN Sequential model')
    parser.add_argument('--cnn-lstm-gan-seq-model', type=str, help='Path to pre-trained CNN LSTM GAN Sequential model')
    parser.add_argument('--output', type=str, default='logs', help='Output directory for logs and results')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--max-iterations', type=int, default=100, help='Maximum number of iterations for weight updates')
    parser.add_argument('--regret-threshold', type=float, default=0.99, help='Regret bound threshold to stop (fraction of theoretical bound)')
    parser.add_argument('--short-window', type=int, default=5, help='Short moving average window size')
    parser.add_argument('--long-window', type=int, default=20, help='Long moving average window size')
    parser.add_argument('--trading-cost', type=float, default=0.005, help='Trading cost as a fraction (e.g., 0.005 = 0.5%)')
    parser.add_argument('--initial-capital', type=float, default=100000.0, help='Initial capital for backtesting')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--lr', type=float, default=None, help='Fixed learning rate (eta). If not set, uses adaptive learning rate.')
    parser.add_argument('--max-loss', type=float, default=1.0, help='Maximum loss per round for normalization')
    parser.add_argument('--paper-gan-weight', type=float, default=0.25, help="Initial weight to pre-trained GAN model")
    parser.add_argument('--lstm-cnn-gan-weight', type=float, default=0.25, help="Initial weight to pre-trained LSTM CNN GAN model")
    parser.add_argument('--lstm-cnn-gan-seq-weight', type=float, default=0.25, help="Initial weight to pre-trained LSTM CNN GAN Sequential model")
    parser.add_argument('--cnn-lstm-gan-seq-weight', type=float, default=0.25, help="Initial weight to pre-trained CNN LSTM GAN Sequential model")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Setup logging
    log_file = os.path.join(args.output, f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = Logger.setup_logger('backtester', log_file, args.log_level)

    logger.info("=" * 50)
    logger.info("HEDGE ALGORITHM WITH REGRET BOUND TRACKING")
    logger.info("=" * 50)
    logger.info("")

    # Configuration
    max_iterations = args.max_iterations
    regret_threshold = args.regret_threshold

    # Step 1: Load data
    logger.info("Step 1: Loading price data...")
    data = pd.read_csv(args.input)
    data = _coerce_date_column(data, logger)
    data.sort_values('date', inplace=True)
    data = data[["date", "log_adj_close", "adj_close"]].set_index("date")

    logger.info(f"Loaded {len(data)} rows from {args.input}")
    
    # Step 2: Create expert models
    logger.info("Step 2: Loading expert models...")
    try:
        if GANExpert is None:
            logger.warning("GANExpert class not found. Ensure gan_model.py is available.")
        else:
            gan_expert = GANExpert()
            if args.paper_gan_model:
                gan_expert.load_models(args.paper_gan_model)
                logger.info(f"Loaded GAN expert model from {args.paper_gan_model}.")
            else:
                gan_expert = None
                logger.warning("No GAN expert model provided.")
    except Exception as e:
        logger.error(f"Error loading GAN expert model: {e}")
        gan_expert = None

    try:
        if LSTMCNNGANExpert is None:
            logger.warning("LSTMCNNGANExpert class not found. Ensure lstm_cnn_gan_model.py is available.")
        else:
            lstm_cnn_gan_expert = LSTMCNNGANExpert()
            if args.lstm_cnn_gan_model:
                lstm_cnn_gan_expert.load_models(args.lstm_cnn_gan_model)
                logger.info(f"Loaded LSTM CNN GAN expert model from {args.lstm_cnn_gan_model}.")
            else:
                lstm_cnn_gan_expert = None
                logger.warning("No LSTM CNN GAN expert model provided.")
    except Exception as e:
        logger.error(f"Error loading LSTM CNN GAN expert model: {e}")
        lstm_cnn_gan_expert = None

    try:
        if LSTMCNNGANSeqExpert is None:
            logger.warning("LSTMCNNGANSeqExpert class not found. Ensure lstm_cnn_gan_model_sequential.py is available.")
        else:
            lstm_cnn_gan_seq_expert = LSTMCNNGANSeqExpert()
            if args.lstm_cnn_gan_seq_model:
                lstm_cnn_gan_seq_expert.load_models(args.lstm_cnn_gan_seq_model)
                logger.info(f"Loaded LSTM CNN GAN Sequential expert model from {args.lstm_cnn_gan_seq_model}.")
            else:
                lstm_cnn_gan_seq_expert = None
                logger.warning("No LSTM CNN GAN Sequential expert model provided.")
    except Exception as e:
        logger.error(f"Error loading LSTM CNN GAN Sequential expert model: {e}")
        lstm_cnn_gan_seq_expert = None

    try:
        if CNNSLSTMGANSeqExpert is None:
            logger.warning("CNNSLSTMGANSeqExpert class not found. Ensure cnn_lstm_gan_model_sequential.py is available.")
        else:
            cnn_lstm_gan_seq_expert = CNNSLSTMGANSeqExpert()
            if args.cnn_lstm_gan_seq_model:
                cnn_lstm_gan_seq_expert.load_models(args.cnn_lstm_gan_seq_model)
                logger.info(f"Loaded CNN LSTM GAN Sequential expert model from {args.cnn_lstm_gan_seq_model}.")
            else:
                cnn_lstm_gan_seq_expert = None
                logger.warning("No CNN LSTM GAN Sequential expert model provided.")
    except Exception as e:
        logger.error(f"Error loading CNN LSTM GAN Sequential expert model: {e}")
        cnn_lstm_gan_seq_expert = None

    experts = [e for e in [gan_expert, lstm_cnn_gan_expert, lstm_cnn_gan_seq_expert, cnn_lstm_gan_seq_expert] if e is not None]
    if len(experts) == 0:
        logger.error("No expert models loaded. Exiting.")
        return
    else:
        logger.info(f"Loaded {len(experts)} expert models.")

    normalized_weights = np.array([args.paper_gan_weight, args.lstm_cnn_gan_weight, args.lstm_cnn_gan_seq_weight, args.cnn_lstm_gan_seq_weight])
    normalized_weights = normalized_weights/np.sum(normalized_weights)
    
    weights = [normalized_weights[i] for i, e in enumerate([gan_expert, lstm_cnn_gan_expert, lstm_cnn_gan_seq_expert, cnn_lstm_gan_seq_expert]) if e is not None]

    # Step 3: Initialize hedge algorithm
    logger.info("Step 3: Initializing hedge algorithm...")
    hedge = HedgeAlgorithm(
        experts=experts,
        short_window=args.short_window,
        long_window=args.long_window,
        learning_rate=args.lr,
        max_loss=args.max_loss,
        start=args.start,
        end=args.end,
        logger=logger,
        max_iters=max_iterations,
        weights=weights,
    )
    logger.info(f"Initial weights: {hedge.weights}")
    logger.info(f"Using adaptive learning rate: eta(t) = sqrt(2*ln(N)/t)")
    logger.info("")

    # Step 4: Split time periods
    logger.info("Step 4: Splitting time periods for online learning...")
    periods = hedge._split_time_periods(data)
    logger.info(f"Created {len(periods)} time periods")
    logger.info("")

    # Step 5: Pre-compute all anomaly scores
    logger.info("Step 5: Pre-computing anomaly scores for all experts...")
    logger.info("This is done once at the beginning to enable efficient caching")
    
    anomaly_test_data = data.copy()
    anomaly_test_data.drop(columns=['adj_close'], inplace=True, errors='ignore')
    
    hedge._compute_anomaly_scores_full(anomaly_test_data)
    logger.info("Anomaly score computation complete. Cached for all experts.")
    logger.info("")
    
    # Step 6: Run online MWU updates over time periods
    logger.info("Step 6: Run online MWU updates over time periods")
    logger.info("=" * 50)
    logger.info("ONLINE WEIGHT UPDATES - TRACKING REGRET")
    logger.info("=" * 50)
    logger.info("")
    
    iteration_results = []
    prev_period_end = None

    for i, period_end in enumerate(periods):
        if prev_period_end is None:
            prev_period_end = data.index[0] if args.start is None else pd.Timestamp(args.start).normalize()
        
        logger.info(f"Iteration {i + 1}/{len(periods)}:")
        logger.info(f"Period: {prev_period_end.date()} to {period_end.date()}")
        logger.info("-" * 60)

        # Run backtest with current weights for this period
        results = hedge.backtest_period(
            data, period_end, 
            trading_cost=args.trading_cost, 
            initial_capital=args.initial_capital
        )

        # Update weights based on this period's performance
        weight_update = hedge.update_weights(
            data, period_end, prev_period_end,
            trading_cost=args.trading_cost
        )

        # Store results
        iteration_results.append({
            'iteration': i + 1,
            'period_start': results['period_start'],
            'period_end': period_end,
            'results': results,
            'weight_update': weight_update
        })
        
        # Display iteration summary
        logger.info(f"  Period Return:        {weight_update['expert_returns'].mean():>8.2f}%")
        logger.info(f"  Cumulative Return:    {results['total_return']:>8.2f}%")
        logger.info(f"  Current Position:     {results['current_position']}")
        logger.info(f"  Learning Rate (eta):  {weight_update['learning_rate']:>8.4f}")
        logger.info(f"  Current Regret:       {weight_update['regret']:>8.4f}")
        logger.info(f"  New Weights:          " + ", ".join([f"{w:.2%}" for w in weight_update['new_weights']]))
        logger.info(f"  Regret Bound:         {weight_update['regret_bound']:>8.4f}")
        logger.info(f"  Regret Ratio:         {weight_update['regret_ratio']:>8.2%}")
        logger.info(f"  Best Expert Loss:     {np.min(weight_update['expert_cumulative_losses']):>8.4f}")
        logger.info("")

        prev_period_end = period_end
        
        # Check if regret bound is reached
        if hedge.has_reached_regret_bound(threshold=regret_threshold):
            logger.info("=" * 50)
            logger.info(f"REGRET BOUND REACHED! (>= {regret_threshold*100:.0f}% of theoretical bound)")
            logger.info("=" * 50)
            logger.info("")
            break

    # FIXED: Use all_diagnostics from hedge object (already compiled correctly)
    final_diagnostics = hedge.all_diagnostics.copy()

    # Save all iterations of weights
    with open(os.path.join(args.output, 'weights_history.json'), 'w') as f:
        json.dump(_prepare_for_json(hedge.weight_history), f, indent=4)

    # Final results
    final_results = iteration_results[-1]['results']
    final_weights = iteration_results[-1]['weight_update']

    logger.info("=" * 50)
    logger.info("FINAL RESULTS")
    logger.info("=" * 50)
    logger.info("")

    # Save results to JSON files
    with open(os.path.join(args.output, 'final_weights.json'), 'w') as f:
        json.dump({
            "Final Expert Weights": [
                {"Expert": i + 1, "Weight": float(weight), "Cumulative Loss": float(cum_loss)}
                for i, (weight, cum_loss) in enumerate(zip(hedge.weights, hedge.expert_cumulative_losses))
            ]
        }, f, indent=4)

    with open(os.path.join(args.output, 'overall_results.json'), 'w') as f:
        json.dump({
            "Overall Backtest Results": {
                "Total Iterations": int(hedge.iteration),
                "Total Return": float(final_results['total_return']),
                "Buy & Hold Return": float(final_results['buy_hold_return']),
                "Sharpe Ratio": float(final_results['sharpe_ratio']),
                "Max Drawdown": float(final_results['max_drawdown']),
                "Total Trades": int(final_results['total_trades']),
                "Win Rate": float(final_results['win_rate']),
                "Final Portfolio Value": float(final_results['final_value'])
            },
            "Final Expert Weights": [
                {"Expert": i + 1, "Weight": float(weight), "Cumulative Loss": float(cum_loss)}
                for i, (weight, cum_loss) in enumerate(zip(hedge.weights, hedge.expert_cumulative_losses))
            ],
            "Regret Analysis": {
                "Final Regret": float(final_weights['regret']),
                "Theoretical Bound": float(final_weights['regret_bound']),
                "Regret / Bound Ratio": float(final_weights['regret_ratio'])
            }
        }, f, indent=4)

    with open(os.path.join(args.output, 'iteration_details.json'), 'w') as f:
        json.dump({
            "Iteration Details": [
                {
                    "Iteration": int(res['iteration']),
                    "Strategy Return": float(res['results']['total_return']),
                    "Learning Rate (eta)": float(res['weight_update']['learning_rate']),
                    "Current Regret": float(res['weight_update']['regret']),
                    "Regret Bound": float(res['weight_update']['regret_bound']),
                    "Regret Ratio": float(res['weight_update']['regret_ratio']),
                    "Best Expert Loss": float(np.min(res['weight_update']['expert_cumulative_losses']))
                }
                for res in iteration_results
            ]
        }, f, indent=4)

    logger.info(f"Total Iterations:      {hedge.iteration}")
    logger.info(f"Total Return:          {final_results['total_return']:>10.2f}%")
    logger.info(f"Buy & Hold Return:     {final_results['buy_hold_return']:>10.2f}%")
    logger.info(f"Sharpe Ratio:          {final_results['sharpe_ratio']:>10.2f}")
    logger.info(f"Max Drawdown:          {final_results['max_drawdown']:>10.2f}%")
    logger.info(f"Total Trades:          {final_results['total_trades']:>10.0f}")
    logger.info(f"Win Rate:              {final_results['win_rate']:>10.2f}%")
    logger.info(f"Final Portfolio Value: ${final_results['final_value']:>10.2f}")
    logger.info("")
    
    logger.info("Final Expert Weights:")
    logger.info("-" * 60)
    for i, (weight, cum_loss) in enumerate(zip(final_weights['new_weights'], 
                                                final_weights['expert_cumulative_losses'])):
        logger.info(f"Expert {i+1}: Weight = {weight:>6.2%}  |  Cumulative Loss = {cum_loss:>8.4f}")
    logger.info("-" * 60)
    logger.info("")

    logger.info("Regret Analysis:")
    logger.info("-" * 60)
    logger.info(f"Final Regret:          {final_weights['regret']:>10.4f}")
    logger.info(f"Theoretical Bound:     {final_weights['regret_bound']:>10.4f}")
    logger.info(f"Regret / Bound Ratio:  {final_weights['regret_ratio']:>10.2%}")
    logger.info("-" * 60)
    logger.info("")

    # Plot results
    logger.info("Generating plots...")
    start_bound_plot = data.index[0] if args.start is None else pd.Timestamp(args.start).normalize()
    end_bound_plot = data.index[-1] if args.end is None else pd.Timestamp(args.end).normalize()
    plot_data = data.loc[start_bound_plot:end_bound_plot]
    hedge.plot_results(plot_data['adj_close'], final_results)
    plot_file = os.path.join(args.output, 'online_mwu_results.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved as '{plot_file}'")
    logger.info("")

    # Plot weight evolution
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Weight evolution
    weight_history = np.array(hedge.weight_history)
    iterations = list(range(len(weight_history)))
    
    for i in range(hedge.n_experts):
        ax1.plot(iterations, weight_history[:, i], label=f'Expert {i+1}', 
                marker='o', markersize=3, linewidth=2)
    
    ax1.set_title('Expert Weight Evolution Over Time Periods (Online MWU)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Weight')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Learning rate evolution
    if len(hedge.regret_history) > 0:
        iterations = [h['iteration'] for h in hedge.regret_history]
        learning_rates = [h['learning_rate'] for h in hedge.regret_history]
        
        ax2.plot(iterations, learning_rates, color='purple', linewidth=2, marker='o', markersize=4)
        ax2.set_title('Adaptive Learning Rate Over Iterations', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Learning Rate (eta)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(args.output, 'online_mwu_weight_evolution.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Weight evolution plot saved as '{plot_file}'")
    logger.info("")

    # FIXED: Save correctly aligned results
    # Get the metrics from the final iteration (already properly aligned)
    final_metrics = final_results['metrics']
    
    # Save portfolio results
    results_file = os.path.join(args.output, 'backtest_results_Portfolio.json')
    json_results = _prepare_for_json(final_metrics)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
        
    logger.info(f"Results saved to {results_file}")
        
    # Save trades as CSV
    if final_metrics['trades']:
        trades_df = pd.DataFrame(final_metrics['trades'])
        trades_file = os.path.join(args.output, 'trades_Portfolio.csv')
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"Trades saved to {trades_file}")
        
    # FIXED: Save daily data with proper alignment
    # All arrays should now have the same length since we aligned them in _calculate_vectorbt_metrics
    try:
        daily_df = pd.DataFrame({
            'Date': final_metrics['dates'],
            'Price': final_metrics['prices'],
            'Anomaly_Score': final_metrics['anomaly_scores'],
            'Anomaly_Signal': final_metrics['anomaly_signals'],
            'Short_MA': final_metrics['short_ma'],
            'Long_MA': final_metrics['long_ma'],
            'Portfolio_Value': final_metrics['portfolio_values']
        })
        daily_file = os.path.join(args.output, 'daily_data_Portfolio.csv')
        daily_df.to_csv(daily_file, index=False)
        logger.info(f"Daily data saved to {daily_file}")
    except Exception as e:
        logger.error(f"Error saving daily data: {str(e)}")
        logger.error(f"Array lengths - Dates: {len(final_metrics['dates'])}, "
                    f"Prices: {len(final_metrics['prices'])}, "
                    f"Anomaly Scores: {len(final_metrics['anomaly_scores'])}, "
                    f"Signals: {len(final_metrics['anomaly_signals'])}, "
                    f"Short MA: {len(final_metrics['short_ma'])}, "
                    f"Long MA: {len(final_metrics['long_ma'])}, "
                    f"Portfolio Values: {len(final_metrics['portfolio_values'])}")

    logger.info("=" * 50)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 50)
    
    # Return results for further analysis if needed
    return hedge, iteration_results


if __name__ == "__main__":
    hedge, iteration_results = main()