#!/usr/bin/env python3
"""
Common Backtesting Module using VectorBT
Supports multiple data configurations (OHLCAV vs Price-only)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import json
import os
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import date

import vectorbt as vbt


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    # Moving average ranges for optimization
    short_ma_min: int = 5
    short_ma_max: int = 60
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
    model_type: str = "paper_gan_price"  # NEW: Specify model type
    anomaly_threshold_percentile: float = 95.0
    use_full_history: bool = False  # NEW: Use full history for anomaly scoring
    
    # VectorBT configuration
    freq: str = '1D'  # Data frequency
    
    def __post_init__(self):
        # Validate ranges
        if self.short_ma_max >= self.long_ma_min:
            self.long_ma_min = self.short_ma_max + 1


class VectorBTBacktester:
    """Main backtesting engine using VectorBT with proper daily anomaly scoring"""
    
    def __init__(
        self, 
        config: BacktestConfig, 
        anomaly_detector, 
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.anomaly_detector = anomaly_detector
        
        self.logger.info(f"Initialized VectorBTBacktester with config: {asdict(config)}")
    
    def generate_anomaly_signals(
        self, 
        price_data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate anomaly signals with proper daily alignment
        
        Args:
            price_data: DataFrame with price columns (index should be datetime)
            
        Returns:
            Tuple of (anomaly_signals, short_ma, long_ma, anomaly_scores)
        """
        self.logger.info("Generating daily anomaly scores...")
        
        # Get daily anomaly scores
        anomaly_scores = self.anomaly_detector.compute_daily_anomaly_scores(
            price_data,
            use_full_history=self.config.use_full_history
        )
        
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
            self.logger.info(
                f"Anomaly score statistics: mean={valid_scores.mean():.4f}, "
                f"std={valid_scores.std():.4f}, max={valid_scores.max():.4f}"
            )
        
        self.logger.info(
            f"Generated {anomaly_signals.sum()} anomaly signals out of "
            f"{len(anomaly_signals)} days ({anomaly_signals.mean()*100:.2f}%)"
        )
        
        return anomaly_signals, short_ma, long_ma, anomaly_scores_filled
    
    def create_trading_signals(
        self, 
        anomaly_signals: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
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
        prev_signal = anomaly_signals.shift(1)
        
        # Exit when anomaly starts (transition from 0 to 1)
        exits = (prev_signal == 0) & (anomaly_signals == 1)
        
        # Enter when anomaly ends (transition from 1 to 0)
        entries = (prev_signal == 1) & (anomaly_signals == 0)
        
        # Handle first day
        if anomaly_signals.iloc[0] == 0:
            entries.iloc[0] = True  # Enter on first day if no anomaly
        
        self.logger.info(
            f"Generated {entries.sum()} entry signals and {exits.sum()} exit signals"
        )
        
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
            anomaly_signals, short_ma, long_ma, anomaly_scores = self.generate_anomaly_signals(
                test_data
            )
            
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
        """Coerce date column to datetime with various format handling"""
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

        ser = df[col]

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
            dt.loc[mask_num] = pd.to_datetime(
                s.loc[mask_num], unit=unit, utc=True
            ).dt.tz_convert(None)

        # Parse remaining string timestamps
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

        # Drop rows that could not be parsed
        good = dt.notna()
        if (~good).sum() > 0:
            self.logger.warning(
                "Dropping %d row(s) with unparseable timestamps in %s",
                int((~good).sum()), col
            )

        df = df.loc[good].copy()
        df["date"] = dt.loc[good].astype("datetime64[ns]")

        return df

    def _filter_data_by_time_period(
        self,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Filter data based on configured start and end dates (YYYY-MM-DD), inclusive"""
        df = data.copy()

        # Ensure date is datetime64[ns]
        df = self._coerce_date_column(df, 'date')

        # Remove timezone to avoid comparison surprises
        if pd.api.types.is_datetime64tz_dtype(df['date'].dtype):
            df['date'] = df['date'].dt.tz_convert(None)

        # Compare on calendar dates only
        date_only = df['date'].dt.normalize().dt.date

        def _parse_ymd(s: str) -> date:
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
    
    def _calculate_vectorbt_metrics(
        self, 
        pf, 
        price_data: pd.DataFrame, 
        anomaly_signals: pd.Series,
        short_ma: pd.Series, 
        long_ma: pd.Series, 
        anomaly_scores: pd.Series
    ) -> Dict:
        """Calculate comprehensive performance metrics"""
        self.logger.info("Calculating performance metrics...")
        
        try:
            # Basic portfolio metrics
            total_return = pf.total_return() * 100
            
            # Buy and hold comparison
            buy_hold_return = (
                (price_data['adj_close'].iloc[-1] / price_data['adj_close'].iloc[0]) - 1
            ) * 100
            
            # VectorBT statistics
            stats = pf.stats()
            
            # Risk metrics - handle potential NaN values properly
            sharpe_ratio = pf.sharpe_ratio()
            sharpe_ratio = float(sharpe_ratio) if not pd.isna(sharpe_ratio) else 0.0
            
            max_drawdown = pf.max_drawdown() * 100
            max_drawdown = float(max_drawdown) if not pd.isna(max_drawdown) else 0.0
            
            calmar_ratio = pf.calmar_ratio()
            calmar_ratio = float(calmar_ratio) if not pd.isna(calmar_ratio) else 0.0
            
            # Trade statistics
            trades = pf.trades
            try:
                if hasattr(trades, 'records_readable'):
                    records = trades.records_readable
                    has_trades = len(records) > 0 if hasattr(records, '__len__') else False
                else:
                    has_trades = False
                    records = None
                
                if has_trades:
                    if hasattr(trades, 'win_rate'):
                        win_rate_val = trades.win_rate()
                        win_rate = (
                            float(win_rate_val) * 100 
                            if not pd.isna(win_rate_val) else 0
                        )
                    else:
                        win_rate = 0
                    
                    if hasattr(trades, 'winning'):
                        winning_trades = trades.winning
                        if hasattr(winning_trades, 'count'):
                            win_count = (
                                winning_trades.count() 
                                if callable(winning_trades.count) 
                                else winning_trades.count
                            )
                        else:
                            win_count = 0
                        
                        if win_count > 0:
                            avg_win = (
                                float(winning_trades.mean()) 
                                if hasattr(winning_trades, 'mean') else 0
                            )
                        else:
                            avg_win = 0
                    else:
                        avg_win = 0
                        
                    if hasattr(trades, 'losing'):
                        losing_trades = trades.losing
                        if hasattr(losing_trades, 'count'):
                            loss_count = (
                                losing_trades.count() 
                                if callable(losing_trades.count) 
                                else losing_trades.count
                            )
                        else:
                            loss_count = 0
                            
                        if loss_count > 0:
                            avg_loss = (
                                abs(float(losing_trades.mean())) 
                                if hasattr(losing_trades, 'mean') else 0
                            )
                        else:
                            avg_loss = 0
                    else:
                        avg_loss = 0
                        
                    profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
                    num_trades = len(records)
                    trades_records = (
                        records.to_dict('records') 
                        if hasattr(records, 'to_dict') else []
                    )
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
            
            # Calculate daily returns
            portfolio_values = pf.value()
            if hasattr(portfolio_values, 'values'):
                portfolio_values = portfolio_values.values
            
            if len(portfolio_values) > 1:
                daily_returns = []
                for i in range(1, len(portfolio_values)):
                    if portfolio_values[i-1] != 0:
                        ret = (
                            (portfolio_values[i] - portfolio_values[i-1]) / 
                            portfolio_values[i-1]
                        )
                        daily_returns.append(ret)
                    else:
                        daily_returns.append(0.0)
            else:
                daily_returns = []
            
            # Handle date column
            if 'date' in price_data.columns:
                if pd.api.types.is_datetime64_any_dtype(price_data['date']):
                    dates = price_data['date'].dt.strftime('%Y-%m-%d').tolist()
                else:
                    dates = price_data['date'].astype(str).tolist()
            else:
                dates = (
                    price_data.index.strftime('%Y-%m-%d').tolist() 
                    if hasattr(price_data.index, 'strftime') 
                    else list(range(len(price_data)))
                )
            
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
                'excess_return': (
                    float(total_return - buy_hold_return) 
                    if not (pd.isna(total_return) or pd.isna(buy_hold_return)) 
                    else 0.0
                ),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': float(win_rate) if not pd.isna(win_rate) else 0.0,
                'profit_factor': (
                    float(profit_factor) 
                    if profit_factor != float('inf') else 999.99
                ),
                'num_trades': num_trades,
                'portfolio_values': (
                    portfolio_values.tolist() 
                    if hasattr(portfolio_values, 'tolist') 
                    else list(portfolio_values)
                ),
                'anomaly_signals': (
                    anomaly_signals.values.tolist() 
                    if hasattr(anomaly_signals, 'values') 
                    else anomaly_signals.tolist()
                ),
                'short_ma': (
                    short_ma.values.tolist() 
                    if hasattr(short_ma, 'values') 
                    else short_ma.tolist()
                ),
                'long_ma': (
                    long_ma.values.tolist() 
                    if hasattr(long_ma, 'values') 
                    else long_ma.tolist()
                ),
                'anomaly_scores': (
                    anomaly_scores.values.tolist() 
                    if hasattr(anomaly_scores, 'values') 
                    else anomaly_scores.tolist()
                ),
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
            'date': results['dates'],
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
        """Generate comprehensive visualization of backtest results"""
        self.logger.info("Generating plots...")

        try:
            fig, axes = plt.subplots(5, 1, figsize=(15, 20))
            
            # Parse dates and sort ascending
            dates = pd.to_datetime(results['dates'])
            n = len(dates)
            sort_idx = np.argsort(dates.values)
            dates_sorted = dates.values[sort_idx]

            def _reorder(x):
                arr = np.asarray(x)
                return arr[sort_idx] if arr.shape[0] == n else arr

            prices = _reorder(results['prices'])
            anomaly_scores = _reorder(results['anomaly_scores'])
            short_ma = _reorder(results['short_ma'])
            long_ma = _reorder(results['long_ma'])
            anomaly_signals = _reorder(results['anomaly_signals']).astype(int)
            portfolio_values = _reorder(results['portfolio_values'])

            daily_returns = results.get('daily_returns', None)
            if daily_returns is not None:
                daily_returns = _reorder(daily_returns)

            # Common date formatting
            locator = mdates.AutodateLocator()
            formatter = mdates.ConcisedateFormatter(locator)

            # Plot 1: Price with anomaly periods
            ax1 = axes[0]
            ax1.plot(dates_sorted, prices, label='Price', color='black', linewidth=1.5)

            if anomaly_signals.any():
                changes = np.diff(np.concatenate(([0], anomaly_signals.astype(np.int8), [0])))
                starts = np.where(changes == 1)[0]
                ends = np.where(changes == -1)[0]
                for s, e in zip(starts, ends):
                    start_idx = min(s, len(dates_sorted) - 1)
                    end_idx = min(e, len(dates_sorted))
                    if start_idx < len(dates_sorted) and end_idx > 0:
                        end_date_idx = max(0, min(end_idx - 1, len(dates_sorted) - 1))
                        ax1.axvspan(
                            dates_sorted[start_idx], 
                            dates_sorted[end_date_idx], 
                            alpha=0.3, 
                            color='red'
                        )

            ax1.set_title('Stock Price with Anomaly Periods', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_locator(locator)
            ax1.xaxis.set_major_formatter(formatter)

            # Plot 2: Daily anomaly scores with moving averages
            ax2 = axes[1]
            ax2.plot(
                dates_sorted, anomaly_scores, 
                label='Daily Anomaly Score', 
                alpha=0.5, color='gray', linewidth=0.5
            )
            ax2.plot(
                dates_sorted, short_ma, 
                label=f'Short MA ({self.config.short_ma_period})', 
                color='blue', linewidth=1.5
            )
            ax2.plot(
                dates_sorted, long_ma, 
                label=f'Long MA ({self.config.long_ma_period})', 
                color='orange', linewidth=1.5
            )
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
            ax3.plot(
                dates_sorted, portfolio_values, 
                label='Anomaly Strategy', 
                color='green', linewidth=2
            )
            ax3.plot(
                dates_sorted, buy_hold_values, 
                label='Buy & Hold', 
                color='blue', linewidth=2, alpha=0.7
            )
            ax3.set_title('Portfolio Value Comparison', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Portfolio Value ($)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_locator(locator)
            ax3.xaxis.set_major_formatter(formatter)

            if len(portfolio_values) > 0 and len(buy_hold_values) > 0:
                ax3.text(
                    dates_sorted[-1], portfolio_values[-1],
                    f'${portfolio_values[-1]:,.0f}',
                    ha='right', va='bottom', fontweight='bold', color='green'
                )
                ax3.text(
                    dates_sorted[-1], buy_hold_values[-1],
                    f'${buy_hold_values[-1]:,.0f}',
                    ha='right', va='top', fontweight='bold', color='blue'
                )

            # Plot 4: Daily returns distribution
            ax4 = axes[3]
            if isinstance(daily_returns, (list, np.ndarray)) and len(daily_returns) > 0:
                dr = np.asarray(daily_returns) * 100.0
                dr = dr[np.isfinite(dr)]
                if len(dr) > 0:
                    positive_returns = dr[dr >= 0]
                    negative_returns = dr[dr < 0]
                    
                    n_bins = min(30, max(10, len(dr) // 10))
                    
                    if len(positive_returns) > 0:
                        ax4.hist(
                            positive_returns, bins=n_bins, alpha=0.7, color='green',
                            label=f'Positive ({len(positive_returns)} days)'
                        )
                    if len(negative_returns) > 0:
                        ax4.hist(
                            negative_returns, bins=n_bins, alpha=0.7, color='red',
                            label=f'Negative ({len(negative_returns)} days)'
                        )
                    
                    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
                    ax4.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
                    ax4.set_xlabel('Daily Return (%)')
                    ax4.set_ylabel('Frequency')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(
                        0.5, 0.5, 'No valid daily returns data', 
                        ha='center', va='center', transform=ax4.transAxes
                    )
            else:
                ax4.text(
                    0.5, 0.5, 'No daily returns data available', 
                    ha='center', va='center', transform=ax4.transAxes
                )

            # Plot 5: Drawdown analysis
            ax5 = axes[4]
            if len(portfolio_values) > 0:
                peak = np.maximum.accumulate(portfolio_values)
                drawdown = (portfolio_values - peak) / peak * 100.0
                ax5.fill_between(dates_sorted, drawdown, 0, alpha=0.3, color='red')
                ax5.plot(dates_sorted, drawdown, color='red', linewidth=1)

                max_dd_idx = int(np.argmin(drawdown))
                ax5.plot(
                    dates_sorted[max_dd_idx], drawdown[max_dd_idx], 
                    'ro', markersize=8
                )
                ax5.text(
                    dates_sorted[max_dd_idx], drawdown[max_dd_idx],
                    f'Max DD: {drawdown[max_dd_idx]:.1f}%',
                    ha='right', va='top', fontweight='bold'
                )

                ax5.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
                ax5.set_ylabel('Drawdown (%)')
                ax5.set_xlabel('date')
                ax5.grid(True, alpha=0.3)
                ax5.xaxis.set_major_locator(locator)
                ax5.xaxis.set_major_formatter(formatter)
            else:
                ax5.text(
                    0.5, 0.5, 'No portfolio data available', 
                    ha='center', va='center', transform=ax5.transAxes
                )

            # Keep all subplots aligned on a forward-in-time x-axis
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
        self.logger.info(
            f"Performance: Return={results['total_return']:.2f}%, "
            f"Sharpe={results['sharpe_ratio']:.3f}, "
            f"Drawdown={results['max_drawdown']:.2f}%"
        )