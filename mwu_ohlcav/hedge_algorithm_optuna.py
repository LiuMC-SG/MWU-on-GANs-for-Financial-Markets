# optuna_tune_windows.py
# ------------------------------------------------------------
# Hyperparameter tuning for (short_window, long_window) using
# Optuna with Sharpe ratio as the objective.
#
# Usage (example):
#   python optuna_tune_windows.py \
#       --input path/to/data.csv \
#       --start 2018-01-01 --end 2024-12-31 \
#       --paper-gan-model models/paper_gan/ \
#       --lstm-cnn-gan-model models/lstm_cnn_gan/ \
#       --n-trials 50 --study-name "mwu_ma_tuning" \
#       --storage sqlite:///mwu_ma_tuning.db
# ------------------------------------------------------------

import os
import json
import argparse
import numpy as np
import pandas as pd
import optuna
import optuna.visualization as vis
import matplotlib.pyplot as plt
from datetime import datetime

# Reuse your existing implementation
from hedge_algorithm_python import (
    HedgeAlgorithm,
    _coerce_date_column,
    _prepare_for_json,
    Logger
)

# -----------------------------
# Utility: load experts (mirrors your main())
# -----------------------------
def load_experts_and_weights(args, logger):
    """Load expert models based on provided paths."""
    try:
        from ..paper_gan_ohlcav.gan_model import GANExpert
    except ImportError:
        GANExpert = None
        logger.warning("GANExpert class not found. Ensure gan_model.py is available.")

    try:
        from ..lstm_cnn_parallel_ohlcav.lstm_cnn_gan_model_parallel import LSTMCNNGANExpert
    except ImportError:
        LSTMCNNGANExpert = None
        logger.warning("LSTMCNNGANExpert class not found. Ensure lstm_cnn_gan_model.py is available.")

    try:
        from ..lstm_cnn_seq_ohlcav.lstm_cnn_gan_model_sequential import LSTMCNNGANExpert as LSTMCNNGANSeqExpert
    except ImportError:
        LSTMCNNGANSeqExpert = None
        logger.warning("LSTMCNNGANSeqExpert class not found. Ensure lstm_cnn_gan_model_sequential.py is available.")

    try:
        from ..cnn_lstm_seq_ohlcav.cnn_lstm_gan_model_sequential import LSTMCNNGANExpert as CNNSLSTMGANSeqExpert
    except ImportError:
        CNNSLSTMGANSeqExpert = None
        logger.warning("CNNSLSTMGANSeqExpert class not found. Ensure cnn_lstm_gan_model_sequential.py is available.")

    experts = []
    provided_weights = []

    # Paper GAN
    if GANExpert is not None and args.paper_gan_model:
        try:
            gan_expert = GANExpert()
            gan_expert.load_models(args.paper_gan_model)
            logger.info(f"Loaded GAN expert model from {args.paper_gan_model}.")
            experts.append(gan_expert)
            provided_weights.append(args.paper_gan_weight)
        except Exception as e:
            logger.error(f"Error loading GAN expert: {e}")

    # LSTM-CNN-GAN
    if LSTMCNNGANExpert is not None and args.lstm_cnn_gan_model:
        try:
            lstm_cnn_gan_expert = LSTMCNNGANExpert()
            lstm_cnn_gan_expert.load_models(args.lstm_cnn_gan_model)
            logger.info(f"Loaded LSTM CNN GAN expert model from {args.lstm_cnn_gan_model}.")
            experts.append(lstm_cnn_gan_expert)
            provided_weights.append(args.lstm_cnn_gan_weight)
        except Exception as e:
            logger.error(f"Error loading LSTM CNN GAN expert: {e}")

    # LSTM-CNN-GAN Sequential
    if LSTMCNNGANSeqExpert is not None and args.lstm_cnn_gan_seq_model:
        try:
            lstm_cnn_gan_seq_expert = LSTMCNNGANSeqExpert()
            lstm_cnn_gan_seq_expert.load_models(args.lstm_cnn_gan_seq_model)
            logger.info(f"Loaded LSTM CNN GAN Sequential expert model from {args.lstm_cnn_gan_seq_model}.")
            experts.append(lstm_cnn_gan_seq_expert)
            provided_weights.append(args.lstm_cnn_gan_seq_weight)
        except Exception as e:
            logger.error(f"Error loading LSTM CNN GAN Sequential expert: {e}")

    # CNN-LSTM-GAN Sequential
    if CNNSLSTMGANSeqExpert is not None and args.cnn_lstm_gan_seq_model:
        try:
            cnn_lstm_gan_seq_expert = CNNSLSTMGANSeqExpert()
            cnn_lstm_gan_seq_expert.load_models(args.cnn_lstm_gan_seq_model)
            logger.info(f"Loaded CNN LSTM GAN Sequential expert model from {args.cnn_lstm_gan_seq_model}.")
            experts.append(cnn_lstm_gan_seq_expert)
            provided_weights.append(args.cnn_lstm_gan_seq_weight)
        except Exception as e:
            logger.error(f"Error loading CNN LSTM GAN Sequential expert: {e}")

    if len(experts) == 0:
        raise RuntimeError("No expert models were loaded. Please provide at least one model path.")

    # Normalize weights
    w = np.array(provided_weights, dtype=float)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(len(experts), dtype=float) / float(len(experts))
    else:
        w = w / w.sum()

    logger.info(f"Loaded {len(experts)} experts with weights: {w}")
    return experts, w


# -----------------------------
# Objective for Optuna
# -----------------------------
def build_objective(data: pd.DataFrame, args, logger, dump_root):
    """
    Returns a closure that:
      1) Samples (short_window, long_window),
      2) Builds a HedgeAlgorithm with those windows,
      3) Runs the online loop,
      4) Returns final Sharpe ratio (maximize).
    """
    # Pre-load experts and weights once for all trials.
    experts, weights = load_experts_and_weights(args, logger)

    # Precompute anomaly features DataFrame once (before any trial).
    anomaly_df = data.copy()
    anomaly_df.drop(columns=['adj_close'], inplace=True, errors='ignore')

    # FIXED: Create a base hedge just for computing anomaly scores once
    logger.info("Pre-computing anomaly scores once for all trials...")
    base_hedge = HedgeAlgorithm(
        experts=experts,
        short_window=5,  # Dummy values, not used for anomaly computation
        long_window=20,
        learning_rate=args.lr,
        max_loss=args.max_loss,
        start=args.start,
        end=args.end,
        logger=logger,
        max_iters=args.max_iterations,
        weights=weights,
        initial_capital=args.initial_capital
    )

    # Compute anomaly scores once
    base_hedge._compute_anomaly_scores_full(anomaly_df)
    cached_anomaly_scores = [s.copy() for s in base_hedge.cache]
    logger.info("Anomaly scores pre-computed and cached for all trials.")

    def objective(trial: optuna.Trial) -> float:
        """Objective function for a single trial."""
        try:
            # FIXED: Use fresh copy of data for each trial
            curr_data = data.copy()

            # Sample hyperparameters with validity constraints
            short_min, short_max = 2, 60
            short_window = trial.suggest_int("short_window", short_min, short_max, step=1)
            long_window = trial.suggest_int("long_window", short_window + 1, 300, step=1)

            # Prune if windows are too close (can cause instability)
            if long_window - short_window < 2:
                logger.info(f"Trial {trial.number}: Pruned (long - short < 2)")
                raise optuna.exceptions.TrialPruned()

            logger.info(f"\n{'='*60}")
            logger.info(f"Trial {trial.number}: short_window={short_window}, long_window={long_window}")
            logger.info(f"{'='*60}")

            # FIXED: Create fresh HedgeAlgorithm instance for this trial
            hedge = HedgeAlgorithm(
                experts=experts,
                short_window=short_window,
                long_window=long_window,
                learning_rate=args.lr,
                max_loss=args.max_loss,
                start=args.start,
                end=args.end,
                logger=logger,
                max_iters=args.max_iterations,
                weights=weights.copy(),  # Fresh copy of weights
                initial_capital=args.initial_capital
            )

            # FIXED: Inject pre-computed anomaly scores (avoid recomputation)
            hedge.cache = [s.copy() for s in cached_anomaly_scores]

            # Split periods
            periods = hedge._split_time_periods(curr_data)
            if len(periods) == 0:
                logger.warning(f"Trial {trial.number}: No periods, pruning")
                raise optuna.exceptions.TrialPruned()

            # Run online MWU updates
            iteration_results = []
            prev_period_end = None
            
            for i, period_end in enumerate(periods):
                if prev_period_end is None:
                    prev_period_end = curr_data.index[0] if args.start is None else pd.Timestamp(args.start).normalize()

                # Backtest this period
                res = hedge.backtest_period(
                    curr_data,
                    period_end,
                    trading_cost=args.trading_cost,
                    initial_capital=args.initial_capital
                )

                # Update weights
                upd = hedge.update_weights(
                    curr_data,
                    period_end,
                    prev_period_end,
                    trading_cost=args.trading_cost
                )

                iteration_results.append({
                    'iteration': i + 1,
                    'period_start': res['period_start'],
                    'period_end': period_end,
                    'results': res,
                    'weight_update': upd
                })

                prev_period_end = period_end

                # OPTIONAL: Early stopping if regret bound reached
                if hedge.has_reached_regret_bound(threshold=0.99):
                    logger.info(f"Trial {trial.number}: Regret bound reached at iteration {i+1}")
                    break

            # Extract final metrics
            final_results = iteration_results[-1]['results']
            final_weights = iteration_results[-1]['weight_update']

            # Get Sharpe ratio as objective value
            sharpe_ratio = final_results.get('sharpe_ratio', np.nan)
            
            # FIXED: Handle NaN/Inf Sharpe ratios
            if not np.isfinite(sharpe_ratio):
                logger.warning(f"Trial {trial.number}: Invalid Sharpe ratio ({sharpe_ratio}), returning -1e9")
                sharpe_ratio = -1e9
            
            sharpe_val = float(sharpe_ratio)

            # FIXED: Create trial directory and save results
            trial_dir = os.path.join(dump_root, f"trial_{trial.number:03d}")
            os.makedirs(trial_dir, exist_ok=True)

            # Save trial parameters
            with open(os.path.join(trial_dir, "params.json"), "w") as f:
                json.dump({
                    "trial_number": trial.number,
                    "short_window": int(short_window),
                    "long_window": int(long_window),
                    "sharpe_ratio": sharpe_val
                }, f, indent=2)

            # Save final weights
            with open(os.path.join(trial_dir, 'final_weights.json'), 'w') as f:
                json.dump({
                    "Final Expert Weights": [
                        {
                            "Expert": i + 1, 
                            "Weight": float(weight), 
                            "Cumulative Loss": float(cum_loss)
                        }
                        for i, (weight, cum_loss) in enumerate(zip(hedge.weights, hedge.expert_cumulative_losses))
                    ]
                }, f, indent=4)

            # Save overall results
            with open(os.path.join(trial_dir, 'overall_results.json'), 'w') as f:
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
                        {
                            "Expert": i + 1, 
                            "Weight": float(weight), 
                            "Cumulative Loss": float(cum_loss)
                        }
                        for i, (weight, cum_loss) in enumerate(zip(hedge.weights, hedge.expert_cumulative_losses))
                    ],
                    "Regret Analysis": {
                        "Final Regret": float(final_weights['regret']),
                        "Theoretical Bound": float(final_weights['regret_bound']),
                        "Regret / Bound Ratio": float(final_weights['regret_ratio'])
                    }
                }, f, indent=4)

            # Save iteration details
            with open(os.path.join(trial_dir, 'iteration_details.json'), 'w') as f:
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

            # FIXED: Save metrics with proper alignment
            final_metrics = final_results['metrics']
            results_file = os.path.join(trial_dir, 'backtest_results_Portfolio.json')
            json_results = _prepare_for_json(final_metrics)
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)

            # Save trades if available
            if final_metrics.get('trades'):
                try:
                    trades_df = pd.DataFrame(final_metrics['trades'])
                    trades_file = os.path.join(trial_dir, 'trades_Portfolio.csv')
                    trades_df.to_csv(trades_file, index=False)
                except Exception as e:
                    logger.warning(f"Trial {trial.number}: Could not save trades: {e}")

            # Save daily data
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
                daily_file = os.path.join(trial_dir, 'daily_data_Portfolio.csv')
                daily_df.to_csv(daily_file, index=False)
            except Exception as e:
                logger.warning(f"Trial {trial.number}: Could not save daily data: {e}")

            # Generate plots
            try:
                start_bound = curr_data.index[0] if args.start is None else pd.Timestamp(args.start).normalize()
                end_bound = curr_data.index[-1] if args.end is None else pd.Timestamp(args.end).normalize()
                plot_data = curr_data.loc[start_bound:end_bound]
                
                hedge.plot_results(plot_data['adj_close'], final_results)
                plot_file = os.path.join(trial_dir, 'online_mwu_results.png')
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()

                # Weight evolution plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                weight_history = np.array(hedge.weight_history)
                iterations = list(range(len(weight_history)))
                
                for i in range(hedge.n_experts):
                    ax1.plot(iterations, weight_history[:, i], label=f'Expert {i+1}', 
                            marker='o', markersize=3, linewidth=2)
                
                ax1.set_title('Expert Weight Evolution Over Time Periods (Online MWU)', 
                             fontsize=14, fontweight='bold')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Weight')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Learning rate evolution
                if len(hedge.regret_history) > 0:
                    iterations = [h['iteration'] for h in hedge.regret_history]
                    learning_rates = [h['learning_rate'] for h in hedge.regret_history]
                    
                    ax2.plot(iterations, learning_rates, color='purple', 
                            linewidth=2, marker='o', markersize=4)
                    ax2.set_title('Adaptive Learning Rate Over Iterations', 
                                 fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Iteration')
                    ax2.set_ylabel('Learning Rate (eta)')
                    ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_file = os.path.join(trial_dir, 'online_mwu_weight_evolution.png')
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"Trial {trial.number}: Could not generate plots: {e}")

            logger.info(f"Trial {trial.number} complete: Sharpe={sharpe_val:.4f}, "
                       f"Return={final_results['total_return']:.2f}%, "
                       f"Drawdown={final_results['max_drawdown']:.2f}%")

            return sharpe_val

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {str(e)}", exc_info=True)
            # Return very poor value instead of crashing
            return -1e9

    return objective


# -----------------------------
# Main: run the study
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for MWU MA windows (Sharpe objective)")
    parser.add_argument('--input', type=str, required=True, 
                       help='Input CSV with columns including date, log_adj_close, adj_close')
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')

    # Expert model paths
    parser.add_argument('--paper-gan-model', type=str, default=None)
    parser.add_argument('--lstm-cnn-gan-model', type=str, default=None)
    parser.add_argument('--lstm-cnn-gan-seq-model', type=str, default=None)
    parser.add_argument('--cnn-lstm-gan-seq-model', type=str, default=None)

    # Initial weights for experts
    parser.add_argument('--paper-gan-weight', type=float, default=0.25)
    parser.add_argument('--lstm-cnn-gan-weight', type=float, default=0.25)
    parser.add_argument('--lstm-cnn-gan-seq-weight', type=float, default=0.25)
    parser.add_argument('--cnn-lstm-gan-seq-weight', type=float, default=0.25)

    parser.add_argument('--trading-cost', type=float, default=0.005, 
                       help='Trading fee fraction (0.005 = 0.5%)')
    parser.add_argument('--initial-capital', type=float, default=100000.0)
    parser.add_argument('--max-iterations', type=int, default=50, 
                       help='Online MWU iterations; lower => faster tuning')
    parser.add_argument('--max-loss', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=None, 
                       help='Fixed learning rate eta; None => adaptive')

    # Optuna settings
    parser.add_argument('--study-name', type=str, default='mwu_ma_tuning')
    parser.add_argument('--storage', type=str, default=None, 
                       help='e.g., sqlite:///mwu_ma_tuning.db (optional)')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--direction', type=str, default='maximize', 
                       choices=['maximize', 'minimize'])
    parser.add_argument('--sampler', type=str, default='tpe', 
                       choices=['tpe', 'random', 'cmaes'])
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs (1 = sequential)')

    # Logging / output
    parser.add_argument('--output', type=str, default='logs_optuna')
    parser.add_argument('--log-level', type=str, default='INFO')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output, f'optuna_tuning_{args.study_name}_{timestamp}.log')
    logger = Logger.setup_logger('tuner', log_file, args.log_level)

    logger.info("=" * 60)
    logger.info("OPTUNA HYPERPARAMETER TUNING")
    logger.info("=" * 60)
    logger.info("")

    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(args.input)
    df = _coerce_date_column(df, logger)
    df.sort_values('date', inplace=True)
    df.set_index("date", inplace=True)

    logger.info(f"Loaded {len(df)} rows from {args.input}")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    logger.info("")

    # Build objective
    logger.info("Building objective function...")
    objective = build_objective(df, args, logger, args.output)
    logger.info("")

    # Sampler selection
    if args.sampler == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=args.seed)
    elif args.sampler == 'random':
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        sampler = optuna.samplers.CmaEsSampler(seed=args.seed)

    logger.info(f"Using {args.sampler.upper()} sampler with seed {args.seed}")

    # Create / load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        sampler=sampler,
        load_if_exists=True
    )

    logger.info(f"Starting Optuna optimization with {args.n_trials} trial(s)")
    logger.info(f"Objective: {args.direction} Sharpe ratio")
    logger.info("")

    # Run optimization
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)

    logger.info("")
    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best value (Sharpe): {study.best_value:.6f}")
    logger.info(f"Best params: {study.best_params}")
    logger.info("")

    # Persist best params to JSON
    best_path = os.path.join(args.output, f'best_{args.study_name}.json')
    with open(best_path, 'w') as f:
        json.dump({
            "best_value": float(study.best_value),
            "best_params": study.best_params,
            "best_trial": study.best_trial.number,
            "direction": args.direction,
            "n_trials": len(study.trials),
            "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "timestamp": timestamp
        }, f, indent=2)
    logger.info(f"Saved best params to {best_path}")

    # Generate visualization plots
    try:
        import plotly
        
        logger.info("Generating optimization visualizations...")
        
        fig1 = vis.plot_optimization_history(study)
        fig1.write_html(os.path.join(args.output, 'optimization_history.html'))
        
        fig2 = vis.plot_param_importances(study)
        fig2.write_html(os.path.join(args.output, 'param_importances.html'))
        
        fig3 = vis.plot_parallel_coordinate(study)
        fig3.write_html(os.path.join(args.output, 'parallel_coordinate.html'))
        
        fig4 = vis.plot_slice(study)
        fig4.write_html(os.path.join(args.output, 'slice_plot.html'))
        
        logger.info(f"Optimization plots saved to {args.output}")
        
    except ImportError:
        logger.warning("Plotly not available for optimization visualization")
    except Exception as e:
        logger.error(f"Error creating optimization plots: {str(e)}")

    # Print summary statistics
    logger.info("")
    logger.info("=" * 60)
    logger.info("STUDY STATISTICS")
    logger.info("=" * 60)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
        values = [t.value for t in completed_trials]
        logger.info(f"Completed trials: {len(completed_trials)}")
        logger.info(f"Best value: {study.best_value:.6f}")
        logger.info(f"Mean value: {np.mean(values):.6f}")
        logger.info(f"Std value: {np.std(values):.6f}")
        logger.info(f"Min value: {np.min(values):.6f}")
        logger.info(f"Max value: {np.max(values):.6f}")
    
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    logger.info(f"Pruned trials: {len(pruned_trials)}")
    
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    logger.info(f"Failed trials: {len(failed_trials)}")
    
    logger.info("=" * 60)

    # Print to stdout
    print("\n" + "=" * 60)
    print("OPTUNA RESULTS")
    print("=" * 60)
    print(f"Best Sharpe: {study.best_value:.6f}")
    print(f"Best Params: {study.best_params}")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Results saved to: {args.output}")
    print("=" * 60)

    try:
        rerun_best_trial(study, df, args, logger)
    except Exception as e:
        logger.error(f"Error rerunning best trial: {e}", exc_info=True)

    try:
        analyze_study_results(study, args.output, logger)
    except Exception as e:
        logger.error(f"Error analyzing study results: {e}", exc_info=True)

    return study


def rerun_best_trial(study: optuna.Study, data: pd.DataFrame, args, logger):
    """
    Rerun the best trial with full detail and save to best_result folder.
    
    Args:
        study: Completed Optuna study
        data: Original data DataFrame
        args: Command line arguments
        logger: Logger instance
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("RERUNNING BEST TRIAL WITH FULL DETAIL")
    logger.info("=" * 60)
    logger.info("")
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    best_trial_num = study.best_trial.number
    
    logger.info(f"Best Trial Number: {best_trial_num}")
    logger.info(f"Best Sharpe Ratio: {best_value:.6f}")
    logger.info(f"Best Parameters: {best_params}")
    logger.info("")
    
    # Create best_result directory
    best_result_dir = os.path.join(args.output, 'best_result')
    os.makedirs(best_result_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {best_result_dir}")
    logger.info("")
    
    # Load experts
    experts, weights = load_experts_and_weights(args, logger)
    
    # Prepare data
    curr_data = data.copy()
    anomaly_df = curr_data.copy()
    anomaly_df.drop(columns=['adj_close'], inplace=True, errors='ignore')
    
    # Create HedgeAlgorithm with best parameters
    logger.info("Initializing HedgeAlgorithm with best parameters...")
    hedge = HedgeAlgorithm(
        experts=experts,
        short_window=best_params['short_window'],
        long_window=best_params['long_window'],
        learning_rate=args.lr,
        max_loss=args.max_loss,
        start=args.start,
        end=args.end,
        logger=logger,
        max_iters=args.max_iterations,
        weights=weights.copy(),
        initial_capital=args.initial_capital
    )
    
    # Split periods and compute anomaly scores
    logger.info("Computing anomaly scores...")
    periods = hedge._split_time_periods(curr_data)
    hedge._compute_anomaly_scores_full(anomaly_df)
    logger.info(f"Split into {len(periods)} periods")
    logger.info("")
    
    # Run online MWU updates
    logger.info("Running online MWU updates...")
    logger.info("-" * 60)
    
    iteration_results = []
    prev_period_end = None
    
    for i, period_end in enumerate(periods):
        if prev_period_end is None:
            prev_period_end = curr_data.index[0] if args.start is None else pd.Timestamp(args.start).normalize()
        
        logger.info(f"Iteration {i + 1}/{len(periods)}: {prev_period_end.date()} to {period_end.date()}")
        
        # Backtest this period
        res = hedge.backtest_period(
            curr_data,
            period_end,
            trading_cost=args.trading_cost,
            initial_capital=args.initial_capital
        )
        
        # Update weights
        upd = hedge.update_weights(
            curr_data,
            period_end,
            prev_period_end,
            trading_cost=args.trading_cost
        )
        
        iteration_results.append({
            'iteration': i + 1,
            'period_start': res['period_start'],
            'period_end': period_end,
            'results': res,
            'weight_update': upd
        })
        
        # Log iteration summary
        logger.info(f"  Return: {res['total_return']:.2f}%, Sharpe: {res['sharpe_ratio']:.4f}, "
                   f"Trades: {res['total_trades']}, Position: {res['current_position']}")
        
        prev_period_end = period_end
        
        # Check regret bound
        if hedge.has_reached_regret_bound(threshold=0.99):
            logger.info(f"Regret bound reached at iteration {i+1}")
            break
    
    logger.info("-" * 60)
    logger.info("")
    
    # Extract final results
    final_results = iteration_results[-1]['results']
    final_weights = iteration_results[-1]['weight_update']
    
    # Save best parameters
    logger.info("Saving results...")
    with open(os.path.join(best_result_dir, 'best_params.json'), 'w') as f:
        json.dump({
            "trial_number": int(best_trial_num),
            "short_window": int(best_params['short_window']),
            "long_window": int(best_params['long_window']),
            "best_sharpe_ratio": float(best_value),
            "rerun_sharpe_ratio": float(final_results['sharpe_ratio']),
            "total_return": float(final_results['total_return']),
            "max_drawdown": float(final_results['max_drawdown']),
            "trading_cost": float(args.trading_cost),
            "initial_capital": float(args.initial_capital),
            "date_range": {
                "start": str(args.start) if args.start else str(curr_data.index[0].date()),
                "end": str(args.end) if args.end else str(curr_data.index[-1].date())
            }
        }, f, indent=2)
    
    # Save final weights
    with open(os.path.join(best_result_dir, 'final_weights.json'), 'w') as f:
        json.dump({
            "Final Expert Weights": [
                {
                    "Expert": i + 1,
                    "Weight": float(weight),
                    "Cumulative Loss": float(cum_loss)
                }
                for i, (weight, cum_loss) in enumerate(zip(hedge.weights, hedge.expert_cumulative_losses))
            ]
        }, f, indent=4)
    
    # Save overall results
    with open(os.path.join(best_result_dir, 'overall_results.json'), 'w') as f:
        json.dump({
            "Overall Backtest Results": {
                "Total Iterations": int(hedge.iteration),
                "Total Return": float(final_results['total_return']),
                "Buy & Hold Return": float(final_results['buy_hold_return']),
                "Excess Return": float(final_results['total_return'] - final_results['buy_hold_return']),
                "Sharpe Ratio": float(final_results['sharpe_ratio']),
                "Max Drawdown": float(final_results['max_drawdown']),
                "Total Trades": int(final_results['total_trades']),
                "Win Rate": float(final_results['win_rate']),
                "Final Portfolio Value": float(final_results['final_value']),
                "Initial Capital": float(args.initial_capital)
            },
            "Final Expert Weights": [
                {
                    "Expert": i + 1,
                    "Weight": float(weight),
                    "Cumulative Loss": float(cum_loss)
                }
                for i, (weight, cum_loss) in enumerate(zip(hedge.weights, hedge.expert_cumulative_losses))
            ],
            "Regret Analysis": {
                "Final Regret": float(final_weights['regret']),
                "Theoretical Bound": float(final_weights['regret_bound']),
                "Regret / Bound Ratio": float(final_weights['regret_ratio'])
            }
        }, f, indent=4)
    
    # Save iteration details
    with open(os.path.join(best_result_dir, 'iteration_details.json'), 'w') as f:
        json.dump({
            "Iteration Details": [
                {
                    "Iteration": int(res['iteration']),
                    "Period Start": str(res['period_start'].date()),
                    "Period End": str(res['period_end'].date()),
                    "Strategy Return": float(res['results']['total_return']),
                    "Sharpe Ratio": float(res['results']['sharpe_ratio']),
                    "Max Drawdown": float(res['results']['max_drawdown']),
                    "Total Trades": int(res['results']['total_trades']),
                    "Learning Rate (eta)": float(res['weight_update']['learning_rate']),
                    "Current Regret": float(res['weight_update']['regret']),
                    "Regret Bound": float(res['weight_update']['regret_bound']),
                    "Regret Ratio": float(res['weight_update']['regret_ratio']),
                    "Best Expert Loss": float(np.min(res['weight_update']['expert_cumulative_losses'])),
                    "Weights": [float(w) for w in res['weight_update']['new_weights']]
                }
                for res in iteration_results
            ]
        }, f, indent=4)
    
    # Save metrics with proper alignment
    final_metrics = final_results['metrics']
    results_file = os.path.join(best_result_dir, 'backtest_results_Portfolio.json')
    json_results = _prepare_for_json(final_metrics)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    # Save trades if available
    if final_metrics.get('trades'):
        try:
            trades_df = pd.DataFrame(final_metrics['trades'])
            trades_file = os.path.join(best_result_dir, 'trades_Portfolio.csv')
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Saved {len(trades_df)} trades to trades_Portfolio.csv")
        except Exception as e:
            logger.warning(f"Could not save trades: {e}")
    
    # Save daily data
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
        daily_file = os.path.join(best_result_dir, 'daily_data_Portfolio.csv')
        daily_df.to_csv(daily_file, index=False)
        logger.info(f"Saved daily data ({len(daily_df)} rows) to daily_data_Portfolio.csv")
    except Exception as e:
        logger.warning(f"Could not save daily data: {e}")
    
    # Generate plots
    logger.info("Generating visualizations...")
    try:
        start_bound = curr_data.index[0] if args.start is None else pd.Timestamp(args.start).normalize()
        end_bound = curr_data.index[-1] if args.end is None else pd.Timestamp(args.end).normalize()
        plot_data = curr_data.loc[start_bound:end_bound]
        
        # Main results plot
        hedge.plot_results(plot_data['adj_close'], final_results)
        plot_file = os.path.join(best_result_dir, 'online_mwu_results.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved main results plot to online_mwu_results.png")
        
        # Weight evolution plot
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Subplot 1: Weight evolution
        weight_history = np.array(hedge.weight_history)
        iterations = list(range(len(weight_history)))
        
        for i in range(hedge.n_experts):
            axes[0].plot(iterations, weight_history[:, i], label=f'Expert {i+1}', 
                        marker='o', markersize=3, linewidth=2)
        
        axes[0].set_title(f'Expert Weight Evolution (Best: short={best_params["short_window"]}, long={best_params["long_window"]})', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Weight')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Subplot 2: Learning rate evolution
        if len(hedge.regret_history) > 0:
            iterations = [h['iteration'] for h in hedge.regret_history]
            learning_rates = [h['learning_rate'] for h in hedge.regret_history]
            
            axes[1].plot(iterations, learning_rates, color='purple', 
                        linewidth=2, marker='o', markersize=4)
            axes[1].set_title('Adaptive Learning Rate Over Iterations', 
                             fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Learning Rate (eta)')
            axes[1].grid(True, alpha=0.3)
        
        # Subplot 3: Cumulative returns by iteration
        cum_returns = [res['results']['total_return'] for res in iteration_results]
        iterations = [res['iteration'] for res in iteration_results]
        
        axes[2].plot(iterations, cum_returns, color='green', 
                    linewidth=2, marker='o', markersize=4)
        axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[2].set_title('Cumulative Return Over Iterations', 
                         fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Cumulative Return (%)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(best_result_dir, 'online_mwu_weight_evolution.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved weight evolution plot to online_mwu_weight_evolution.png")
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}", exc_info=True)
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("BEST TRIAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Parameters:")
    logger.info(f"  Short Window: {best_params['short_window']}")
    logger.info(f"  Long Window: {best_params['long_window']}")
    logger.info(f"  Window Difference: {best_params['long_window'] - best_params['short_window']}")
    logger.info("")
    logger.info(f"Performance:")
    logger.info(f"  Total Return: {final_results['total_return']:.2f}%")
    logger.info(f"  Buy & Hold Return: {final_results['buy_hold_return']:.2f}%")
    logger.info(f"  Excess Return: {final_results['total_return'] - final_results['buy_hold_return']:.2f}%")
    logger.info(f"  Sharpe Ratio: {final_results['sharpe_ratio']:.4f}")
    logger.info(f"  Max Drawdown: {final_results['max_drawdown']:.2f}%")
    logger.info(f"  Total Trades: {final_results['total_trades']}")
    logger.info(f"  Win Rate: {final_results['win_rate']:.2f}%")
    logger.info(f"  Final Portfolio Value: ${final_results['final_value']:,.2f}")
    logger.info("")
    logger.info(f"Expert Weights:")
    for i, (weight, cum_loss) in enumerate(zip(hedge.weights, hedge.expert_cumulative_losses)):
        logger.info(f"  Expert {i+1}: {weight:.2%} (Cumulative Loss: {cum_loss:.4f})")
    logger.info("")
    logger.info(f"Regret Analysis:")
    logger.info(f"  Final Regret: {final_weights['regret']:.4f}")
    logger.info(f"  Theoretical Bound: {final_weights['regret_bound']:.4f}")
    logger.info(f"  Regret / Bound Ratio: {final_weights['regret_ratio']:.2%}")
    logger.info("=" * 60)
    logger.info(f"All results saved to: {best_result_dir}")
    logger.info("=" * 60)
    logger.info("")

    return hedge, iteration_results, best_result_dir

if __name__ == "__main__":
    study = main()


# -----------------------------
# Additional Analysis Functions
# -----------------------------
def analyze_study_results(study: optuna.Study, output_dir: str, logger):
    """Analyze and visualize study results in detail."""
    
    logger.info("Performing detailed study analysis...")
    
    # Get completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) == 0:
        logger.warning("No completed trials to analyze")
        return
    
    # Create DataFrame of trial results
    trial_data = []
    for trial in completed_trials:
        trial_data.append({
            'trial': trial.number,
            'short_window': trial.params['short_window'],
            'long_window': trial.params['long_window'],
            'sharpe_ratio': trial.value,
            'window_diff': trial.params['long_window'] - trial.params['short_window']
        })
    
    df_trials = pd.DataFrame(trial_data)
    
    # Save trial summary
    summary_file = os.path.join(output_dir, 'trial_summary.csv')
    df_trials.to_csv(summary_file, index=False)
    logger.info(f"Saved trial summary to {summary_file}")
    
    # Create additional matplotlib plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Sharpe vs Short Window
    axes[0, 0].scatter(df_trials['short_window'], df_trials['sharpe_ratio'], alpha=0.6)
    axes[0, 0].set_xlabel('Short Window')
    axes[0, 0].set_ylabel('Sharpe Ratio')
    axes[0, 0].set_title('Sharpe Ratio vs Short Window')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Sharpe vs Long Window
    axes[0, 1].scatter(df_trials['long_window'], df_trials['sharpe_ratio'], alpha=0.6, color='orange')
    axes[0, 1].set_xlabel('Long Window')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].set_title('Sharpe Ratio vs Long Window')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Sharpe vs Window Difference
    axes[1, 0].scatter(df_trials['window_diff'], df_trials['sharpe_ratio'], alpha=0.6, color='green')
    axes[1, 0].set_xlabel('Window Difference (Long - Short)')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].set_title('Sharpe Ratio vs Window Difference')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: 2D Heatmap of parameter space
    pivot_table = df_trials.pivot_table(
        values='sharpe_ratio',
        index='short_window',
        columns='long_window',
        aggfunc='mean'
    )
    im = axes[1, 1].imshow(pivot_table, aspect='auto', cmap='viridis', origin='lower')
    axes[1, 1].set_xlabel('Long Window')
    axes[1, 1].set_ylabel('Short Window')
    axes[1, 1].set_title('Sharpe Ratio Heatmap')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'parameter_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved parameter analysis plot to {plot_file}")
    
    # Statistical summary
    stats_summary = {
        'n_trials': len(completed_trials),
        'best_sharpe': float(df_trials['sharpe_ratio'].max()),
        'mean_sharpe': float(df_trials['sharpe_ratio'].mean()),
        'std_sharpe': float(df_trials['sharpe_ratio'].std()),
        'median_sharpe': float(df_trials['sharpe_ratio'].median()),
        'short_window_range': [int(df_trials['short_window'].min()), int(df_trials['short_window'].max())],
        'long_window_range': [int(df_trials['long_window'].min()), int(df_trials['long_window'].max())],
        'best_short_window': int(df_trials.loc[df_trials['sharpe_ratio'].idxmax(), 'short_window']),
        'best_long_window': int(df_trials.loc[df_trials['sharpe_ratio'].idxmax(), 'long_window']),
        'correlation_short_sharpe': float(df_trials['short_window'].corr(df_trials['sharpe_ratio'])),
        'correlation_long_sharpe': float(df_trials['long_window'].corr(df_trials['sharpe_ratio'])),
        'correlation_diff_sharpe': float(df_trials['window_diff'].corr(df_trials['sharpe_ratio']))
    }
    
    stats_file = os.path.join(output_dir, 'statistical_summary.json')
    with open(stats_file, 'w') as f:
        json.dump(stats_summary, f, indent=2)
    logger.info(f"Saved statistical summary to {stats_file}")
    
    # Top 5 trials
    top5 = df_trials.nlargest(5, 'sharpe_ratio')
    logger.info("\nTop 5 Trials:")
    logger.info("-" * 60)
    for idx, row in top5.iterrows():
        logger.info(f"Trial {int(row['trial'])}: short={int(row['short_window'])}, "
                   f"long={int(row['long_window'])}, Sharpe={row['sharpe_ratio']:.4f}")
    
    return df_trials, stats_summary