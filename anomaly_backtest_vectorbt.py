#!/usr/bin/env python3
"""
Financial Anomaly Detection Backtesting System with Optuna Optimization and VectorBT
Designed for SLURM cluster execution with comprehensive logging
"""

import pandas as pd
import numpy as np
import argparse
import json
import os
import sys
from datetime import datetime

try:
    from mpi4py import MPI
    mpi_available = True
except ImportError:
    mpi_available = False

# Import common modules
from common.anomaly_detector import create_anomaly_detector
from common.backtest_common import BacktestConfig, VectorBTBacktester
from common.optuna_backtest import OptunaBayesianOptimizer
from common.data_utils import prepare_data
from common.io_utils import load_config, save_config
from common.logger_utils import Logger

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='Financial Anomaly Detection Backtesting with Optuna and VectorBT'
    )
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--input', type=str, required=True, help='Input data file (CSV)')
    parser.add_argument('--output', type=str, default='./results', help='Output directory')
    parser.add_argument('--model-path', type=str, default='./models', help='GAN model path')
    parser.add_argument(
        '--model-type', 
        type=str, 
        default='paper_gan_price',
        choices=[
            'cnn_lstm_seq_ohlcav', 'cnn_lstm_seq_price',
            'lstm_cnn_parallel_ohlcav', 'lstm_cnn_parallel_price',
            'lstm_cnn_seq_ohlcav', 'lstm_cnn_seq_price',
            'paper_gan_ohlcav', 'paper_gan_price'
        ],
        help='GAN model type'
    )
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR']
    )
    
    # Time period parameters
    parser.add_argument('--start-date', type=str, help='Test start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Test end date (YYYY-MM-DD)')
    parser.add_argument('--train-start-date', type=str, help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--train-end-date', type=str, help='Training end date (YYYY-MM-DD)')
    
    # Optimization parameters
    parser.add_argument(
        '--optimize', 
        action='store_true', 
        help='Run Bayesian parameter optimization'
    )
    parser.add_argument('--n-trials', type=int, default=100, help='Number of optimization trials')
    
    # Parameter ranges for optimization
    parser.add_argument('--short-ma-min', type=int, default=2, help='Minimum short MA period')
    parser.add_argument('--short-ma-max', type=int, default=60, help='Maximum short MA period')
    parser.add_argument('--long-ma-min', type=int, default=20, help='Minimum long MA period')
    parser.add_argument('--long-ma-max', type=int, default=300, help='Maximum long MA period')
    
    # Manual parameter override (if not optimizing)
    parser.add_argument('--short-ma', type=int, default=10, help='Short moving average period')
    parser.add_argument('--long-ma', type=int, default=50, help='Long moving average period')
    parser.add_argument(
        '--lookback', 
        type=int, 
        default=20, 
        help='Lookback window for anomaly detection'
    )
    
    # Trading parameters
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital')
    parser.add_argument(
        '--transaction-cost', 
        type=float, 
        default=0.005, 
        help='Transaction cost (0.005 = 0.5%)'
    )
    
    # Anomaly detection parameters
    parser.add_argument(
        '--use-full-history', 
        action='store_true',
        help='Use full history for anomaly scoring (vs limited lookback)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(
        args.output, 
        f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
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
                model_type=args.model_type,
                log_level=args.log_level,
                optimize_params=args.optimize,
                n_trials=args.n_trials,
                short_ma_min=args.short_ma_min,
                short_ma_max=args.short_ma_max,
                long_ma_min=args.long_ma_min,
                long_ma_max=args.long_ma_max,
                use_full_history=args.use_full_history,
            )
            logger.info("Using default configuration")
        
        # Override config with command line arguments
        config.output_dir = args.output
        config.log_level = args.log_level
        config.model_path = args.model_path
        config.model_type = args.model_type
        config.optimize_params = args.optimize
        config.n_trials = args.n_trials
        config.initial_capital = args.initial_capital
        config.transaction_cost = args.transaction_cost
        config.use_full_history = args.use_full_history
        
        # Override time periods if provided
        if args.start_date:
            config.start_date = args.start_date
        if args.end_date:
            config.end_date = args.end_date
        if args.train_start_date:
            config.train_start_date = args.train_start_date
        if args.train_end_date:
            config.train_end_date = args.train_end_date
        
        config.input_file = args.input
        
        # Log configuration
        logger.info(
            f"Configuration: Short MA={config.short_ma_period}, "
            f"Long MA={config.long_ma_period}, "
            f"Lookback={config.lookback_window}, TX Cost={config.transaction_cost*100:.1f}%"
        )
        
        if config.start_date or config.end_date:
            logger.info(
                f"Test period: {config.start_date or 'beginning'} to "
                f"{config.end_date or 'end'}"
            )
        if config.train_start_date or config.train_end_date:
            logger.info(
                f"Training period: {config.train_start_date or 'beginning'} to "
                f"{config.train_end_date or 'auto'}"
            )
        
        # Save current configuration
        config_file = os.path.join(args.output, 'config.json')
        save_config(config, config_file)
        logger.info(f"Configuration saved to {config_file}")

        # Load lookback window from model params
        model_params_path = os.path.join(args.model_path, 'best_params.json')
        if os.path.exists(model_params_path):
            with open(model_params_path, "r", encoding="utf-8") as f:
                model_params = json.load(f)
            config.lookback_window = model_params["sequence_length"]
            logger.info(f"Loaded lookback window from model: {config.lookback_window}")
        
        # Load price data
        if not os.path.exists(config.input_file):
            logger.error(f"Input file not found: {config.input_file}")
            sys.exit(1)
        
        logger.info(f"Loading price data from {config.input_file}")
        price_data = prepare_data(config.input_file, config.model_type)
        
        logger.info(
            f"Loaded {len(price_data)} data points from "
            f"{price_data['date'].min()} to {price_data['date'].max()}"
        )
        
        # Validate minimum data requirements
        min_required_points = config.lookback_window + config.long_ma_max + 50
        if len(price_data) < min_required_points:
            logger.warning(
                f"Data has only {len(price_data)} points, but {min_required_points} "
                "recommended for robust optimization"
            )
        
        # Create anomaly detector
        anomaly_detector = create_anomaly_detector(
            model_type=config.model_type,
            model_path=config.model_path,
            lookback_window=config.lookback_window,
            start_date=config.start_date,
            end_date=config.end_date,
            logger=logger
        )

        if config.optimize_params:
            # Run Bayesian optimization with Optuna
            logger.info("Starting Bayesian optimization with Optuna...")
            optimizer = OptunaBayesianOptimizer(
                config, 
                anomaly_detector, 
                logger,
                VectorBTBacktester
            )
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
        
        # Run main backtest
        if mpi_available:
            should_run = not config.optimize_params or MPI.COMM_WORLD.Get_rank() == 0
        else:
            should_run = True
        
        if should_run:
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
