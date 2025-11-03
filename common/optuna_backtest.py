#!/usr/bin/env python3
"""
Common Optuna Optimization Module for Backtesting
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, Optional
from datetime import datetime
from copy import deepcopy

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import optuna.visualization as vis

# Try to import MPI for parallel processing
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


class OptunaBayesianOptimizer:
    """Bayesian optimization using Optuna for hyperparameter tuning"""
    
    def __init__(
        self, 
        config, 
        anomaly_detector, 
        logger: logging.Logger,
        backtester_class
    ):
        """
        Initialize optimizer
        
        Args:
            config: BacktestConfig instance
            anomaly_detector: AnomalyDetector instance
            logger: Logger instance
            backtester_class: VectorBTBacktester class (not instance)
        """
        self.config = config
        self.logger = logger
        self.study = None
        self.anomaly_detector = anomaly_detector
        self.backtester_class = backtester_class
        
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
    
    def objective(self, trial: optuna.Trial, price_data: pd.DataFrame) -> float:
        """Objective function for Optuna optimization"""
        
        # Suggest hyperparameters
        short_ma = trial.suggest_int(
            'short_ma_period', 
            self.config.short_ma_min, 
            self.config.short_ma_max
        )
        long_ma = trial.suggest_int(
            'long_ma_period', 
            max(short_ma + 1, self.config.long_ma_min), 
            self.config.long_ma_max
        )
        
        # Create temporary config with trial parameters
        temp_config = deepcopy(self.config)
        temp_config.short_ma_period = short_ma
        temp_config.long_ma_period = long_ma
        
        try:
            # Run backtest
            backtester = self.backtester_class(
                temp_config, 
                self.anomaly_detector, 
                self.logger
            )
            results = backtester.backtest(price_data)
            
            # Optimization objective: maximize risk-adjusted return
            sharpe_ratio = results['sharpe_ratio']
            
            objective_score = (
                float(sharpe_ratio) 
                if not pd.isna(sharpe_ratio) 
                else -10.0
            )

            # Report for pruning
            trial.report(objective_score, step=0)
            
            self.logger.debug(
                f"Trial {trial.number}: Short={short_ma}, Long={long_ma}, "
                f"Lookback={temp_config.lookback_window}, "
                f"Objective={objective_score:.4f}"
            )
            
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
        n_trials = (
            self.config.n_trials // self.size 
            if MPI_AVAILABLE else self.config.n_trials
        )
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
            return {
                'best_params': None, 
                'best_value': None, 
                'optimization_history': []
            }
    
    def plot_optimization_history(self, output_dir: str):
        """Plot optimization history and parameter importance"""
        if self.study is None or self.rank != 0:
            return
            
        try:
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
