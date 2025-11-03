#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import argparse
import logging
from typing import List, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless plotting for SLURM
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.preprocessing import MinMaxScaler

# Import your existing GAN implementation
from gan_model import GANExpert
from gan_runner import (
    read_ohlc_csv, slice_by_date, make_sequences, 
    save_training_plot, save_json, _expand
)

LOG = logging.getLogger("optuna_gan_optimizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class GANOptunaOptimizer:
    """Optuna-based hyperparameter optimizer for GAN models."""
    
    def __init__(
        self,
        data_dir: str,
        ticker: str,
        out_dir: str,
        n_trials: int = 100,
        timeout: int = 3600,  # 1 hour timeout
        n_jobs: int = 1,
        epoch_unit: str = "auto",
        string_date_format: Optional[str] = None
    ):
        self.data_dir = _expand(data_dir)
        self.ticker = ticker
        self.out_dir = _expand(out_dir)
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.epoch_unit = epoch_unit
        self.string_date_format = string_date_format
        
        # Create output directory
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Load and prepare data once
        self._load_data()
        
        # Best trial tracking
        self.best_params = None
        self.best_value = float('inf')
        
    def _load_data(self):
        """Load and preprocess data for the specified ticker."""
        LOG.info(f"Loading data for ticker: {self.ticker}")
        
        try:
            df_all = read_ohlc_csv(
                self.data_dir,
                self.ticker,
                epoch_unit=self.epoch_unit,
                string_date_format=self.string_date_format
            )
            
            # Split data into train/validation/test
            self.df_train1 = slice_by_date(df_all, "2003-01-01", "2006-12-31")
            self.df_train2 = slice_by_date(df_all, "2010-01-01", "2017-12-31")
            self.df_validation = slice_by_date(df_all, "2007-01-01", "2009-12-31")
            self.df_test = slice_by_date(df_all, "2021-07-01", "2025-07-31")
            
            features_list = df_all.columns.tolist()
            features_list.remove("Date")
            self.features_list = features_list
            
            LOG.info(f"Data loaded successfully:")
            LOG.info(f"  Train period 1: {len(self.df_train1)} rows")
            LOG.info(f"  Train period 2: {len(self.df_train2)} rows") 
            LOG.info(f"  Validation: {len(self.df_validation)} rows")
            LOG.info(f"  Test: {len(self.df_test)} rows")
            
        except Exception as e:
            LOG.error(f"Error loading data for {self.ticker}: {e}")
            raise
    
    def _create_sequences(self, sequence_length: int):
        """Create sequences for training and validation."""
        X_train1 = make_sequences(self.df_train1, self.features_list, sequence_length)
        X_train2 = make_sequences(self.df_train2, self.features_list, sequence_length)
        X_validation = make_sequences(self.df_validation, self.features_list, sequence_length)
        
        if X_train1.shape[0] == 0 or X_train2.shape[0] == 0:
            raise ValueError(f"Insufficient training data for sequence length {sequence_length}")
        
        if X_validation.shape[0] == 0:
            raise ValueError(f"Insufficient validation data for sequence length {sequence_length}")
        
        X_train = np.concatenate([X_train1, X_train2], axis=0)
        
        return X_train, X_validation
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function to minimize."""
        
        # Sample hyperparameters
        sequence_length = trial.suggest_categorical('sequence_length', [5, 7, 10, 14, 15, 20, 21, 25, 28, 30])
        epochs = trial.suggest_int('epochs', 10, 200, step=10)  # 10, 20, ..., 200
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        latent_dim = trial.suggest_int('latent_dim', 10, 1000, step=10)  # 10, 20, ..., 1000
        
        # Learning rates - log scale sampling
        learning_rate_g = trial.suggest_float('learning_rate_g', 1e-5, 1e-3, log=True)
        learning_rate_d = trial.suggest_float('learning_rate_d', 1e-5, 1e-3, log=True)  
        learning_rate_anomaly = trial.suggest_float('learning_rate_anomaly', 1e-4, 0.01, log=True)
        
        resample_z = trial.suggest_int('resample_z', 1, 10)
        negative_slope_g = trial.suggest_float('negative_slope_g', 0.0, 0.5, step=0.01)
        negative_slope_d = trial.suggest_float('negative_slope_d', 0.0, 0.5, step=0.01)
        backprop_steps = trial.suggest_int('backprop_steps', 1, 100)  # 1, 5, 10, ..., 100

        base_channels_g = trial.suggest_categorical('base_channels_g', [16, 32, 64, 128, 256])
        base_channels_d = trial.suggest_categorical('base_channels_d', [16, 32, 64, 128, 256])

        try:
            # Create sequences with current sequence length
            X_train, X_validation = self._create_sequences(sequence_length)
            
            # Create trial-specific output directory
            trial_dir = os.path.join(self.out_dir, f"trial_{trial.number}")
            os.makedirs(trial_dir, exist_ok=True)
            
            # Initialize model with current hyperparameters
            model = GANExpert(
                seq_len=sequence_length,
                features=len(self.features_list),
                latent_dim=latent_dim,
                epochs=epochs,
                batch_sizes=batch_size,
                lr_G=learning_rate_g,
                lr_D=learning_rate_d,
                lr_anomaly=learning_rate_anomaly,
                scaler_type='robust',  # Keep this fixed for consistency
                model_save_path=trial_dir,
                resample_z=resample_z,
                negative_slope_G=negative_slope_g,
                negative_slope_D=negative_slope_d,
                backprop_steps=backprop_steps,
                base_channels_G=base_channels_g,
                base_channels_D=base_channels_d,
                lambda_anom=0.9,  # Keep this fixed
                kernel_size_G=[10, 5, 3, 2],  # Keep architecture fixed
                kernel_size_D=[3, 3, 3, 3],
            )
            
            LOG.info(f"Trial {trial.number}: Training with params:")
            LOG.info(f"  seq_len={sequence_length}, epochs={epochs}, batch_size={batch_size}")
            LOG.info(f"  latent_dim={latent_dim}, lr_G={learning_rate_g:.2e}, lr_D={learning_rate_d:.2e}")
            
            # Train the model
            model.fit(X_train=X_train, X_val=X_validation, verbose=0, log_every=100)
            
            # Calculate validation loss (TanoGAN loss)
            try:
                import torch
                X_val_tensor = torch.FloatTensor(model._preprocess_financial_data(X_validation)).to(model.device)
                val_loss = model._evaluate_tanogan_loss(X_val_tensor).mean().item()
            except Exception as e:
                LOG.warning(f"Could not compute TanoGAN loss, using generator loss: {e}")
                val_loss = model.training_history['g_loss'][-1] if model.training_history['g_loss'] else float('inf')
            
            # Save trial results
            trial_results = {
                'trial_number': trial.number,
                'params': trial.params,
                'validation_loss': val_loss,
                'training_history': model.training_history
            }
            
            save_json(trial_results, os.path.join(trial_dir, 'trial_results.json'))
            
            # Save training plot
            save_training_plot(model.training_history, os.path.join(trial_dir, 'training_history.png'))
            
            LOG.info(f"Trial {trial.number} completed. Validation loss: {val_loss:.6f}")
            
            # Intermediate reporting for pruning
            trial.report(val_loss, step=epochs)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return val_loss
            
        except optuna.TrialPruned:
            LOG.info(f"Trial {trial.number} was pruned")
            raise
        except Exception as e:
            LOG.error(f"Trial {trial.number} failed: {e}")
            # Return a large penalty value instead of failing completely
            return float('inf')
    
    def optimize(self) -> Dict[str, Any]:
        """Run Optuna optimization."""
        
        # Create study
        study_name = f"gan_optimization_{self.ticker}"
        storage_url = f"sqlite:///{os.path.join(self.out_dir, 'optuna_study.db')}"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            load_if_exists=True
        )
        
        LOG.info(f"Starting optimization for {self.ticker}")
        LOG.info(f"Target trials: {self.n_trials}")
        LOG.info(f"Timeout: {self.timeout} seconds")
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        # Get results
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        LOG.info("Optimization completed!")
        LOG.info(f"Best validation loss: {self.best_value:.6f}")
        LOG.info(f"Best parameters: {self.best_params}")
        
        # Save optimization results
        optimization_results = {
            'ticker': self.ticker,
            'n_trials': len(study.trials),
            'n_completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'n_failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'best_value': self.best_value,
            'best_params': self.best_params,
            'study_name': study_name,
        }
        
        save_json(optimization_results, os.path.join(self.out_dir, 'optimization_results.json'))
        
        # Generate optimization plots
        self._generate_optimization_plots(study)
        
        return optimization_results
    
    def _generate_optimization_plots(self, study: optuna.Study):
        """Generate optimization visualization plots."""
        try:
            # Optimization history
            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            fig.savefig(os.path.join(self.out_dir, 'optimization_history.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Parameter importances
            try:
                fig = optuna.visualization.matplotlib.plot_param_importances(study)
                fig.savefig(os.path.join(self.out_dir, 'param_importances.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                LOG.warning(f"Could not generate parameter importance plot: {e}")
            
            # Parallel coordinate plot
            try:
                fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
                fig.savefig(os.path.join(self.out_dir, 'parallel_coordinate.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                LOG.warning(f"Could not generate parallel coordinate plot: {e}")
            
            # Slice plot
            try:
                fig = optuna.visualization.matplotlib.plot_slice(study)
                fig.savefig(os.path.join(self.out_dir, 'slice_plot.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                LOG.warning(f"Could not generate slice plot: {e}")
                
        except Exception as e:
            LOG.warning(f"Error generating optimization plots: {e}")
    
    def train_best_model(self) -> GANExpert:
        """Train the final model with best hyperparameters."""
        if self.best_params is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
        
        LOG.info("Training final model with best hyperparameters...")
        
        # Create sequences with best sequence length
        X_train, X_validation = self._create_sequences(self.best_params['sequence_length'])
        X_test = make_sequences(self.df_test, self.features_list, self.best_params['sequence_length'])
        
        # Create final model directory
        final_model_dir = os.path.join(self.out_dir, 'best_model')
        os.makedirs(final_model_dir, exist_ok=True)
        
        # Initialize best model
        best_model = GANExpert(
            seq_len=self.best_params['sequence_length'],
            features=len(self.features_list),
            latent_dim=self.best_params['latent_dim'],
            epochs=self.best_params['epochs'],
            batch_sizes=self.best_params['batch_size'],
            lr_G=self.best_params['learning_rate_g'],
            lr_D=self.best_params['learning_rate_d'],
            lr_anomaly=self.best_params['learning_rate_anomaly'],
            scaler_type='robust',
            model_save_path=final_model_dir,
            resample_z=self.best_params['resample_z'],
            negative_slope_G=self.best_params['negative_slope_g'],
            negative_slope_D=self.best_params['negative_slope_d'],
            backprop_steps=self.best_params['backprop_steps'],
            base_channels_G=self.best_params['base_channels_g'],
            base_channels_D=self.best_params['base_channels_d'],
            lambda_anom=0.9,
            kernel_size_G=[10, 5, 3, 2],
            kernel_size_D=[3, 3, 3, 3],
        )
        
        # Train the best model
        best_model.fit(X_train=X_train, X_val=X_validation, verbose=1)
        
        # Save the best model
        best_model.save_models(path=final_model_dir)
        save_training_plot(best_model.training_history, os.path.join(final_model_dir, 'training_history.png'))
        save_json(best_model.training_history, os.path.join(final_model_dir, 'training_history.json'))
        save_json(self.best_params, os.path.join(final_model_dir, 'best_params.json'))
        
        # Run anomaly detection on test set if available
        if X_test.shape[0] > 0:
            LOG.info("Running anomaly detection on test set...")
            details = best_model.detect_financial_anomalies(X_test, threshold_percentile=95.0, return_details=True)
            
            end_indices = self.df_test.index[self.best_params['sequence_length'] - 1:]
            end_dates = self.df_test.loc[end_indices, "Date"].dt.strftime("%Y-%m-%d").tolist()
            scores = details["anomaly_scores"].tolist()
            labels = details["anomaly_labels"].tolist()
            end_dates_epoch = self.df_test.loc[end_indices, "Date"].tolist()

            out_scores = pd.DataFrame({
                "window_end_date": end_dates[:len(scores)],
                "score": scores,
                "label": labels,
                "window_end_date_epoch": end_dates_epoch
            })
            out_scores.to_csv(os.path.join(final_model_dir, "anomaly_scores.csv"), index=False)
            
            # Generate anomaly plot
            self._save_price_anomaly_plot(out_scores, self.df_test, 
                                        os.path.join(final_model_dir, "price_anomalies.png"))
        
        LOG.info(f"Best model trained and saved to: {final_model_dir}")
        return best_model
    
    def _save_price_anomaly_plot(self, out_scores: pd.DataFrame, df_test: pd.DataFrame, out_png: str):
        """Save price anomaly visualization plot."""
        out_scores_normalized = MinMaxScaler(feature_range=(0,1)).fit_transform(out_scores[["score"]])
        out_scores["score_normalized"] = out_scores_normalized

        df_test_normalized = MinMaxScaler(feature_range=(0,1)).fit_transform(df_test[["log_adj_close"]])
        df_test["adj_close_normalized"] = df_test_normalized

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_test["Date"], df_test["adj_close_normalized"], label="Actual Prices", alpha=0.7)
        ax.plot(out_scores["window_end_date_epoch"], out_scores["score_normalized"], 
                color="red", label="Anomaly Scores", alpha=0.8)
        ax.set_title(f"Price Anomalies - {self.ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Values")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight", dpi=150)
        plt.close(fig)


def main():
    """Main function for running Optuna optimization."""
    parser = argparse.ArgumentParser(description="Optuna-based GAN hyperparameter optimization")
    
    # Data arguments
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker to optimize on")
    parser.add_argument("--data-dir", type=str, default="~/Data/OHLC", help="Directory with {TICKER}.csv files")
    parser.add_argument("--out-dir", type=str, default="./optuna_results", help="Output directory for results")
    
    # Optimization arguments  
    parser.add_argument("--n-trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds (default: 1 hour)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs")
    
    # Data processing arguments
    parser.add_argument("--epoch-unit", type=str, default="auto", choices=["auto", "s", "ms"])
    parser.add_argument("--string-date-format", type=str, default=None, 
                       help="Date format for non-numeric timestamps")
    
    # Control arguments
    parser.add_argument("--train-best", action="store_true", 
                       help="Train final model with best hyperparameters after optimization")
    parser.add_argument("--resume", action="store_true",
                       help="Resume existing optimization study")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = GANOptunaOptimizer(
        data_dir=args.data_dir,
        ticker=args.ticker,
        out_dir=os.path.join(args.out_dir, args.ticker),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        epoch_unit=args.epoch_unit,
        string_date_format=args.string_date_format
    )
    
    try:
        # Run optimization
        results = optimizer.optimize()
        
        LOG.info("=" * 50)
        LOG.info("OPTIMIZATION SUMMARY")
        LOG.info("=" * 50)
        LOG.info(f"Ticker: {results['ticker']}")
        LOG.info(f"Total trials: {results['n_trials']}")
        LOG.info(f"Completed trials: {results['n_completed_trials']}")
        LOG.info(f"Pruned trials: {results['n_pruned_trials']}")
        LOG.info(f"Failed trials: {results['n_failed_trials']}")
        LOG.info(f"Best validation loss: {results['best_value']:.6f}")
        LOG.info(f"Best parameters:")
        for param, value in results['best_params'].items():
            LOG.info(f"  {param}: {value}")
        
        # Train best model if requested
        if args.train_best:
            LOG.info("=" * 50)
            LOG.info("TRAINING BEST MODEL")
            LOG.info("=" * 50)
            best_model = optimizer.train_best_model()
            LOG.info("Best model training completed!")
        
    except Exception as e:
        LOG.error(f"Optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()
