"""
Unified GAN training & Optuna optimization runner.

This module replaces the separate `gan_runner.py` and `gan_optuna_optimizer.py` entrypoints
with a single, refactored CLI that reuses shared utilities in `data_utils.py`, `plot_utils.py`,
`io_utils.py`, and the Optuna base class in `optuna_common.py`.

Key features:
- "train" mode: per‑ticker training pipeline (parity with former gan_runner)
- "optimize" mode: Optuna study using BaseOptunaOptimizer with a GAN subclass
- Single source of truth for data loading, sequence making, plotting, and JSON I/O
- SLURM/headless friendly (Agg backend), minimal external assumptions

Usage examples:
    python -m gan_train_optuna --mode train --tickers AAPL --data-dir ~/Data/OHLC --out-dir ./runs
    python -m gan_train_optuna --mode optimize --ticker AAPL --data-dir ~/Data/OHLC --out-dir ./optuna_out --n-trials 50 --timeout 1800 --train-best
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Shared utilities ---
from common.data_utils import (
    expand_path as _expand,
    read_ohlc_csv,
    slice_by_date,
    make_sequences,
    parse_tickers_arg,
)
from common.plot_utils import save_training_plot, save_price_anomaly_plot
from common.io_utils import save_json

# --- Model dependency ---
from paper_gan_price.gan_model import GANExpert

# --- Optuna base optimizer ---
from common.optuna_common import BaseOptunaOptimizer  # uses the "common.*" shims above

LOG = logging.getLogger("gan_train_optuna")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =====================================================================
# Optuna subclass: adapts GANExpert to the BaseOptunaOptimizer protocol
# =====================================================================

class GANOptunaStudy(BaseOptunaOptimizer):
    """Optuna optimizer for GANExpert using the project's shared helpers.

    Overrides:
      - sample_hparams
      - build_model
      - (optionally) model_eval_loss (inherited default tries TAnoGAN loss, then g_loss)
    """

    def sample_hparams(self, trial) -> Dict[str, Any]:
        # Mirror prior ranges while keeping them tidy and reproducible
        params = {
            "sequence_length": trial.suggest_categorical("sequence_length", [5, 7, 10, 14, 15, 20, 21, 25, 28, 30]),
            "epochs": trial.suggest_int("epochs", 10, 200, step=10),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "latent_dim": trial.suggest_int("latent_dim", 10, 1000, step=10),
            "learning_rate_g": trial.suggest_float("learning_rate_g", 1e-5, 1e-3, log=True),
            "learning_rate_d": trial.suggest_float("learning_rate_d", 1e-5, 1e-3, log=True),
            "learning_rate_anomaly": trial.suggest_float("learning_rate_anomaly", 1e-4, 1e-2, log=True),
            "resample_z": trial.suggest_int("resample_z", 1, 10),
            "negative_slope_g": trial.suggest_float("negative_slope_g", 0.0, 0.5, step=0.01),
            "negative_slope_d": trial.suggest_float("negative_slope_d", 0.0, 0.5, step=0.01),
            "backprop_steps": trial.suggest_int("backprop_steps", 1, 100),
            "base_channels_g": trial.suggest_categorical("base_channels_g", [16, 32, 64, 128, 256]),
            "base_channels_d": trial.suggest_categorical("base_channels_d", [16, 32, 64, 128, 256]),
        }
        return params

    def build_model(self, params: Dict[str, Any], trial_dir: str) -> GANExpert:
        # Use project defaults that were stable in prior code paths
        model = GANExpert(
            seq_len=params["sequence_length"],
            features=len(self.features_list),
            latent_dim=params["latent_dim"],
            epochs=params["epochs"],
            batch_sizes=params["batch_size"],
            lr_G=params["learning_rate_g"],
            lr_D=params["learning_rate_d"],
            lr_anomaly=params["learning_rate_anomaly"],
            scaler_type="robust",
            model_save_path=trial_dir,
            resample_z=params["resample_z"],
            negative_slope_G=params["negative_slope_g"],
            negative_slope_D=params["negative_slope_d"],
            backprop_steps=params["backprop_steps"],
            base_channels_G=params["base_channels_g"],
            base_channels_D=params["base_channels_d"],
            lambda_anom=0.9,
            kernel_size_G=[10, 5, 3, 2],
            kernel_size_D=[3, 3, 3, 3],
        )
        return model


# ======================================
# Plain training (parity with runner)
# ======================================

def train_one_ticker(
    ticker: str,
    data_dir: str,
    out_root: str,
    sequence_length: int = 64,
    epochs: int = 200,
    batch_size: int = 64,
    latent_dim: int = 100,
    learning_rate_g: float = 1.5e-4,
    learning_rate_d: float = 1e-5,
    learning_rate_anomaly: float = 5e-2,
    scaler: str = "robust",
    lambda_anomaly: float = 0.9,
    resample_z: int = 3,
    negative_slope_G: float = 0.2,
    negative_slope_D: float = 0.2,
    backprop_steps: int = 50,
    base_channels_G: int = 16,
    base_channels_D: int = 64,
    epoch_unit: str = "auto",
    string_date_format: Optional[str] = None,
) -> Optional["GANExpert"]:
    """Train a GANExpert on a single ticker using the shared utilities."""
    LOG.info("Training ticker: %s", ticker)

    df_all = read_ohlc_csv(
        data_dir,
        ticker,
        epoch_unit=epoch_unit,
        string_date_format=string_date_format,
    )
    # Project convention: use log_adj_close only if present
    if "log_adj_close" not in df_all.columns:
        raise ValueError("Column 'log_adj_close' missing in input CSV for %s" % ticker)
    df_all = df_all[["date", "log_adj_close"]]

    # date windows kept consistent with prior scripts
    df_train1 = slice_by_date(df_all, "2003-01-01", "2006-12-31")
    df_train2 = slice_by_date(df_all, "2010-01-01", "2017-12-31")
    df_test = slice_by_date(df_all, "2018-01-01", "2023-12-31")
    df_validation = slice_by_date(df_all, "2007-01-01", "2009-12-31")

    features_list = df_all.columns.tolist()
    features_list.remove("date")

    X_train1 = make_sequences(df_train1, features_list, sequence_length)
    X_train2 = make_sequences(df_train2, features_list, sequence_length)
    if X_train1.shape[0] == 0 or X_train2.shape[0] == 0:
        LOG.warning("Not enough training sequences for %s at length %d. Skipping.", ticker, sequence_length)
        return None
    X_train = np.concatenate([X_train1, X_train2], axis=0)
    X_val = make_sequences(df_validation, features_list, sequence_length)
    if X_val.shape[0] == 0:
        LOG.warning("Not enough validation sequences for %s. Skipping.", ticker)
        return None
    X_test = make_sequences(df_test, features_list, sequence_length)

    model_out_dir = os.path.join(out_root, ticker)
    os.makedirs(model_out_dir, exist_ok=True)

    model = GANExpert(
        seq_len=sequence_length,
        features=len(features_list),
        latent_dim=latent_dim,
        epochs=epochs,
        batch_sizes=batch_size,
        lr_G=learning_rate_g,
        lr_D=learning_rate_d,
        scaler_type=scaler,
        model_save_path=model_out_dir,
        resample_z=resample_z,
        negative_slope_G=negative_slope_G,
        negative_slope_D=negative_slope_D,
        backprop_steps=backprop_steps,
        base_channels_D=base_channels_D,
        base_channels_G=base_channels_G,
        lr_anomaly=learning_rate_anomaly,
        lambda_anom=lambda_anomaly,
        kernel_size_D=[3],  # prior runner default
    )

    LOG.info("Fitting on %d training sequences; validating on %d sequences.", X_train.shape[0], X_val.shape[0])
    model.fit(X_train=X_train, X_val=X_val, verbose=1)

    # Persist artifacts
    model.save_models(path=model_out_dir)
    save_json(model.training_history, os.path.join(model_out_dir, "training_history.json"))
    save_training_plot(model.training_history, os.path.join(model_out_dir, "training_history.png"))

    # Optional anomaly scoring on test segment
    if X_test.shape[0] > 0:
        details = model.detect_financial_anomalies(X_test, threshold_percentile=95.0, return_details=True)
        end_indices = df_test.index[sequence_length - 1:]
        end_dates = df_test.loc[end_indices, "date"].dt.strftime("%Y-%m-%d").tolist()
        scores = details["anomaly_scores"].tolist()
        labels = details["anomaly_labels"].tolist()
        end_dates_epoch = df_test.loc[end_indices, "date"].tolist()

        out_scores = pd.DataFrame({
            "window_end_date": end_dates[:len(scores)],
            "score": scores,
            "label": labels,
            "window_end_date_epoch": end_dates_epoch,
        })
        out_scores.to_csv(os.path.join(model_out_dir, "anomaly_scores.csv"), index=False)
        save_price_anomaly_plot(out_scores, df_test, os.path.join(model_out_dir, "price_anomalies.png"), ticker=ticker)

    return model


# ======================
# Command‑line interface
# ======================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Unified GAN train/optimize runner (refactored).")

    sub = ap.add_subparsers(dest="mode", required=True)

    # --- TRAIN subcommand ---
    tr = sub.add_parser("train", help="Train GANExpert on one or more tickers.")
    tr.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers, e.g., 'AAPL,MSFT'.")
    tr.add_argument("--tickers-file", type=str, default=None, help="Path to file with one ticker per line.")
    tr.add_argument("--data-dir", type=str, default="~/Data/OHLC")
    tr.add_argument("--out-dir", type=str, default="./runs")
    tr.add_argument("--epoch-unit", type=str, default="auto", choices=["auto", "s", "ms"])
    tr.add_argument("--string-date-format", type=str, default=None)

    # Model hyperparameters (parity with prior runner)
    tr.add_argument("--sequence-length", type=int, default=64)
    tr.add_argument("--epochs", type=int, default=200)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--latent-dim", type=int, default=100)
    tr.add_argument("--learning-rate-g", type=float, default=1.5e-4)
    tr.add_argument("--learning-rate-d", type=float, default=1e-5)
    tr.add_argument("--learning-rate-anomaly", type=float, default=5e-2)
    tr.add_argument("--scaler", type=str, default="robust", choices=["robust", "standard", "minmax"])
    tr.add_argument("--lambda-anomaly", type=float, default=0.9)
    tr.add_argument("--resample-z", type=int, default=3)
    tr.add_argument("--negative-slope-G", type=float, default=0.2)
    tr.add_argument("--negative-slope-D", type=float, default=0.2)
    tr.add_argument("--backpropagation-steps", type=int, default=50)
    tr.add_argument("--base-channels-G", type=int, default=16)
    tr.add_argument("--base-channels-D", type=int, default=64)

    # --- OPTIMIZE subcommand ---
    op = sub.add_parser("optimize", help="Run Optuna study for a single ticker.")
    op.add_argument("--ticker", type=str, required=True)
    op.add_argument("--data-dir", type=str, default="~/Data/OHLC")
    op.add_argument("--out-dir", type=str, default="./optuna_results")
    op.add_argument("--n-trials", type=int, default=100)
    op.add_argument("--timeout", type=int, default=3600)
    op.add_argument("--n-jobs", type=int, default=1)
    op.add_argument("--epoch-unit", type=str, default="auto", choices=["auto", "s", "ms"])
    op.add_argument("--string-date-format", type=str, default=None)
    op.add_argument("--study-name", type=str, default=None, help="Optional Optuna study name.")
    op.add_argument("--train-best", action="store_true", help="Train final model with best hyperparameters.")

    return ap


def main(argv: Optional[List[str]] = None) -> None:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    if args.mode == "train":
        tickers = parse_tickers_arg(args.tickers, args.tickers_file)
        data_dir = _expand(args.data_dir)
        out_root = _expand(args.out_dir)
        os.makedirs(out_root, exist_ok=True)

        last_model = None
        for i, ticker in enumerate(tickers, 1):
            LOG.info("=== [%d/%d] %s ===", i, len(tickers), ticker)
            try:
                last_model = train_one_ticker(
                    ticker=ticker,
                    data_dir=data_dir,
                    out_root=out_root,
                    sequence_length=args.sequence_length,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    latent_dim=args.latent_dim,
                    learning_rate_g=args.learning_rate_g,
                    learning_rate_d=args.learning_rate_d,
                    learning_rate_anomaly=args.learning_rate_anomaly,
                    scaler=args.scaler,
                    lambda_anomaly=args.lambda_anomaly,
                    resample_z=args.resample_z,
                    negative_slope_G=args.negative_slope_G,
                    negative_slope_D=args.negative_slope_D,
                    backprop_steps=args.backpropagation_steps,
                    base_channels_G=args.base_channels_G,
                    base_channels_D=args.base_channels_D,
                    epoch_unit=args.epoch_unit,
                    string_date_format=args.string_date_format,
                )
            except Exception:
                LOG.exception("Training failed for %s", ticker)

        if last_model is not None:
            final_dir = os.path.join(out_root, "final_model")
            os.makedirs(final_dir, exist_ok=True)
            last_model.save_models(path=final_dir)
            save_training_plot(last_model.training_history, os.path.join(final_dir, "training_history.png"))
            LOG.info("Final model saved to: %s", final_dir)
        else:
            LOG.warning("No models were trained; please check data ranges and CSV formats.")

    elif args.mode == "optimize":
        data_dir = _expand(args.data_dir)
        out_dir = _expand(os.path.join(args.out_dir, args.ticker))
        os.makedirs(out_dir, exist_ok=True)

        try:
            study = GANOptunaStudy(
                data_dir=data_dir,
                ticker=args.ticker,
                out_dir=out_dir,
                n_trials=args.n_trials,
                timeout=args.timeout,
                n_jobs=args.n_jobs,
                epoch_unit=args.epoch_unit,
                string_date_format=args.string_date_format,
                fix_features_to_log_adj_close=True,
            )
            results = study.optimize(study_name=args.study_name)
            LOG.info("Best value: %s", results["best_value"])
            LOG.info("Best params: %s", results["best_params"])

            if args.train_best:
                LOG.info("Training best model...")
                study.train_best_model()
        except Exception:
            LOG.exception("Optuna optimization failed for %s", args.ticker)

    else:
        ap.print_help()


if __name__ == "__main__":
    main()
