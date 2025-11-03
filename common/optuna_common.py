import os
import json
import logging
from typing import Dict, Any, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.preprocessing import MinMaxScaler

# Reuse the same runner helpers that the project already uses
# (these imports exist in the uploaded optimizers)
from common.data_utils import read_ohlc_csv, slice_by_date, make_sequences, expand_path as _expand
from common.plot_utils import save_training_plot
from common.io_utils import save_json

LOG = logging.getLogger("optuna_common")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class BaseOptunaOptimizer:
    """A reusable Optuna optimizer skeleton.

    Subclasses must override:
      - build_model(params: Dict[str, Any], trial_dir: str) -> Any
      - sample_hparams(trial: optuna.Trial) -> Dict[str, Any]
      - model_eval_loss(model, X_val: np.ndarray) -> float  (optional; default uses _evaluate_tanogan_loss if present)
    """

    def __init__(
        self,
        data_dir: str,
        ticker: str,
        out_dir: str,
        n_trials: int = 100,
        timeout: int = 3600,
        n_jobs: int = 1,
        epoch_unit: str = "auto",
        string_date_format: Optional[str] = None,
        fix_features_to_log_adj_close: bool = True,
    ) -> None:
        self.data_dir = _expand(data_dir)
        self.ticker = ticker
        self.out_dir = _expand(out_dir)
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.epoch_unit = epoch_unit
        self.string_date_format = string_date_format
        self.fix_features_to_log_adj_close = fix_features_to_log_adj_close

        os.makedirs(self.out_dir, exist_ok=True)
        self._load_data()

        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: float = float("inf")

    # ---------- data helpers (shared) ----------
    def _load_data(self) -> None:
        LOG.info(f"Loading data for ticker: {self.ticker}")
        df_all = read_ohlc_csv(
            self.data_dir,
            self.ticker,
            epoch_unit=self.epoch_unit,
            string_date_format=self.string_date_format,
        )
        if self.fix_features_to_log_adj_close:
            # Match the single-feature usage in existing code paths
            if "log_adj_close" not in df_all.columns:
                raise ValueError("Column 'log_adj_close' is missing in the input CSV.")
            df_all = df_all[["Date", "log_adj_close"]]

        # Train/val/test windows copied from project convention
        self.df_train1 = slice_by_date(df_all, "2003-01-01", "2006-12-31")
        self.df_train2 = slice_by_date(df_all, "2010-01-01", "2017-12-31")
        self.df_validation = slice_by_date(df_all, "2007-01-01", "2009-12-31")
        self.df_test = slice_by_date(df_all, "2021-07-01", "2025-07-31")

        features_list = df_all.columns.tolist()
        features_list.remove("Date")
        self.features_list = features_list

        LOG.info(
            "Data loaded. train1=%d, train2=%d, val=%d, test=%d",
            len(self.df_train1), len(self.df_train2),
            len(self.df_validation), len(self.df_test),
        )

    def _create_sequences(self, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        X_train1 = make_sequences(self.df_train1, self.features_list, sequence_length)
        X_train2 = make_sequences(self.df_train2, self.features_list, sequence_length)
        X_val = make_sequences(self.df_validation, self.features_list, sequence_length)

        if X_train1.shape[0] == 0 or X_train2.shape[0] == 0:
            raise ValueError(f"Insufficient training data for sequence length {sequence_length}")
        if X_val.shape[0] == 0:
            raise ValueError(f"Insufficient validation data for sequence length {sequence_length}")

        X_train = np.concatenate([X_train1, X_train2], axis=0)
        return X_train, X_val

    # ---------- hooks to override ----------
    def build_model(self, params: Dict[str, Any], trial_dir: str):
        raise NotImplementedError

    def sample_hparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        raise NotImplementedError

    def model_eval_loss(self, model, X_val: np.ndarray) -> float:
        """Default evaluator tries TAnoGAN loss, falls back to last g_loss."""
        try:
            import torch  # local import to avoid hard dependency at import time
            X_val_tensor = torch.FloatTensor(model._preprocess_financial_data(X_val)).to(model.device)  # type: ignore[attr-defined]
            return float(model._evaluate_tanogan_loss(X_val_tensor).mean().item())  # type: ignore[attr-defined]
        except Exception as e:
            LOG.warning("Could not compute TAnoGAN loss, using generator loss: %s", e)
            hist = getattr(model, "training_history", {})
            g_loss = hist.get("g_loss", [])
            return float(g_loss[-1]) if g_loss else float("inf")

    # ---------- objective & optimization ----------
    def objective(self, trial: optuna.Trial) -> float:
        params = self.sample_hparams(trial)
        seq_len = params["sequence_length"]
        X_train, X_val = self._create_sequences(seq_len)

        trial_dir = os.path.join(self.out_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)

        model = self.build_model(params, trial_dir)

        LOG.info("Trial %d params: %s", trial.number, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()})
        model.fit(X_train=X_train, X_val=X_val, verbose=0, log_every=100)  # project convention

        val_loss = self.model_eval_loss(model, X_val)

        # Persist typical artifacts
        save_json({"trial_number": trial.number, "params": params, "validation_loss": val_loss,
                   "training_history": getattr(model, "training_history", {})},
                  os.path.join(trial_dir, "trial_results.json"))
        save_training_plot(getattr(model, "training_history", {}),
                           os.path.join(trial_dir, "training_history.png"))

        return float(val_loss)

    def optimize(self, study_name: Optional[str] = None) -> Dict[str, Any]:
        storage = None  # file / RDB storage can be added later
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
            storage=storage,
            load_if_exists=False,
        )
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout, n_jobs=self.n_jobs)

        self.best_params = dict(study.best_trial.params)
        self.best_value = float(study.best_value)

        # Summary
        results = {
            "ticker": self.ticker,
            "n_trials": self.n_trials,
            "n_completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "n_failed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            "best_value": self.best_value,
            "best_params": self.best_params,
        }
        save_json(results, os.path.join(self.out_dir, "study_summary.json"))
        return results

    # ---------- best-model training & visualization ----------
    def _prepare_test_sequences(self, sequence_length: int) -> np.ndarray:
        if self.df_test.empty:
            return np.empty((0,))  # nothing to do
        return make_sequences(self.df_test, self.features_list, sequence_length)

    def _save_price_anomaly_plot(self, out_scores: pd.DataFrame, df_test: pd.DataFrame, out_png: str) -> None:
        """Generic visualization used by the original scripts."""
        out_scores = out_scores.copy()
        out_scores["score_normalized"] = MinMaxScaler(feature_range=(0, 1)).fit_transform(out_scores[["score"]])
        df_test = df_test.copy()
        if "log_adj_close" in df_test.columns:
            df_test["adj_close_normalized"] = MinMaxScaler(feature_range=(0, 1)).fit_transform(df_test[["log_adj_close"]])

        fig, ax = plt.subplots(figsize=(12, 6))
        if "adj_close_normalized" in df_test:
            ax.plot(df_test["Date"], df_test["adj_close_normalized"], label="Actual Prices", alpha=0.7)
        ax.plot(out_scores["window_end_date_epoch"], out_scores["score_normalized"], label="Anomaly Scores", alpha=0.8)
        ax.set_title(f"Price Anomalies - {self.ticker}")
        ax.set_xlabel("Date"); ax.set_ylabel("Normalized Values")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight", dpi=150)
        plt.close(fig)

    def train_best_model(self):
        if not self.best_params:
            raise RuntimeError("No best parameters available. Run optimize() first.")
        params = self.best_params
        seq_len = params["sequence_length"]
        X_train, X_val = self._create_sequences(seq_len)
        X_test = self._prepare_test_sequences(seq_len)

        final_model_dir = os.path.join(self.out_dir, "best_model")
        os.makedirs(final_model_dir, exist_ok=True)

        model = self.build_model(params, final_model_dir)
        model.fit(X_train=X_train, X_val=X_val, verbose=0, log_every=100)

        # Save artifacts
        save_training_plot(getattr(model, "training_history", {}), os.path.join(final_model_dir, "training_history.png"))
        save_json(getattr(model, "training_history", {}), os.path.join(final_model_dir, "training_history.json"))
        save_json(params, os.path.join(final_model_dir, "best_params.json"))

        # Optional anomaly scoring on test set (if available)
        if X_test.shape[0] > 0 and hasattr(model, "detect_financial_anomalies"):
            details = model.detect_financial_anomalies(X_test, threshold_percentile=95.0, return_details=True)
            end_indices = self.df_test.index[seq_len - 1:]
            end_dates = self.df_test.loc[end_indices, "Date"].dt.strftime("%Y-%m-%d").tolist()
            end_dates_epoch = self.df_test.loc[end_indices, "Date"].tolist()

            out_scores = pd.DataFrame({
                "window_end_date": end_dates[: len(details["anomaly_scores"])],
                "score": details["anomaly_scores"].tolist(),
                "label": details["anomaly_labels"].tolist(),
                "window_end_date_epoch": end_dates_epoch,
            })
            out_scores.to_csv(os.path.join(final_model_dir, "anomaly_scores.csv"), index=False)
            self._save_price_anomaly_plot(out_scores, self.df_test, os.path.join(final_model_dir, "price_anomalies.png"))

        LOG.info("Best model trained and saved to: %s", final_model_dir)
        return model
