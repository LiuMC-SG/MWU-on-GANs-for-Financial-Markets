#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import argparse
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
from sklearn.preprocessing import MinMaxScaler
matplotlib.use("Agg")  # headless plotting for SLURM
import matplotlib.pyplot as plt

# Your existing GAN implementation must be importable here.
from lstm_cnn_gan_model import LSTMCNNGANExpert  # noqa: E402

LOG = logging.getLogger("train_gan")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# FEATURES_DEFAULT = ["Open", "High", "Low", "Close", "Adj Close","Volume"]
FEATURES_DEFAULT = ["Adj Close"]

def _expand(p: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(p)))

def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in norm:
            return norm[key]
    return None

def read_ohlc_csv(
    data_dir: str,
    ticker: str,
    epoch_unit: str = "auto",
    string_date_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load OHLCV for `ticker` and return columns: Date(datetime64[ns]), Open, High, Low, Close, Volume.

    - `epoch_unit`: 'auto' (default), 's', or 'ms'
    - `string_date_format`: if provided, any non-numeric timestamps are parsed with this exact
      strptime format (e.g., '%Y-%m-%d', '%Y-%m-%d %H:%M:%S'). If omitted, non-numeric rows are dropped.
    """
    data_dir = _expand(data_dir)
    candidates = [
        os.path.join(data_dir, f"{ticker}.csv"),
        os.path.join(data_dir, f"{ticker.upper()}.csv"),
        *glob.glob(os.path.join(data_dir, f"*{ticker}*.csv")),
    ]
    path = next((c for c in candidates if os.path.isfile(c)), None)
    if path is None:
        raise FileNotFoundError(f"CSV for ticker '{ticker}' not found under {data_dir}")

    df = pd.read_csv(path)

    date_col = _find_column(df, ["date", "timestamp", "time", "epoch", "unix", "ts"])

    # ---- primary: parse numeric epochs only ----
    raw = df[date_col]
    s = pd.to_numeric(raw, errors="coerce").astype("float64")
    s[~np.isfinite(s)] = np.nan
    mask_num = s.notna()

    if epoch_unit not in {"auto", "s", "ms"}:
        raise ValueError("epoch_unit must be one of {'auto','s','ms'}")

    if epoch_unit == "auto":
        unit = "ms" if mask_num.any() and float(s[mask_num].median()) > 1e11 else "s"
    else:
        unit = epoch_unit

    dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if mask_num.any():
        dt.loc[mask_num] = pd.to_datetime(s.loc[mask_num], unit=unit, utc=True).dt.tz_convert(None)

    # ---- optional: explicitly parse remaining string timestamps with a known format ----
    mask_str = ~mask_num
    if string_date_format:
        if mask_str.any():
            dt2 = pd.to_datetime(
                raw.loc[mask_str],
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
        LOG.warning("%s: Dropping %d row(s) with unparseable timestamps in '%s'.%s",
                    os.path.basename(path), int((~good).sum()), date_col, hint)

    df = df.loc[good].copy()
    df["date"] = dt.loc[good].astype("datetime64[ns]")

    columns = df.columns.tolist()
    for col in columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out = df.copy()
    out.rename(columns={"date": "Date"}, inplace=True)

    before = len(out)
    out = out.dropna().sort_values("Date").reset_index(drop=True)
    dropped = before - len(out)
    if dropped > 0:
        LOG.warning("%s: Dropped %d row(s) with NaN in OHLCV after timestamp parsing.",
                    os.path.basename(path), dropped)

    return out

def slice_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    m = (df["Date"] >= start_dt) & (df["Date"] <= end_dt)
    return df.loc[m].reset_index(drop=True)

def make_sequences(df: pd.DataFrame, features: List[str], seq_len: int) -> np.ndarray:
    arr = df[features].to_numpy(dtype=np.float32)
    n = arr.shape[0]
    if n < seq_len:
        return np.zeros((0, seq_len, len(features)), dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(arr, (seq_len, arr.shape[1]))
    windows = windows[:, 0, :, :]
    return windows.copy()

def save_training_plot(history: dict, out_png: str) -> None:
    g = history.get("g_loss", [])
    d_real = history.get("d_loss_real", [])
    d_fake = history.get("d_loss_fake", [])
    d_acc = history.get("d_acc", [])
    tanogan_loss = history.get("tanogan_loss", [])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(g, label="Generator Loss")
    axes[0, 0].plot(d_real, label="D Loss (Real)")
    axes[0, 0].plot(d_fake, label="D Loss (Fake)")
    axes[0, 0].set_title("Training Losses")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(d_acc, label="Discriminator Accuracy")
    axes[0, 1].set_title("Discriminator Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if tanogan_loss:
        axes[1, 0].plot(tanogan_loss, label="TAnoGAN Loss")
        axes[1, 0].set_title("TAnoGAN Loss")
        axes[1, 0].set_xlabel("Epoch (x10)")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].axis("off")

    if g and d_real and d_fake:
        g_arr = np.array(g)
        d_avg = (np.array(d_real) + np.array(d_fake)) / 2.0
        axes[1, 1].plot(g_arr - d_avg, label="G_loss - D_avg")
        axes[1, 1].axhline(0.0, linestyle="--", alpha=0.6)
        axes[1, 1].set_title("Loss Difference")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("ΔLoss")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)

def save_price_anomaly_plot(out_scores: pd.DataFrame, df_test: pd.DataFrame, out_png: str) -> None:
    out_scores_minmax_normalized = MinMaxScaler(feature_range=(0,1)).fit_transform(out_scores[["score"]])
    out_scores["score_normalized"] = out_scores_minmax_normalized

    df_test_minmax_normalized = MinMaxScaler(feature_range=(0,1)).fit_transform(df_test[["log_adj_close"]])
    df_test["adj_close_normalized"] = df_test_minmax_normalized

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_test["Date"], df_test["adj_close_normalized"], label="Actual Prices")
    ax.plot(out_scores["window_end_date_epoch"], out_scores["score_normalized"], color="red", label="Anomalies")
    ax.set_title("Price Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)

def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def parse_tickers(args) -> List[str]:
    if args.tickers and args.tickers_file:
        raise ValueError("Provide either --tickers or --tickers-file, not both.")
    if args.tickers_file:
        with open(_expand(args.tickers_file), "r", encoding="utf-8") as f:
            t = [line.strip() for line in f if line.strip()]
    elif args.tickers:
        t = [x.strip() for x in args.tickers.split(",") if x.strip()]
    else:
        raise ValueError("Please provide --tickers or --tickers-file.")
    return t

def main():
    ap = argparse.ArgumentParser(description="TAnoGAN SLURM-compatible trainer (per-ticker).")
    ap.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers, e.g. 'AAPL,MSFT,TSLA'")
    ap.add_argument("--tickers-file", type=str, default=None, help="Path to a file with one ticker per line.")
    ap.add_argument("--data-dir", type=str, default="~/Data/OHLC", help="Directory with {TICKER}.csv files.")
    ap.add_argument("--out-dir", type=str, default="./runs", help="Directory to write models, plots, and artifacts.")

    # Model/Training hyperparameters
    ap.add_argument("--sequence-length", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--latent-dim", type=int, default=100)
    ap.add_argument("--learning-rate-g", type=float, default=1.5e-4)
    ap.add_argument("--learning-rate-d", type=float, default=1e-5)
    ap.add_argument("--learning-rate-anomaly", type=float, default=0.05)
    ap.add_argument("--scaler", type=str, default="robust", choices=["robust", "standard", "minmax"])
    ap.add_argument("--lambda-anomaly", type=float, default=0.9)
    ap.add_argument("--resample-z", type=int, default=3)
    ap.add_argument("--negative-slope-G", type=float, default=0.2)
    ap.add_argument("--negative-slope-D", type=float, default=0.2)
    ap.add_argument("--validation-split", type=float, default=0.2)
    ap.add_argument("--backpropagation-steps", type=int, default=50)
    ap.add_argument("--lstm-hidden-size-G", type=int, default=256)
    ap.add_argument("--lstm-num-layers-G", type=int, default=2)
    ap.add_argument("--dropout-G", type=float, default=0.2)
    ap.add_argument("--lstm-hidden-size-D", type=int, default=256)
    ap.add_argument("--lstm-num-layers-D", type=int, default=2)
    ap.add_argument("--dropout-D", type=float, default=0.2)

    # Extra
    ap.add_argument("--epoch-unit", type=str, default="auto", choices=["auto", "s", "ms"],
                    help="Unit of the CSV epoch timestamps; 'auto' detects seconds vs. milliseconds by magnitude.")
    ap.add_argument(
        "--string-date-format",
        type=str,
        default=None,
        help="Optional strptime format for non-numeric timestamps (e.g., '%Y-%m-%d'). "
            "If omitted, non-numeric timestamp rows are dropped."
    )

    args = ap.parse_args()

    tickers = parse_tickers(args)
    data_dir = _expand(args.data_dir)
    out_root = _expand(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    LOG.info("Job started. Tickers: %s", tickers)
    LOG.info("Data dir: %s", data_dir)
    LOG.info("Output dir: %s", out_root)
    LOG.info("Epoch unit: %s", args.epoch_unit)

    last_model: Optional[LSTMCNNGANExpert] = None

    for i, ticker in enumerate(tickers, 1):
        try:
            LOG.info("=== [%d/%d] Ticker: %s ===", i, len(tickers), ticker)
            df_all = read_ohlc_csv(
                data_dir,
                ticker,
                epoch_unit=args.epoch_unit,
                string_date_format=args.string_date_format
            )

            df_train1 = slice_by_date(df_all, "2003-01-01", "2006-12-31")
            df_train2 = slice_by_date(df_all, "2010-01-01", "2021-06-30")
            df_test  = slice_by_date(df_all, "2021-07-01", "2023-08-30")
            df_validation = slice_by_date(df_all, "2007-01-01", "2009-12-31")

            # Validate columns
            # for f in FEATURES_DEFAULT:
            #     if f not in df_train1.columns or f not in df_train2.columns:
            #         raise ValueError(f"Training data missing required column '{f}' for {ticker}")

            # features_list=["Log_Adj_Close"]
            features_list = df_all.columns.tolist()
            features_list.remove("Date")

            X_train1 = make_sequences(df_train1, features_list, args.sequence_length)
            if X_train1.shape[0] == 0:
                LOG.warning("Not enough training data for %s in [%s, %s] to form sequences of length %d. Skipping.",
                            ticker, "2003-01-01", "2006-12-31", args.sequence_length)
                continue

            X_train2 = make_sequences(df_train2, features_list, args.sequence_length)
            if X_train2.shape[0] == 0:
                LOG.warning("Not enough training data for %s in [%s, %s] to form sequences of length %d. Skipping.",
                            ticker, "2010-01-01", "2021-06-30", args.sequence_length)
                continue

            X_train = np.concatenate([X_train1, X_train2], axis=0)

            X_validation = make_sequences(df_validation, features_list, args.sequence_length)
            if X_validation.shape[0] == 0:
                LOG.warning("Not enough validation data for %s in [%s, %s] to form sequences of length %d. Skipping.",
                            ticker, "2007-01-01", "2009-12-31", args.sequence_length)
                continue

            X_test = make_sequences(df_test, features_list, args.sequence_length)
            if X_test.shape[0] == 0:
                LOG.warning("No test sequences for %s in [%s, %s]. Anomaly scoring will be skipped.",
                            ticker, "2021-01-01", "2023-08-30")

            model_out_dir = os.path.join(out_root, ticker)
            os.makedirs(model_out_dir, exist_ok=True)

            model = LSTMCNNGANExpert(
                seq_len=args.sequence_length,
                features=len(features_list),
                latent_dim=args.latent_dim,
                epochs=args.epochs,
                batch_sizes=args.batch_size,
                lr_G=args.learning_rate_g,
                lr_D=args.learning_rate_d,
                scaler_type=args.scaler,
                model_save_path=model_out_dir,
                resample_z=args.resample_z,
                negative_slope_G=args.negative_slope_G,
                negative_slope_D=args.negative_slope_D,
                backprop_steps=args.backpropagation_steps,
                lr_anomaly=args.learning_rate_anomaly,
                lambda_anom=args.lambda_anomaly,
                lstm_hidden_size_G=args.lstm_hidden_size_G,
                lstm_num_layers_G=args.lstm_num_layers_G,
                dropout_G=args.dropout_G,
                lstm_hidden_size_D=args.lstm_hidden_size_D,
                lstm_num_layers_D=args.lstm_num_layers_D,
                dropout_D=args.dropout_D
            )

            LOG.info("Training on %d sequences for %s", X_train.shape[0], ticker)
            model.fit(X_train=X_train, X_val=X_validation, verbose=1)

            # Save per-ticker artifacts
            model.save_models(path=model_out_dir)
            save_json(model.training_history, os.path.join(model_out_dir, "training_history.json"))
            save_training_plot(model.training_history, os.path.join(model_out_dir, "training_history.png"))
            LOG.info("Saved training history for %s.", ticker)

            # Anomaly scoring on the abnormal window
            if X_test.shape[0] > 0:
                details = model.detect_financial_anomalies(X_test, threshold_percentile=95.0, return_details=True)
                end_indices = df_test.index[args.sequence_length - 1:]
                end_dates = df_test.loc[end_indices, "Date"].dt.strftime("%Y-%m-%d").tolist()
                scores = details["anomaly_scores"].tolist()
                labels = details["anomaly_labels"].tolist()
                end_dates_epoch = df_test.loc[end_indices, "Date"].tolist()

                out_scores = pd.DataFrame({
                    "window_end_date": end_dates[:len(scores)],
                    "score": scores,
                    "label": labels,
                    "window_end_date_epoch": end_dates_epoch
                })
                out_scores.to_csv(os.path.join(model_out_dir, "anomaly_scores.csv"), index=False)
                LOG.info("Saved anomaly scores for %s.", ticker)

                save_price_anomaly_plot(out_scores, df_test, os.path.join(model_out_dir, "price_anomalies.png"))

            last_model = model

        except Exception as e:
            LOG.exception("Error while processing %s: %s", ticker, e)
            continue

    # Save final model
    if last_model is not None:
        final_dir = os.path.join(out_root, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        last_model.save_models(path=final_dir)
        save_training_plot(last_model.training_history, os.path.join(final_dir, "training_history.png"))
        LOG.info("Final model saved to: %s", final_dir)
    else:
        LOG.warning("No models were trained; please check data ranges and CSV formats.")

if __name__ == "__main__":
    main()
