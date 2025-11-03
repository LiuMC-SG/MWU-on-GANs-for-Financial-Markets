import os
import glob
from typing import List, Optional

import numpy as np
import pandas as pd

def expand_path(p: str) -> str:
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
    Load OHLCV for `ticker` and return at least columns: Date(datetime64[ns]), and numeric OHLCV fields.
    Supports epoch timestamps (s/ms, auto-detected) and optional formatted strings via `string_date_format`.
    """
    data_dir = expand_path(data_dir)
    candidates = [
        os.path.join(data_dir, f"{ticker}.csv"),
        os.path.join(data_dir, f"{ticker.upper()}.csv"),
        *glob.glob(os.path.join(data_dir, f"*{ticker}*.csv")),
    ]
    path = next((c for c in candidates if os.path.isfile(c)), None)
    if path is None:
        raise FileNotFoundError(f"CSV for ticker '{ticker}' not found under {data_dir}")

    df = pd.read_csv(path)

    date_col = _find_column(df, ["date", "timestamp", "time", "epoch", "unix", "ts"]) or "Date"
    if date_col not in df.columns:
        raise ValueError("No date/timestamp column found in CSV.")

    # numeric epochs first
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

    # fallback to formatted strings if provided
    mask_str = ~mask_num
    if string_date_format and mask_str.any():
        dt2 = pd.to_datetime(
            raw.loc[mask_str],
            format=string_date_format,
            errors="coerce",
            utc=True,
        )
        dt.loc[mask_str] = dt2.dt.tz_convert(None)

    good = dt.notna()
    df = df.loc[good].copy()
    df["Date"] = dt.loc[good].astype("datetime64[ns]")

    # coerce other columns numeric
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out = df.dropna().sort_values("Date").reset_index(drop=True)
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

def parse_tickers_arg(tickers: Optional[str], tickers_file: Optional[str]) -> List[str]:
    if tickers and tickers_file:
        raise ValueError("Provide either explicit tickers or a tickers file, not both.")
    if tickers_file:
        with open(expand_path(tickers_file), "r", encoding="utf-8") as f:
            t = [line.strip() for line in f if line.strip()]
    elif tickers:
        t = [x.strip() for x in tickers.split(",") if x.strip()]
    else:
        raise ValueError("Please provide tickers or a tickers file.")
    return t
