#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared plotting utilities for training history and anomaly visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def save_training_plot(history: dict, out_png: str) -> None:
    """
    Plot GAN training history (losses and discriminator accuracy)
    
    Args:
        history: Dictionary with keys 'g_loss', 'd_loss_real', 'd_loss_fake', 'd_acc', 'tanogan_loss'
        out_png: Output file path
    """
    g = history.get("g_loss", [])
    d_real = history.get("d_loss_real", [])
    d_fake = history.get("d_loss_fake", [])
    d_acc = history.get("d_acc", [])
    tanogan_loss = history.get("tanogan_loss", [])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Training losses
    axes[0, 0].plot(g, label="Generator Loss")
    axes[0, 0].plot(d_real, label="D Loss (Real)")
    axes[0, 0].plot(d_fake, label="D Loss (Fake)")
    axes[0, 0].set_title("Training Losses")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Discriminator accuracy
    axes[0, 1].plot(d_acc, label="Discriminator Accuracy")
    axes[0, 1].set_title("Discriminator Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # TAnoGAN loss (if available)
    if tanogan_loss:
        axes[1, 0].plot(tanogan_loss, label="TAnoGAN Loss")
        axes[1, 0].set_title("TAnoGAN Loss")
        axes[1, 0].set_xlabel("Epoch (x10)")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].axis("off")

    # Loss difference
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


def save_price_anomaly_plot(
    out_scores: pd.DataFrame, 
    df_test: pd.DataFrame, 
    out_png: str,
    ticker: str = ""
) -> None:
    """
    Plot price series with anomaly scores overlay
    
    Args:
        out_scores: DataFrame with columns 'score', 'window_end_date_epoch'
        df_test: DataFrame with columns 'date', 'log_adj_close'
        out_png: Output file path
        ticker: Stock ticker for title
    """
    # Normalize scores and prices to [0,1]
    out_scores_normalized = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        out_scores[["score"]]
    )
    out_scores["score_normalized"] = out_scores_normalized

    df_test_normalized = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        df_test[["log_adj_close"]]
    )
    df_test["adj_close_normalized"] = df_test_normalized

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        df_test["date"], 
        df_test["adj_close_normalized"], 
        label="Actual Prices", 
        alpha=0.7
    )
    ax.plot(
        out_scores["window_end_date_epoch"],
        out_scores["score_normalized"],
        color="red",
        label="Anomaly Scores",
        alpha=0.8
    )
    
    title = f"Price Anomalies - {ticker}" if ticker else "Price Anomalies"
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("Normalized Values")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)