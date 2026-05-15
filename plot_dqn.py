"""Generate training curve plots for the DQN agent.

Usage:
    python plot_dqn.py

Saves plots to results/dqn/:
    training_reward.png   — reward per episode over training
    training_loss.png     — average loss per episode
    epsilon_decay.png     — epsilon over training
    smoothed_reward.png   — smoothed reward curve (window=20)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

LOG_PATH = os.path.join(config.RESULTS_DIR, "dqn", "dqn_training_log.csv")
OUT_DIR  = os.path.join(config.RESULTS_DIR, "dqn")

sns.set_theme(style="whitegrid", palette="muted")


def smooth(values: list, window: int = 20) -> np.ndarray:
    """Rolling mean over a list of values."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_training_reward(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["episode"], df["total_reward"], alpha=0.3, color="steelblue", label="Raw")
    smoothed = smooth(df["total_reward"].tolist(), window=20)
    ax.plot(
        df["episode"].iloc[19:], smoothed,
        color="steelblue", linewidth=2, label="Smoothed (window=20)"
    )
    ax.set_title("DQN Training Reward per Episode", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "training_reward.png"), dpi=150)
    plt.close(fig)
    print("  Saved: training_reward.png")


def plot_training_loss(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["episode"], df["avg_loss"], alpha=0.4, color="coral", label="Raw")
    smoothed = smooth(df["avg_loss"].tolist(), window=20)
    if len(smoothed) > 0:
        ax.plot(
            df["episode"].iloc[len(df) - len(smoothed):], smoothed,
            color="coral", linewidth=2, label="Smoothed (window=20)"
        )
    ax.set_title("DQN Average Loss per Episode", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss (Smooth L1)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "training_loss.png"), dpi=150)
    plt.close(fig)
    print("  Saved: training_loss.png")


def plot_epsilon(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["episode"], df["epsilon"], color="seagreen", linewidth=2)
    ax.set_title("Epsilon Decay over Training", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "epsilon_decay.png"), dpi=150)
    plt.close(fig)
    print("  Saved: epsilon_decay.png")


def main() -> None:
    if not os.path.exists(LOG_PATH):
        print(f"  No training log found at {LOG_PATH}")
        print("  Run train_dqn.py first.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(LOG_PATH)
    print(f"  Loaded {len(df)} episodes from {LOG_PATH}\n")

    plot_training_reward(df)
    plot_training_loss(df)
    plot_epsilon(df)

    print(f"\n  All plots saved to {OUT_DIR}/")
    print(f"  Best reward so far: {df['total_reward'].max():.1f}")
    print(f"  Last 50 ep average: {df['total_reward'].tail(50).mean():.1f}")


if __name__ == "__main__":
    main()
