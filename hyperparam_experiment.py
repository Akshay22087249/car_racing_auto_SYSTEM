"""Hyperparameter experiment: compare different DQN configurations.

Runs multiple short training runs with different hyperparameter settings
and saves a comparison plot. The opdracht requires hyperparameter
experimentation and documentation.

Usage:
    python hyperparam_experiment.py
    python hyperparam_experiment.py --episodes 100
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from agents.dqn_agent import DQNAgent
from environment import make_env

OUT_DIR  = os.path.join(config.RESULTS_DIR, "dqn", "hyperparams")
WARMUP   = 50

sns.set_theme(style="whitegrid")

# Configurations to compare — edit these to experiment
CONFIGS = [
    {
        "name":          "Baseline (lr=1e-4, γ=0.99)",
        "lr":            1e-4,
        "gamma":         0.99,
        "epsilon_decay": 0.9995,
        "batch_size":    64,
    },
    {
        "name":          "High lr (lr=5e-4)",
        "lr":            5e-4,
        "gamma":         0.99,
        "epsilon_decay": 0.9995,
        "batch_size":    64,
    },
    {
        "name":          "Low gamma (γ=0.95)",
        "lr":            1e-4,
        "gamma":         0.95,
        "epsilon_decay": 0.9995,
        "batch_size":    64,
    },
    {
        "name":          "Fast decay (ε-decay=0.999)",
        "lr":            1e-4,
        "gamma":         0.99,
        "epsilon_decay": 0.999,
        "batch_size":    64,
    },
    {
        "name":          "Large batch (batch=128)",
        "lr":            1e-4,
        "gamma":         0.99,
        "epsilon_decay": 0.9995,
        "batch_size":    128,
    },
]


def run_config(cfg: dict, num_episodes: int, seed: int) -> list:
    """Train a DQN with the given config and return reward per episode."""
    agent = DQNAgent(
        lr             = cfg["lr"],
        gamma          = cfg["gamma"],
        epsilon_decay  = cfg["epsilon_decay"],
        batch_size     = cfg["batch_size"],
        buffer_capacity= 20_000,  # smaller buffer for quick experiments
    )

    rewards = []
    for ep in range(num_episodes):
        ep_seed = seed + ep
        env     = make_env(render=False, seed=ep_seed)
        obs, _  = env.reset(seed=ep_seed)
        state   = agent.reset_frame_stack(obs)

        total_reward = 0.0
        done = False
        step = 0

        while not done:
            action = 3 if step < WARMUP else agent.act(state, training=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated
            next_state = agent.push_frame(obs)

            if step >= WARMUP:
                agent.store(state, action, reward, next_state, done)
                agent.learn()

            state         = next_state
            total_reward += reward
            step         += 1

        env.close()
        rewards.append(total_reward)

    return rewards


def smooth(values: list, window: int = 10) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=150,
                        help="Episodes per config (default 150, ~15 min total)")
    parser.add_argument("--seed",     type=int, default=config.BASE_SEED)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    all_rewards = {}
    for cfg in CONFIGS:
        print(f"\n  Running: {cfg['name']}")
        rewards = run_config(cfg, args.episodes, args.seed)
        all_rewards[cfg["name"]] = rewards
        avg = np.mean(rewards[-20:])
        print(f"  Last 20 ep average: {avg:.1f}")

    # Save raw data
    df = pd.DataFrame(all_rewards)
    df.index.name = "episode"
    df.to_csv(os.path.join(OUT_DIR, "hyperparam_results.csv"))

    # Plot smoothed rewards
    fig, ax = plt.subplots(figsize=(11, 6))
    window = max(5, args.episodes // 15)
    for name, rewards in all_rewards.items():
        smoothed = smooth(rewards, window=window)
        episodes = range(window, len(rewards) + 1)
        ax.plot(list(episodes), smoothed, linewidth=2, label=name)

    ax.set_title("Hyperparameter Comparison — Smoothed Reward", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward (smoothed)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "hyperparam_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n  Plot saved to {out_path}")

    # Print summary table
    print("\n  SUMMARY (last 20 episodes avg reward)")
    print(f"  {'Config':<40s} {'Avg Reward':>12s}")
    print("  " + "-" * 55)
    for name, rewards in all_rewards.items():
        avg = np.mean(rewards[-20:])
        print(f"  {name:<40s} {avg:>12.1f}")


if __name__ == "__main__":
    main()
