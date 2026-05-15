"""Evaluate and compare DQN agent against rule-based agents and random baseline.

Usage:
    python evaluate_dqn.py                  # evaluate all agents
    python evaluate_dqn.py --episodes 20    # fewer episodes
    python evaluate_dqn.py --dqn-only       # only evaluate DQN
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import config
from agents import ALL_AGENTS
from agents.dqn_agent import DQNAgent
from environment import ObservationProcessor, make_env
from evaluation.metrics import EpisodeMetrics, build_results_dataframe, summary_table
from evaluation.visualize import generate_all_plots

CHECKPOINT_PATH = os.path.join(config.RESULTS_DIR, "dqn", "checkpoints", "dqn_best.pt")
RL_RESULTS_CSV  = os.path.join(config.RESULTS_DIR, "dqn", "dqn_comparison.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed",     type=int, default=config.BASE_SEED)
    parser.add_argument("--dqn-only", action="store_true")
    return parser.parse_args()


def evaluate_dqn(agent: DQNAgent, num_episodes: int, base_seed: int) -> list:
    """Run the trained DQN agent and collect metrics."""
    results = []
    for ep in range(num_episodes):
        seed = base_seed + ep
        env  = make_env(render=False, seed=seed)
        obs, _ = env.reset(seed=seed)

        state   = agent.reset_frame_stack(obs)
        metrics = EpisodeMetrics()
        processor = ObservationProcessor()
        processor.reset()

        done = False
        step = 0
        while not done:
            features = processor.process(obs)
            action   = agent.act(state, training=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = agent.push_frame(obs)
            metrics.step(reward, features)
            state = next_state
            step += 1

        env.close()
        results.append(metrics.summarise())

        print(
            f"  [DQN] Ep {ep + 1:2d}/{num_episodes} | "
            f"Reward: {results[-1]['total_reward']:7.1f} | "
            f"Completion: {results[-1]['tiles_visited_pct']:5.1f}%"
        )
    return results


def evaluate_rule_based(num_episodes: int, base_seed: int) -> dict:
    """Run all rule-based agents and collect metrics."""
    from evaluation.metrics import EpisodeMetrics
    all_results = {}

    for cls in ALL_AGENTS:
        agent = cls()
        processor = ObservationProcessor()
        ep_results = []

        print(f"\n  Evaluating {agent.name}...")
        for ep in range(num_episodes):
            seed = base_seed + ep
            env  = make_env(render=False, seed=seed)
            obs, _ = env.reset(seed=seed)
            processor.reset()
            agent.reset()

            metrics = EpisodeMetrics()
            done = False
            while not done:
                features = processor.process(obs)
                action   = agent.act(features)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                metrics.step(reward, features)
                if hasattr(agent, "observe_reward"):
                    agent.observe_reward(reward)

            env.close()
            ep_results.append(metrics.summarise())

        all_results[agent.name] = ep_results

    return all_results


def main() -> None:
    args = parse_args()

    # Load DQN agent
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"  No checkpoint found at {CHECKPOINT_PATH}")
        print("  Run train_dqn.py first.")
        return

    dqn = DQNAgent()
    dqn.load(CHECKPOINT_PATH)

    print(f"\n  Evaluating DQN agent ({args.episodes} episodes)...")
    dqn_results = evaluate_dqn(dqn, args.episodes, args.seed)

    all_results = {"DQN Agent": dqn_results}

    if not args.dqn_only:
        rule_results = evaluate_rule_based(args.episodes, args.seed)
        all_results.update(rule_results)

    df      = build_results_dataframe(all_results)
    summary = summary_table(df)

    os.makedirs(os.path.dirname(RL_RESULTS_CSV), exist_ok=True)
    df.to_csv(RL_RESULTS_CSV, index=False)

    print("\n  COMPARISON TABLE")
    print(summary.to_string())

    generate_all_plots(df, summary)
    print(f"\n  Results saved to {RL_RESULTS_CSV}")


if __name__ == "__main__":
    main()
