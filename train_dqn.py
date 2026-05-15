"""Train the DQN agent on CarRacing-v3.

Usage:
    python train_dqn.py                        # default settings
    python train_dqn.py --episodes 1000        # more episodes
    python train_dqn.py --resume               # continue from checkpoint
    python train_dqn.py --lr 0.0001 --gamma 0.99

The script saves:
    results/dqn/checkpoints/dqn_best.pt    — best model so far
    results/dqn/checkpoints/dqn_latest.pt  — latest checkpoint
    results/dqn/dqn_training_log.csv       — reward and loss per episode
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

import config
from agents.dqn_agent import DQNAgent
from environment import make_env

CHECKPOINT_DIR = os.path.join(config.RESULTS_DIR, "dqn", "checkpoints")
LOG_PATH       = os.path.join(config.RESULTS_DIR, "dqn", "dqn_training_log.csv")

WARMUP_FRAMES = 50  # skip zoom-in animation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on CarRacing-v3.")
    parser.add_argument("--episodes",       type=int,   default=500)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--gamma",          type=float, default=0.99)
    parser.add_argument("--epsilon-start",  type=float, default=1.0)
    parser.add_argument("--epsilon-end",    type=float, default=0.05)
    parser.add_argument("--epsilon-decay",  type=float, default=0.9995)
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--target-update",  type=int,   default=1000)
    parser.add_argument("--buffer-size",    type=int,   default=50_000)
    parser.add_argument("--seed",           type=int,   default=config.BASE_SEED)
    parser.add_argument("--resume",         action="store_true")
    parser.add_argument("--render",         action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    agent = DQNAgent(
        lr             = args.lr,
        gamma          = args.gamma,
        epsilon_start  = args.epsilon_start,
        epsilon_end    = args.epsilon_end,
        epsilon_decay  = args.epsilon_decay,
        batch_size     = args.batch_size,
        target_update  = args.target_update,
        buffer_capacity= args.buffer_size,
    )

    latest_path = os.path.join(CHECKPOINT_DIR, "dqn_latest.pt")
    best_path   = os.path.join(CHECKPOINT_DIR, "dqn_best.pt")

    if args.resume and os.path.exists(latest_path):
        agent.load(latest_path)

    log_rows   = []
    best_reward = -float("inf")

    print(f"\n  Training DQN on {config.ENV_ID}")
    print(f"  Episodes: {args.episodes} | Seed: {args.seed}\n")

    for ep in range(1, args.episodes + 1):
        seed = args.seed + ep
        env  = make_env(render=args.render, seed=seed)
        obs, _ = env.reset(seed=seed)

        state        = agent.reset_frame_stack(obs)
        total_reward = 0.0
        total_loss   = 0.0
        loss_count   = 0
        step         = 0
        done         = False
        t0           = time.time()

        while not done:
            # Skip warmup animation — just give gas
            if step < WARMUP_FRAMES:
                action = 3  # GAS
            else:
                action = agent.act(state, training=True)

            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = agent.push_frame(obs_next)

            # Only store and learn after warmup
            if step >= WARMUP_FRAMES:
                agent.store(state, action, reward, next_state, done)
                loss = agent.learn()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1

            state         = next_state
            total_reward += reward
            step         += 1

        env.close()

        avg_loss = total_loss / max(loss_count, 1)
        elapsed  = time.time() - t0

        # Save checkpoints
        agent.save(latest_path)
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(best_path)
            star = " ★ NEW BEST"
        else:
            star = ""

        # Log
        log_rows.append({
            "episode":      ep,
            "total_reward": round(total_reward, 2),
            "avg_loss":     round(avg_loss, 6),
            "epsilon":      round(agent.epsilon, 4),
            "steps":        step,
            "elapsed_s":    round(elapsed, 1),
        })

        print(
            f"  Ep {ep:4d}/{args.episodes} | "
            f"Reward: {total_reward:7.1f} | "
            f"Loss: {avg_loss:.5f} | "
            f"Eps: {agent.epsilon:.3f} | "
            f"Steps: {step:4d} | "
            f"{elapsed:.1f}s{star}"
        )

        # Save log every 10 episodes
        if ep % 10 == 0:
            pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)

    pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)
    print(f"\n  Training done. Best reward: {best_reward:.1f}")
    print(f"  Log saved to {LOG_PATH}")


if __name__ == "__main__":
    train(parse_args())
