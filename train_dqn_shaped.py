"""Train the DQN agent on CarRacing-v3 with reward shaping.

The shaped reward per step is:
    shaped = (1 - w) * env_reward + w * on_track_bonus
where w = --track-weight and on_track_bonus is 1.0 when the car patch is on
road, else 0.0. The agent learns from `shaped`; we also log the raw env
reward per episode so the underlying score is visible during training.

Usage:
    python train_dqn_shaped.py --track-weight 0.5 --output-tag w50
    python train_dqn_shaped.py --track-weight 0.7 --output-tag w70 --episodes 500

Outputs go to:
    results/dqn/shaped/{tag}/checkpoints/dqn_best.pt   (selected by raw env reward)
    results/dqn/shaped/{tag}/checkpoints/dqn_latest.pt
    results/dqn/shaped/{tag}/dqn_training_log.csv
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd

import config
from agents.dqn_agent import DQNAgent
from environment import compute_on_track, make_env

WARMUP_FRAMES = 50  # skip zoom-in animation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on CarRacing-v3 with on-track reward shaping.")
    parser.add_argument("--track-weight",   type=float, required=True,
                        help="Weight on on-track bonus in [0, 1]. Env reward weight = 1 - this.")
    parser.add_argument("--output-tag",     type=str,   required=True,
                        help="Folder name under results/dqn/shaped/ for this run's outputs.")
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
    args = parser.parse_args()
    if not 0.0 <= args.track_weight <= 1.0:
        parser.error("--track-weight must be in [0.0, 1.0]")
    return args


def train(args: argparse.Namespace) -> None:
    run_dir        = os.path.join(config.RESULTS_DIR, "dqn", "shaped", args.output_tag)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    log_path       = os.path.join(run_dir, "dqn_training_log.csv")
    os.makedirs(checkpoint_dir, exist_ok=True)

    w = args.track_weight

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

    latest_path = os.path.join(checkpoint_dir, "dqn_latest.pt")
    best_path   = os.path.join(checkpoint_dir, "dqn_best.pt")

    if args.resume and os.path.exists(latest_path):
        agent.load(latest_path)

    log_rows    = []
    best_reward = -float("inf")

    print(f"\n  Training DQN on {config.ENV_ID} with reward shaping")
    print(f"  track_weight = {w:.2f}  →  shaped = {1-w:.2f}*env + {w:.2f}*on_track")
    print(f"  Episodes: {args.episodes} | Seed: {args.seed} | Output: {run_dir}\n")

    for ep in range(1, args.episodes + 1):
        seed = args.seed + ep
        env  = make_env(render=args.render, seed=seed)
        obs, _ = env.reset(seed=seed)

        state              = agent.reset_frame_stack(obs)
        total_env_reward   = 0.0
        total_shaped_reward= 0.0
        on_track_steps     = 0
        total_loss         = 0.0
        loss_count         = 0
        step               = 0
        done               = False
        t0                 = time.time()

        while not done:
            # Skip warmup animation — just give gas
            if step < WARMUP_FRAMES:
                action = 3  # GAS
            else:
                action = agent.act(state, training=True)

            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = agent.push_frame(obs_next)

            on_track = compute_on_track(obs_next)
            bonus    = 1.0 if on_track else 0.0
            shaped   = (1.0 - w) * float(reward) + w * bonus

            # Only store and learn after warmup
            if step >= WARMUP_FRAMES:
                agent.store(state, action, shaped, next_state, done)
                loss = agent.learn()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1

            state               = next_state
            total_env_reward   += float(reward)
            total_shaped_reward+= shaped
            if on_track:
                on_track_steps += 1
            step               += 1

        env.close()

        avg_loss     = total_loss / max(loss_count, 1)
        elapsed      = time.time() - t0
        on_track_pct = 100.0 * on_track_steps / max(step, 1)

        # Save checkpoints — "best" tracks raw env reward (what we ultimately care about)
        agent.save(latest_path)
        if total_env_reward > best_reward:
            best_reward = total_env_reward
            agent.save(best_path)
            star = " ★ NEW BEST (env)"
        else:
            star = ""

        log_rows.append({
            "episode":             ep,
            "total_env_reward":    round(total_env_reward, 2),
            "total_shaped_reward": round(total_shaped_reward, 2),
            "on_track_steps":      on_track_steps,
            "on_track_pct":        round(on_track_pct, 1),
            "avg_loss":            round(avg_loss, 6),
            "epsilon":             round(agent.epsilon, 4),
            "steps":               step,
            "elapsed_s":           round(elapsed, 1),
        })

        print(
            f"  Ep {ep:4d}/{args.episodes} | "
            f"Env: {total_env_reward:7.1f} | "
            f"Shaped: {total_shaped_reward:7.1f} | "
            f"OnTrack: {on_track_pct:5.1f}% | "
            f"Loss: {avg_loss:.5f} | "
            f"Eps: {agent.epsilon:.3f} | "
            f"Steps: {step:4d} | "
            f"{elapsed:.1f}s{star}"
        )

        if ep % 10 == 0:
            pd.DataFrame(log_rows).to_csv(log_path, index=False)

    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"\n  Training done. Best env reward: {best_reward:.1f}")
    print(f"  Log saved to {log_path}")


if __name__ == "__main__":
    train(parse_args())
