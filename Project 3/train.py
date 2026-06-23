"""Self-play training for Warlords with a shared policy (parameter sharing).

Trains one of DQN / A2C / PPO on the vectorised, agent-indicated env so the
three algorithms can be compared under an identical budget. Episodic returns
are logged to results/<algo>_curve.csv for the reward curves.

Example:
  python train.py --algo ppo --steps 2000000
  python train.py --algo dqn --steps 2000000
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from warlords_env import make_vec_env

RESULTS = os.path.join(os.path.dirname(__file__), "results")
MODELS = os.path.join(os.path.dirname(__file__), "models")


class RewardCurveLogger(BaseCallback):
    """Track episodic returns per sub-env and write a smoothed reward curve.

    The SuperSuit vec env does not emit SB3 'episode' infos, so we accumulate
    rewards/dones directly from the rollout.
    """

    def __init__(self, algo, log_every=20000):
        super().__init__()
        self.algo = algo
        self.log_every = log_every
        self.cur = None
        self.returns = []
        self.rows = []
        self.next_log = log_every

    def _on_step(self):
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        if self.cur is None:
            self.cur = np.zeros(len(rewards), dtype=np.float64)
        self.cur += rewards
        for i, d in enumerate(dones):
            if d:
                self.returns.append(self.cur[i])
                self.cur[i] = 0.0
        if self.num_timesteps >= self.next_log:
            recent = self.returns[-200:]
            mean_ret = float(np.mean(recent)) if recent else 0.0
            self.rows.append((self.num_timesteps, mean_ret, len(self.returns)))
            if self.verbose:
                print(f"[{self.algo}] step {self.num_timesteps:>9} | "
                      f"mean_ep_return {mean_ret:+.3f} | episodes {len(self.returns)}")
            self.next_log += self.log_every
        return True

    def save(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timesteps", "mean_ep_return", "episodes"])
            w.writerows(self.rows)


def build_model(algo, env, device):
    # gamma=0.999 -> effective horizon ~1000 steps, so the terminal win/loss is
    # visible for credit assignment over the long Warlords episodes.
    common = dict(policy="MlpPolicy", env=env, verbose=0, device=device,
                  gamma=0.999, policy_kwargs=dict(net_arch=[128, 128]))
    if algo == "ppo":
        return PPO(n_steps=512, batch_size=1024, n_epochs=4, learning_rate=2.5e-4,
                   ent_coef=0.01, clip_range=0.2, gae_lambda=0.95, vf_coef=0.5, **common)
    if algo == "a2c":
        return A2C(n_steps=16, learning_rate=7e-4, ent_coef=0.01, gae_lambda=0.95, **common)
    if algo == "dqn":
        return DQN(learning_rate=1e-4, buffer_size=200_000, learning_starts=20_000,
                   batch_size=128, train_freq=4, target_update_interval=2000,
                   exploration_fraction=0.2, exploration_final_eps=0.05, **common)
    raise ValueError(algo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "a2c", "dqn"], required=True)
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--max-cycles", type=int, default=4000)
    p.add_argument("--alive-bonus", type=float, default=0.005)
    p.add_argument("--outlast-bonus", type=float, default=0.5)
    p.add_argument("--no-shaped", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="warlords", help="output name: models/<algo>_<tag>.zip")
    args = p.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_vec_env(n_envs=args.n_envs, num_cpus=args.cpus, max_cycles=args.max_cycles,
                       shaped=not args.no_shaped, alive_bonus=args.alive_bonus,
                       outlast_bonus=args.outlast_bonus)
    model = build_model(args.algo, env, args.device)
    cb = RewardCurveLogger(args.algo, log_every=max(args.steps // 100, 5000))
    cb.verbose = 1

    print(f"Training {args.algo.upper()} for {args.steps:,} steps "
          f"({args.n_envs} envs x 4 agents, shaped={not args.no_shaped})")
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=cb)
    dt = time.time() - t0

    suffix = "" if args.tag == "warlords" else f"_{args.tag}"
    model_path = os.path.join(MODELS, f"{args.algo}_{args.tag}.zip")
    curve_path = os.path.join(RESULTS, f"{args.algo}{suffix}_curve.csv")
    model.save(model_path)
    cb.save(curve_path)
    print(f"Done in {dt/60:.1f} min ({args.steps/dt:.0f} steps/s). "
          f"Saved {model_path} and {curve_path}")


if __name__ == "__main__":
    main()
