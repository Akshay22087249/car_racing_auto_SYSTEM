"""Opponent-diversity training: one learner controlling a random corner against
a pool of scripted opponents (see warlords_solo.py). This is the paradigm that
actually produces a strong agent, and is used for the tournament submission.

  python train_solo.py --algo ppo --steps 4000000 --opponents random rule
  # self-play refinement on top of a snapshot:
  python train_solo.py --algo ppo --steps 4000000 --opponents random rule self \
      --frozen agents/policy_weights.npz --tag solo2
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from train import build_model, RESULTS, MODELS
from export_policy import export_actor
from warlords_solo import WarlordsSoloEnv


class SnapshotCallback(BaseCallback):
    """Periodically write the live actor to the frozen path so the 'self'
    opponents track the improving policy (rolling self-play)."""

    def __init__(self, path, every):
        super().__init__()
        self.path = path
        self.every = every
        self.next_at = every

    def _on_step(self):
        if self.path and self.num_timesteps >= self.next_at:
            export_actor(self.model, self.path)
            self.next_at += self.every
        return True


class WinRateLogger(BaseCallback):
    """Track win rate and shaped return from completed episodes."""

    def __init__(self, algo, log_every):
        super().__init__()
        self.algo = algo
        self.log_every = log_every
        self.wins, self.rets = [], []
        self.cur = None
        self.rows = []
        self.next_log = log_every

    def _on_step(self):
        if self.cur is None:
            self.cur = np.zeros(len(self.locals["rewards"]))
        self.cur += self.locals["rewards"]
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                if "won" in info:
                    self.wins.append(1.0 if info["won"] else 0.0)
                self.rets.append(self.cur[i]); self.cur[i] = 0.0
        if self.num_timesteps >= self.next_log:
            wr = float(np.mean(self.wins[-300:])) if self.wins else 0.0
            ret = float(np.mean(self.rets[-300:])) if self.rets else 0.0
            self.rows.append((self.num_timesteps, wr, ret))
            print(f"[{self.algo}-solo] step {self.num_timesteps:>9} | win_rate {wr:.3f} | "
                  f"return {ret:+.2f} | episodes {len(self.rets)}")
            self.next_log += self.log_every
        return True

    def save(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timesteps", "win_rate", "mean_return"])
            w.writerows(self.rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "a2c", "dqn"], default="ppo")
    p.add_argument("--steps", type=int, default=4_000_000)
    p.add_argument("--n-envs", type=int, default=12)
    p.add_argument("--max-cycles", type=int, default=8000)
    p.add_argument("--opponents", nargs="+", default=["random", "rule"])
    p.add_argument("--frozen", default=None, help="npz snapshot for the 'self' opponent")
    p.add_argument("--snapshot-every", type=int, default=0, help="refresh --frozen every N steps (rolling self-play)")
    p.add_argument("--load", default=None, help="continue training from this model zip")
    p.add_argument("--tag", default="solo")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(RESULTS, exist_ok=True); os.makedirs(MODELS, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    env = make_vec_env(
        WarlordsSoloEnv, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(opponent_pool=tuple(args.opponents), max_cycles=args.max_cycles,
                        frozen_path=args.frozen),
    )
    if args.load:
        from stable_baselines3 import A2C, DQN, PPO
        cls = {"ppo": PPO, "a2c": A2C, "dqn": DQN}[args.algo]
        model = cls.load(args.load, env=env, device="cpu")
    else:
        model = build_model(args.algo, env, device="cpu")
    logger = WinRateLogger(args.algo, log_every=max(args.steps // 100, 5000))
    cb = [logger]
    if args.snapshot_every and args.frozen:
        cb.append(SnapshotCallback(args.frozen, args.snapshot_every))

    print(f"Training {args.algo.upper()} (solo, opponents={args.opponents}) "
          f"for {args.steps:,} steps on {args.n_envs} envs")
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=cb)
    dt = time.time() - t0

    model_path = os.path.join(MODELS, f"{args.algo}_{args.tag}.zip")
    curve_path = os.path.join(RESULTS, f"{args.algo}_{args.tag}_curve.csv")
    model.save(model_path); logger.save(curve_path)
    print(f"Done in {dt/60:.1f} min ({args.steps/dt:.0f} steps/s). Saved {model_path}")


if __name__ == "__main__":
    main()
