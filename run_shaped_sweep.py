"""Run the full reward-shaping sweep for the DQN agent.

Trains five variants sequentially:
    50/50, 40/60, 30/70, 20/80, 10/90   (env% / on-track%)

Each variant writes to results/dqn/shaped/{tag}/. If a variant's dqn_best.pt
already exists, it is skipped — so the sweep is resumable after interruption.

Usage:
    python run_shaped_sweep.py                 # all five, default 500 episodes
    python run_shaped_sweep.py --episodes 200  # shorter runs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import config

VARIANTS = [
    (0.5, "w50"),
    (0.6, "w60"),
    (0.7, "w70"),
    (0.8, "w80"),
    (0.9, "w90"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DQN reward-shaping sweep.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed",     type=int, default=config.BASE_SEED)
    parser.add_argument("--force",    action="store_true",
                        help="Re-run variants even if dqn_best.pt already exists.")
    args = parser.parse_args()

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_dqn_shaped.py")

    for w, tag in VARIANTS:
        best_path = os.path.join(config.RESULTS_DIR, "dqn", "shaped", tag, "checkpoints", "dqn_best.pt")
        if os.path.exists(best_path) and not args.force:
            print(f"\n[skip] {tag}: {best_path} already exists. Use --force to re-run.\n")
            continue

        print(f"\n{'='*60}\n  Training variant {tag} (track_weight={w})\n{'='*60}")
        cmd = [
            sys.executable, script,
            "--track-weight", str(w),
            "--output-tag",   tag,
            "--episodes",     str(args.episodes),
            "--seed",         str(args.seed),
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[abort] {tag} exited with code {result.returncode}. Stopping sweep.")
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
