"""Evaluate and compare agents in the real Warlords environment.

Runs head-to-head matches (the same AEC loop the tournament notebook uses) and
reports, per agent, win rate / mean reward / mean survival against the random
and rule-based baselines. Also draws the training reward curves and a summary
bar chart. This is where the "try every algorithm, then pick the best" choice
is made empirically rather than argued.

  python evaluate.py --games 40            # compare all trained algos + baselines
  python evaluate.py --round-robin --games 40
  python evaluate.py --video ppo           # record one game to results/videos
"""
import argparse
import csv
import os

import numpy as np
from pettingzoo.atari import warlords_v3

from collections import deque

from baselines import RandomAgent, RuleBasedAgent, _GameTracker
from ram_map import extract_features, PADDLE_BYTES
from warlords_solo import STACK_K, build_obs

RESULTS = os.path.join(os.path.dirname(__file__), "results")
MODELS = os.path.join(os.path.dirname(__file__), "models")
PLAYER_ORDER = ["first_0", "second_0", "third_0", "fourth_0"]


class LearnedAgent:
    """Runs a trained SB3 self-play policy with the same STACK_K-frame stacking
    used in training. Knows its corner via fixed_index (evaluation) or self-
    detects it from the RAM (tournament-realistic)."""

    def __init__(self, model, fixed_index=None, deterministic=True):
        self.model = model
        self.fixed_index = fixed_index
        self.deterministic = deterministic
        self.tr = _GameTracker()
        self._last = None
        self.frames = deque(maxlen=STACK_K)

    def _index(self, obs):
        if self.fixed_index is not None:
            return self.fixed_index
        self.tr.update(obs, self._last)
        if self.tr.paddle_byte is not None:
            return PADDLE_BYTES.index(self.tr.paddle_byte)
        return 0  # default until detected

    def act(self, observation):
        obs = np.asarray(observation, dtype=np.uint8)
        idx = self._index(obs)
        feat = extract_features(obs, idx)
        if not self.frames:
            for _ in range(STACK_K):
                self.frames.append(feat)
        else:
            self.frames.append(feat)
        vec = build_obs(self.frames, idx)
        action, _ = self.model.predict(vec, deterministic=self.deterministic)
        self._last = int(action)
        return self._last


def load_model(name):
    from stable_baselines3 import A2C, DQN, PPO
    algo = name.split("_")[0]
    cls = {"ppo": PPO, "a2c": A2C, "dqn": DQN}[algo]
    return cls.load(os.path.join(MODELS, f"{name}.zip"), device="cpu")


def make_agent(name, models, fixed_index=None):
    if name == "random":
        return RandomAgent(seed=np.random.randint(1 << 30))
    if name == "rule":
        return RuleBasedAgent()
    return LearnedAgent(models[name], fixed_index=fixed_index)


def play_match(agents, seed, max_cycles=3000, video_path=None):
    """One 4-player game. agents[i] controls PLAYER_ORDER[i]. Returns per-slot
    reward / survival / win and the game length."""
    render = video_path is not None
    env = warlords_v3.env(obs_type="ram", max_cycles=max_cycles,
                          render_mode="rgb_array" if render else None)
    env.reset(seed=seed)
    names = list(env.agents)
    mapping = {names[i]: agents[i] for i in range(len(names))}
    reward = {n: 0.0 for n in names}
    survived = {n: 0 for n in names}
    frames = []
    step = 0
    for ag in env.agent_iter():
        obs, r, term, trunc, info = env.last()
        reward[ag] += r
        if not (term or trunc):
            survived[ag] += 1
            action = mapping[ag].act(obs)
        else:
            action = None
        env.step(action)
        if render:
            frames.append(env.render())
        step += 1
    env.close()
    if render and frames:
        import imageio
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, frames, fps=30)
    winner = max(reward, key=reward.get)
    return {names[i]: dict(reward=reward[names[i]], survived=survived[names[i]],
                           won=(names[i] == winner and reward[names[i]] > 0))
            for i in range(len(names))}, step


def eval_vs(test_name, opp_name, models, games, max_cycles, base_seed=1000):
    """Test agent vs three copies of an opponent, rotating the test slot."""
    wins = rewards = survived = 0.0
    for g in range(games):
        slot = g % 4
        agents = [make_agent(opp_name, models) for _ in range(4)]
        agents[slot] = make_agent(test_name, models, fixed_index=slot)
        res, _ = play_match(agents, seed=base_seed + g, max_cycles=max_cycles)
        me = res[PLAYER_ORDER[slot]]
        wins += me["won"]; rewards += me["reward"]; survived += me["survived"]
    return dict(win_rate=wins / games, mean_reward=rewards / games,
                mean_survival=survived / games)


def round_robin(names, models, games, max_cycles, base_seed=2000):
    """Seat four of the agents together each game (rotating which four and their
    corners) and tally wins / survival. This is the tournament-realistic metric:
    distinct agents competing head to head."""
    n = len(names)
    wins = {x: 0 for x in names}
    survived = {x: 0.0 for x in names}
    plays = {x: 0 for x in names}
    for g in range(games):
        chosen = [names[(g + j) % n] for j in range(4)]
        agents = [make_agent(chosen[i], models, fixed_index=i) for i in range(4)]
        res, _ = play_match(agents, seed=base_seed + g, max_cycles=max_cycles)
        for i, x in enumerate(chosen):
            plays[x] += 1
            survived[x] += res[PLAYER_ORDER[i]]["survived"]
            wins[x] += res[PLAYER_ORDER[i]]["won"]
    return {x: dict(win_rate=wins[x] / max(plays[x], 1), wins=wins[x], plays=plays[x],
                    mean_survival=survived[x] / max(plays[x], 1)) for x in names}


def discover_models():
    import glob
    names = [os.path.basename(p)[:-4] for p in glob.glob(os.path.join(MODELS, "*_solo.zip"))]
    return {n: load_model(n) for n in sorted(names)}


def plot_curves():
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    found = False
    for algo, color in [("ppo", "tab:blue"), ("a2c", "tab:green"), ("dqn", "tab:red")]:
        path = os.path.join(RESULTS, f"{algo}_solo_curve.csv")
        if not os.path.exists(path):
            continue
        found = True
        xs, ys = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                xs.append(int(row["timesteps"])); ys.append(float(row["win_rate"]))
        plt.plot(xs, ys, label=algo.upper(), color=color, linewidth=2)
    if found:
        plt.axhline(0.25, ls="--", c="gray", label="fair share (0.25)")
        plt.xlabel("environment steps"); plt.ylabel("win rate vs opponent pool")
        plt.title("Self-play training curves on Warlords"); plt.legend(); plt.grid(alpha=0.3)
        out = os.path.join(RESULTS, "training_curves.png")
        plt.tight_layout(); plt.savefig(out, dpi=130); print("saved", out)


def plot_summary(table):
    import matplotlib.pyplot as plt
    names = list(table.keys())
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(x, [table[n]["vs_random"]["win_rate"] for n in names], 0.38, label="vs random")
    ax1.bar(x + 0.4, [table[n]["vs_rule"]["win_rate"] for n in names], 0.38, label="vs rule-based")
    ax1.axhline(0.25, ls="--", c="gray")
    ax1.set_xticks(x + 0.2); ax1.set_xticklabels(names, rotation=20); ax1.set_ylabel("win rate")
    ax1.set_title("Win rate against 3 baseline opponents"); ax1.legend(); ax1.grid(alpha=0.3, axis="y")
    ax2.bar(x, [table[n]["vs_random"]["mean_survival"] for n in names], color="tab:purple")
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=20); ax2.set_ylabel("mean survival (decisions)")
    ax2.set_title("Survival vs random opponents"); ax2.grid(alpha=0.3, axis="y")
    out = os.path.join(RESULTS, "agent_comparison.png")
    plt.tight_layout(); plt.savefig(out, dpi=130); print("saved", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--max-cycles", type=int, default=30000)  # high enough to resolve naturally
    p.add_argument("--round-robin", action="store_true")
    p.add_argument("--video", default=None, help="record one game for this agent name")
    args = p.parse_args()

    models = discover_models()
    learned = sorted(models.keys())

    if args.video:
        agents = [make_agent("rule", models) for _ in range(4)]
        agents[0] = make_agent(args.video, models, fixed_index=0)
        path = os.path.join(RESULTS, "videos", f"{args.video}_vs_rule.mp4")
        _, length = play_match(agents, seed=7, max_cycles=args.max_cycles, video_path=path)
        print(f"saved {path} ({length} frames)")
        return

    contenders = ["random", "rule"] + learned
    table = {}
    for name in contenders:
        vr = eval_vs(name, "random", models, args.games, args.max_cycles)
        ru = eval_vs(name, "rule", models, args.games, args.max_cycles)
        table[name] = {"vs_random": vr, "vs_rule": ru}
        print(f"{name:12s} | vs random: win {vr['win_rate']:.3f} surv {vr['mean_survival']:6.0f} "
              f"| vs rule: win {ru['win_rate']:.3f} surv {ru['mean_survival']:6.0f}")

    with open(os.path.join(RESULTS, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "vs_random_winrate", "vs_random_survival", "vs_rule_winrate", "vs_rule_survival"])
        for n in contenders:
            t = table[n]
            w.writerow([n, t["vs_random"]["win_rate"], t["vs_random"]["mean_survival"],
                        t["vs_rule"]["win_rate"], t["vs_rule"]["mean_survival"]])

    rr = round_robin(contenders, models, max(args.games, len(contenders) * 6), args.max_cycles)
    print("round-robin (four distinct agents per game):")
    for n in sorted(contenders, key=lambda x: -rr[x]["win_rate"]):
        print(f"  {n:12s} win_rate {rr[n]['win_rate']:.3f} ({rr[n]['wins']}/{rr[n]['plays']}) "
              f"surv {rr[n]['mean_survival']:.0f}")
    with open(os.path.join(RESULTS, "round_robin.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["agent", "win_rate", "wins", "plays", "mean_survival"])
        for n in contenders:
            w.writerow([n, rr[n]["win_rate"], rr[n]["wins"], rr[n]["plays"], rr[n]["mean_survival"]])

    plot_curves()
    plot_summary(table)


if __name__ == "__main__":
    main()
