"""Final faithful benchmark + self-detection reliability for the tournament agent.
Compares win rate (vs 3 random and vs 3 rule opponents, all seats) for every
candidate, and measures how reliably WarlordAgent self-detects its corner."""
import os
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import sys
import json
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool

import numpy as np
from pettingzoo.atari import warlords_v3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))
from baselines import RandomAgent, RuleBasedAgent

HERE = os.path.dirname(__file__)


def _load_params():
    p = os.path.join(HERE, "best_params.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        return {int(k): tuple(v["params"]) for k, v in d.items()}
    return None


def _mk_opp(kind, seat, seed, i):
    if kind == "rule":
        return RuleBasedAgent(player_index=i)
    return RandomAgent(seed=(seed * 13 + i * 71 + 1) & 0x7fffffff)


def _make_test(kind, seat, seed):
    if kind == "random":
        return RandomAgent(seed=(seed * 5 + 999) & 0x7fffffff)
    if kind == "real_rule":
        from rule_agent import RuleAgent
        return RuleAgent(player_index=seat)
    if kind == "real_ppo":
        from submission_agent import WarlordsAgent
        return WarlordsAgent(player_index=seat)
    if kind == "warlord_fixed":
        from warlord_agent import WarlordAgent
        return WarlordAgent(player_index=seat)
    if kind == "warlord_detect":
        from warlord_agent import WarlordAgent
        return WarlordAgent()  # zero-arg: self-detects
    raise ValueError(kind)


def _job(args):
    test_kind, opp_kind, seed, seat = args
    env = warlords_v3.env(obs_type="ram")
    env.reset(seed=seed)
    names = list(env.agents)
    agents = [None] * 4
    for i in range(4):
        agents[i] = _make_test(test_kind, seat, seed) if i == seat else _mk_opp(opp_kind, seat, seed, i)
    mp = {names[i]: agents[i] for i in range(4)}
    reward = {n: 0.0 for n in names}
    me = names[seat]
    for ag in env.agent_iter():
        obs, r, term, trunc, info = env.last()
        reward[ag] += r
        action = None if (term or trunc) else mp[ag].act(obs)
        env.step(action)
    env.close()
    won = reward[me] > 0
    timeout = max(reward.values()) <= 0
    # detected corner (if the agent self-detects)
    det = getattr(mp[me], "k", None)
    det_ok = (det == seat) if det is not None else None
    return int(won), int(timeout), det_ok


def bench(test_kind, opp_kind, games, base_seed=1000, procs=None):
    procs = procs or max(1, (os.cpu_count() or 2) - 1)
    jobs = [(test_kind, opp_kind, base_seed + g, seat) for seat in range(4) for g in range(games)]
    with Pool(procs) as pool:
        res = pool.map(_job, jobs)
    wins = np.mean([r[0] for r in res])
    tos = np.mean([r[1] for r in res])
    dets = [r[2] for r in res if r[2] is not None]
    det_acc = float(np.mean(dets)) if dets else None
    return wins, tos, det_acc


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    params = _load_params()
    print("loaded params:", params)
    cands = ["random", "real_rule", "real_ppo", "warlord_fixed", "warlord_detect"]
    print(f"\ngames/seat={games}  (faithful max_cycles=100000)")
    print(f"{'agent':16s} | vs random win/to/det | vs rule win/to")
    for c in cands:
        wr, tr, da = bench(c, "random", games)
        wru, tru, _ = bench(c, "rule", games)
        ds = f"{da:.2f}" if da is not None else "  - "
        print(f"{c:16s} |  {wr:.3f} {tr:.3f} {ds}  |  {wru:.3f} {tru:.3f}")
