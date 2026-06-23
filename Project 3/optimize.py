"""Tune a per-corner VelInterceptor directly on the REAL win-rate objective
(faithful env, max_cycles=100000, vs 3 random opponents). Timeouts are not wins,
so win rate automatically punishes turtling. Big pool over (candidate, game)."""
import os
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import sys
import json
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool

import numpy as np
from pettingzoo.atari import warlords_v3

from baselines import RandomAgent
from controllers import VelInterceptor

PARAM_NAMES = ["a", "bx", "by", "vx", "vy"]


def _play(seat, params, seed, max_cycles=100000):
    a, bx, by, vx, vy = params
    env = warlords_v3.env(obs_type="ram", max_cycles=max_cycles)
    env.reset(seed=seed)
    names = list(env.agents)
    agents = [None] * 4
    for i in range(4):
        if i == seat:
            agents[i] = VelInterceptor(i, a=a, bx=bx, by=by, vx=vx, vy=vy)
        else:
            agents[i] = RandomAgent(seed=(seed * 13 + i * 71 + 1) & 0x7fffffff)
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
    return int(won), int(timeout)


def _job(args):
    cid, seat, params, seed, max_cycles = args
    won, timeout = _play(seat, params, seed, max_cycles)
    return cid, won, timeout


def _eval_cands(seat, cands, games, base_seed, procs, max_cycles=100000):
    jobs = []
    for cid, params in enumerate(cands):
        for g in range(games):
            jobs.append((cid, seat, params, base_seed + g, max_cycles))
    with Pool(procs) as pool:
        results = pool.map(_job, jobs)
    agg = {cid: [0, 0] for cid in range(len(cands))}
    for cid, won, to in results:
        agg[cid][0] += won; agg[cid][1] += to
    return [(cands[cid], agg[cid][0] / games, agg[cid][1] / games) for cid in range(len(cands))]


def optimize_corner(seat, n_cand=60, games=24, refine_games=60, procs=None):
    procs = procs or max(1, (os.cpu_count() or 2) - 1)
    rng = np.random.RandomState(7000 + seat)
    # priors: the diagonal laws that worked, with zero velocity
    cands = [(0.5, -1.0, 1.0, 0.0, 0.0), (0.5, -1.0, -1.0, 0.0, 0.0),
             (0.5, 1.0, 1.0, 0.0, 0.0), (0.5, 1.0, -1.0, 0.0, 0.0),
             (1.0, -1.5, 1.5, 0.0, 0.0), (0.0, 1.5, 1.5, 0.0, 0.0)]
    while len(cands) < n_cand:
        cands.append((round(rng.uniform(-0.3, 1.3), 3), round(rng.uniform(-2.5, 2.5), 3),
                      round(rng.uniform(-2.5, 2.5), 3), round(rng.uniform(-3, 3), 3),
                      round(rng.uniform(-3, 3), 3)))
    # fast search at a 30k cap (good agents resolve in ~20k; turtles get penalised)
    scored = _eval_cands(seat, cands, games, 500, procs, max_cycles=30000)
    scored.sort(key=lambda r: (-r[1], r[2]))
    top = [p for p, w, t in scored[:8]]
    # local refinement around the best
    best0 = top[0]
    local = [best0]
    for _ in range(16):
        local.append(tuple(round(best0[i] + rng.uniform(-0.4, 0.4) * (2.5 if i else 0.5), 3)
                           for i in range(5)))
    # final refine at the FAITHFUL 100k cap so timeouts are scored exactly as in the tournament
    ref = _eval_cands(seat, top + local, refine_games, 3000, procs, max_cycles=100000)
    ref.sort(key=lambda r: (-r[1], r[2]))
    return ref


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    ncand = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    refine_games = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    best = {}
    for seat in range(4):
        ref = optimize_corner(seat, n_cand=ncand, games=games, refine_games=refine_games)
        print(f"\n=== corner {seat}: refined top (params | win | timeout) ===", flush=True)
        for params, w, t in ref[:6]:
            print(f"  {params}  win={w:.3f}  to={t:.2f}", flush=True)
        best[seat] = {"params": list(ref[0][0]), "win": ref[0][1], "timeout": ref[0][2]}
    print("\nBEST per corner:")
    for s in range(4):
        print(f"  {s}: {best[s]}")
    with open(os.path.join(os.path.dirname(__file__), "best_params.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("\nsaved best_params.json")
