"""Faithful Warlords tournament harness (matches the tournament notebook exactly):
AEC env, obs_type='ram', NO max_cycles override (default 100000 -> games resolve
naturally), agents control fixed seats, last castle standing wins (+1 reward).

Provides multiprocessing evaluation: put the test agent in each of the 4 seats in
turn, fill the other 3 with opponents, and report win rate / mean survival.
"""
import os
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool

import numpy as np
from pettingzoo.atari import warlords_v3

PLAYER_ORDER = ["first_0", "second_0", "third_0", "fourth_0"]


def play_one(make_agents, seed):
    """make_agents: callable(seed)->list of 4 agent objects (index i controls seat i).
    Returns dict seat_index -> dict(won, survived, reward)."""
    env = warlords_v3.env(obs_type="ram")
    env.reset(seed=seed)
    names = list(env.agents)
    agents = make_agents(seed)
    mapping = {names[i]: agents[i] for i in range(len(names))}
    reward = {n: 0.0 for n in names}
    survived = {n: 0 for n in names}
    for ag in env.agent_iter():
        obs, r, term, trunc, info = env.last()
        reward[ag] += r
        if term or trunc:
            action = None
        else:
            survived[ag] += 1
            action = mapping[ag].act(obs)
        env.step(action)
    env.close()
    out = {}
    for i, n in enumerate(names):
        out[i] = dict(won=bool(reward[n] > 0), survived=survived[n], reward=reward[n])
    # survival = number of steps the seat was alive (reward 0 until it dies/wins)
    return out


def _worker(args):
    spec, seed, test_seat = args
    test_factory, opp_factory = spec
    def make(seed):
        ags = [opp_factory(i, seed) for i in range(4)]
        ags[test_seat] = test_factory(test_seat, seed)
        return ags
    res = play_one(make, seed)
    return test_seat, res[test_seat]["won"]


def evaluate(test_factory, opp_factory, games=40, seats=(0, 1, 2, 3),
             base_seed=1000, procs=None, per_seat=False):
    """test_factory(seat, seed)->agent ; opp_factory(seat, seed)->agent.
    Plays `games` per seat in `seats`. Returns overall win rate (+ per-seat if asked)."""
    procs = procs or max(1, (os.cpu_count() or 2) - 1)
    jobs = []
    for seat in seats:
        for g in range(games):
            jobs.append(((test_factory, opp_factory), base_seed + g, seat))
    with Pool(procs) as pool:
        results = pool.map(_worker, jobs)
    by_seat = {s: [] for s in seats}
    for seat, won in results:
        by_seat[seat].append(won)
    overall = np.mean([w for ws in by_seat.values() for w in ws])
    if per_seat:
        return overall, {s: float(np.mean(by_seat[s])) for s in seats}
    return overall
