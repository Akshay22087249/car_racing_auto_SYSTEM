"""Rigorous measurement under FAITHFUL tournament conditions (max_cycles=100000).
For an agent factory, play across all 4 seats vs 3 opponents and report:
  win rate, loss rate, timeout(no-winner) rate, mean game length,
  and mean OUTLAST score = #opponents that die before our seat (0..3; ==3 iff win).
Outlast is a dense, win-aligned objective for optimization.
"""
import os
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool

import numpy as np
from pettingzoo.atari import warlords_v3

PLAYER_ORDER = ["first_0", "second_0", "third_0", "fourth_0"]


def play_record(make_agents, seed, seat):
    """Returns dict: won, lost, timeout, length, outlast, my_death."""
    env = warlords_v3.env(obs_type="ram")
    env.reset(seed=seed)
    names = list(env.agents)
    agents = make_agents(seat, seed)
    mp = {names[i]: agents[i] for i in range(4)}
    reward = {n: 0.0 for n in names}
    death_step = {n: None for n in names}
    me = names[seat]
    step = 0
    alive = set(names)
    for ag in env.agent_iter():
        obs, r, term, trunc, info = env.last()
        reward[ag] += r
        if (term or trunc) and ag in alive:
            alive.discard(ag)
            death_step[ag] = step
            action = None
        elif term or trunc:
            action = None
        else:
            action = mp[ag].act(obs)
        env.step(action)
        step += 1
    env.close()
    won = reward[me] > 0
    # someone won iff max reward > 0
    someone_won = max(reward.values()) > 0
    timeout = not someone_won
    my_d = death_step[me]
    # opponents that died strictly before us (or before timeout if we survived)
    ref = my_d if my_d is not None else step + 1
    outlast = sum(1 for n in names if n != me and death_step[n] is not None and death_step[n] < ref)
    return dict(won=bool(won), lost=bool(not won and not timeout),
                timeout=bool(timeout and not won), length=step, outlast=outlast)


def _job(args):
    make_agents, seed, seat = args
    return play_record(make_agents, seed, seat)


def measure(make_agents, games=40, seats=(0, 1, 2, 3), base_seed=1000, procs=None):
    procs = procs or max(1, (os.cpu_count() or 2) - 1)
    jobs = [(make_agents, base_seed + g, seat) for seat in seats for g in range(games)]
    with Pool(procs) as pool:
        res = pool.map(_job, jobs)
    n = len(res)
    return dict(
        win=np.mean([r["won"] for r in res]),
        loss=np.mean([r["lost"] for r in res]),
        timeout=np.mean([r["timeout"] for r in res]),
        outlast=np.mean([r["outlast"] for r in res]),
        length=np.mean([r["length"] for r in res]),
        n=n,
    )


# ---- agent factories (top-level for spawn) ----
from functools import partial
import sys as _sys
from baselines import RandomAgent, RuleBasedAgent
from controllers import LawAgent, LinearInterceptor, NoopAgent


def _rand_opps(agents, seat, seed):
    for i in range(4):
        if i != seat:
            agents[i] = RandomAgent(seed=(seed * 13 + i * 71 + 1) & 0x7fffffff)
    return agents


def mk_random(seat, seed):
    a = [None] * 4
    a[seat] = RandomAgent(seed=(seed * 5 + 999) & 0x7fffffff)
    return _rand_opps(a, seat, seed)


def mk_noop(seat, seed):
    a = [None] * 4
    a[seat] = NoopAgent()
    return _rand_opps(a, seat, seed)


def mk_rule_baseline(seat, seed):
    a = [None] * 4
    a[seat] = RuleBasedAgent(player_index=seat)
    return _rand_opps(a, seat, seed)


def mk_law_generic(seat, seed, law="sweep"):
    a = [None] * 4
    a[seat] = LawAgent(seat, law=law)
    return _rand_opps(a, seat, seed)


def mk_real_rule(seat, seed):
    """The actual agents/rule_agent.py the user ships."""
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))
    from rule_agent import RuleAgent
    a = [None] * 4
    a[seat] = RuleAgent(player_index=seat)
    return _rand_opps(a, seat, seed)


def mk_real_ppo(seat, seed):
    """The actual agents/submission_agent.py (exported PPO)."""
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))
    from submission_agent import WarlordsAgent
    a = [None] * 4
    a[seat] = WarlordsAgent(player_index=seat)
    return _rand_opps(a, seat, seed)


if __name__ == "__main__":
    games = int(_sys.argv[1]) if len(_sys.argv) > 1 else 20
    print(f"games/seat={games}, max_cycles=100000 (faithful)\n")
    cands = [
        ("random(sanity)", mk_random),
        ("noop(frozen)", mk_noop),
        ("sweep", partial(mk_law_generic, law="sweep")),
        ("baseline_rule", mk_rule_baseline),
        ("real_rule_agent", mk_real_rule),
        ("real_ppo", mk_real_ppo),
        ("track_xinvy", partial(mk_law_generic, law="track_xinvy")),
        ("track_xinvyinv", partial(mk_law_generic, law="track_xinvyinv")),
        ("static", partial(mk_law_generic, law="static")),
    ]
    print(f"{'agent':18s} | win   loss  tmout | outlast | len")
    for name, mk in cands:
        m = measure(mk, games=games)
        print(f"{name:18s} | {m['win']:.3f} {m['loss']:.3f} {m['timeout']:.3f} | "
              f"{m['outlast']:.2f}    | {m['length']:.0f}")
