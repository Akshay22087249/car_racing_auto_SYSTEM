"""Ground-truth probe for the REAL tournament conditions.

The tournament notebook does: warlords_v3.env(obs_type="ram", render_mode=...)
with NO max_cycles and agents instantiated with NO args. So we must learn:
  1. the default max_cycles,
  2. how a game naturally resolves with random agents (length, winner?),
  3. whether FIRE differs from NOOP at the byte level,
  4. which action moves which paddle, per corner.
"""
import numpy as np
from pettingzoo.atari import warlords_v3

NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
PADDLE_BYTES = [91, 92, 93, 94]


def q1_default_maxcycles():
    env = warlords_v3.env(obs_type="ram")
    # walk the wrapper chain looking for max_cycles
    found = {}
    e = env
    seen = 0
    while e is not None and seen < 20:
        seen += 1
        for attr in ("max_cycles", "_max_cycles"):
            if hasattr(e, attr):
                found[type(e).__name__ + "." + attr] = getattr(e, attr)
        e = getattr(e, "env", None) or getattr(e, "unwrapped", None) if not isinstance(getattr(e, "unwrapped", None), type(e)) else None
    print("max_cycles attrs found:", found)
    env.close()


def q2_natural_resolution(n_games=3, seeds=(0, 1, 2)):
    """Play full random games, no max_cycles override, see how they end."""
    rng = np.random.RandomState(0)
    for g in range(n_games):
        env = warlords_v3.env(obs_type="ram")
        env.reset(seed=int(seeds[g]))
        rewards = {a: 0.0 for a in env.agents}
        steps = 0
        alive_history = []
        for agent in env.agent_iter():
            obs, r, term, trunc, info = env.last()
            rewards[agent] += r
            if term or trunc:
                action = None
            else:
                action = int(rng.randint(6))
            env.step(action)
            steps += 1
            if steps % 2000 == 0:
                alive_history.append((steps, list(env.agents)))
            if steps > 60000:
                break
        env.close()
        winners = [a for a, v in rewards.items() if v > 0]
        print(f"game {g}: steps={steps} rewards={ {k: round(v,2) for k,v in rewards.items()} } winners={winners}")
        print(f"   alive checkpoints: {alive_history}")


def q3_fire_vs_noop():
    """Run two identical parallel envs; one player does FIRE, the other NOOP.
    Diff RAM over many steps to see if FIRE ever has an effect."""
    diffs = set()
    for testact, label in [(FIRE, "FIRE"), (UP, "UP")]:
        e1 = warlords_v3.parallel_env(obs_type="ram")
        e2 = warlords_v3.parallel_env(obs_type="ram")
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        agents = list(e1.agents)
        target = agents[0]
        total_diff = np.zeros(128, dtype=np.int64)
        for t in range(400):
            a1 = {a: NOOP for a in e1.agents}
            a2 = {a: NOOP for a in e2.agents}
            if target in a1:
                a1[target] = testact
            o1, *_ = e1.step(a1)
            o2, *_ = e2.step(a2)
            if target in o1 and target in o2:
                d = (o1[target].astype(np.int64) != o2[target].astype(np.int64))
                total_diff += d
        changed = np.where(total_diff > 0)[0]
        print(f"{label} vs NOOP: bytes that ever differ over 400 steps: {list(changed)}  (counts {total_diff[changed].tolist()})")
        e1.close(); e2.close()


def q4_action_to_paddle():
    """For each player, run UP then DOWN runs and see which paddle byte moves."""
    e = warlords_v3.parallel_env(obs_type="ram")
    obs, _ = e.reset(seed=7)
    agents = list(e.agents)
    print("agents:", agents)
    for idx, who in enumerate(agents):
        e2 = warlords_v3.parallel_env(obs_type="ram")
        o, _ = e2.reset(seed=7)
        before = o[who][PADDLE_BYTES].copy()
        for _ in range(30):
            acts = {a: NOOP for a in e2.agents}
            acts[who] = UP
            o, *_ = e2.step(acts)
        after_up = o[who][PADDLE_BYTES].copy()
        for _ in range(30):
            acts = {a: NOOP for a in e2.agents}
            acts[who] = DOWN
            o, *_ = e2.step(acts)
        after_dn = o[who][PADDLE_BYTES].copy()
        d_up = after_up.astype(int) - before.astype(int)
        d_dn = after_dn.astype(int) - after_up.astype(int)
        print(f"player {idx} ({who}): paddle bytes {PADDLE_BYTES}")
        print(f"   UP delta:   {d_up.tolist()}")
        print(f"   DOWN delta: {d_dn.tolist()}")
        e2.close()
    e.close()


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"):
        print("=== Q1: default max_cycles ==="); q1_default_maxcycles()
    if which in ("all", "3"):
        print("\n=== Q3: FIRE vs NOOP ==="); q3_fire_vs_noop()
    if which in ("all", "4"):
        print("\n=== Q4: action -> paddle ==="); q4_action_to_paddle()
    if which in ("all", "2"):
        print("\n=== Q2: natural resolution ==="); q2_natural_resolution()
