"""Calibrate geometry: paddle range & speed per corner, and identify ball bytes."""
import numpy as np
from pettingzoo.atari import warlords_v3

NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
PADDLE_BYTES = [91, 92, 93, 94]
BALL_CANDS = [90, 95, 111]


def paddle_range_and_speed():
    names = ["first_0", "second_0", "third_0", "fourth_0"]
    for idx in range(4):
        e = warlords_v3.parallel_env(obs_type="ram")
        o, _ = e.reset(seed=7)
        who = names[idx]
        pb = PADDLE_BYTES[idx]
        # drive UP a long time, record min
        vals = []
        for _ in range(120):
            acts = {a: NOOP for a in e.agents}
            if who in acts: acts[who] = UP
            o, *_ = e.step(acts); vals.append(int(o[who][pb]))
        up_min = min(vals); up_settle = vals[-1]
        # measure per-step speed from a mid position: go DOWN, record deltas
        vals2 = []
        for _ in range(120):
            acts = {a: NOOP for a in e.agents}
            if who in acts: acts[who] = DOWN
            o, *_ = e.step(acts); vals2.append(int(o[who][pb]))
        dn_max = max(vals2)
        # speed: median abs diff while moving (exclude saturated)
        d2 = np.abs(np.diff(vals2))
        speed = int(np.median(d2[d2 > 0])) if (d2 > 0).any() else 0
        print(f"corner {idx} ({who}): paddle byte {pb}  range~[{up_min},{dn_max}]  up_settle={up_settle}  step_speed~{speed}")
        e.close()


def ball_bytes_over_game(seed=0, nsteps=8000):
    """Random game; log ball candidate bytes + all paddle bytes; report ranges and
    which candidate bytes move smoothly (small per-step deltas => real position)."""
    rng = np.random.RandomState(seed)
    e = warlords_v3.parallel_env(obs_type="ram")
    o, _ = e.reset(seed=seed)
    logs = {b: [] for b in BALL_CANDS}
    pad = {b: [] for b in PADDLE_BYTES}
    for t in range(nsteps):
        if not e.agents:
            break
        acts = {a: int(rng.randint(6)) for a in e.agents}
        o, r, term, trunc, info = e.step(acts)
        any_obs = next(iter(o.values()))
        for b in BALL_CANDS:
            logs[b].append(int(any_obs[b]))
        for b in PADDLE_BYTES:
            pad[b].append(int(any_obs[b]))
    e.close()
    print(f"\nball candidate bytes over {len(logs[BALL_CANDS[0]])} steps (seed {seed}):")
    for b in BALL_CANDS:
        arr = np.asarray(logs[b])
        d = np.abs(np.diff(arr))
        print(f"  byte {b}: range[{arr.min()},{arr.max()}] n_distinct={len(np.unique(arr))} "
              f"mean|delta|={d.mean():.1f} frac_zero_delta={(d==0).mean():.2f} sample={arr[:24].tolist()}")
    print("paddle byte ranges (random play):")
    for b in PADDLE_BYTES:
        arr = np.asarray(pad[b]); print(f"  byte {b}: range[{arr.min()},{arr.max()}]")


if __name__ == "__main__":
    print("=== paddle range & speed ===")
    paddle_range_and_speed()
    print("\n=== ball bytes ===")
    ball_bytes_over_game(seed=0)
    ball_bytes_over_game(seed=3)
