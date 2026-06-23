"""Scan all 128 RAM bytes over real games to find the true ball coordinate(s):
the ball moves almost every frame in small smooth steps over a wide range."""
import numpy as np
from pettingzoo.atari import warlords_v3

PADDLE_BYTES = [91, 92, 93, 94]


def scan(seed=0, nsteps=6000):
    rng = np.random.RandomState(seed)
    e = warlords_v3.parallel_env(obs_type="ram")
    o, _ = e.reset(seed=seed)
    log = []
    while e.agents and len(log) < nsteps:
        acts = {a: int(rng.randint(6)) for a in e.agents}
        o, *_ = e.step(acts)
        log.append(next(iter(o.values())).copy())
    e.close()
    arr = np.asarray(log, dtype=np.int16)  # (T,128)
    T = len(arr)
    d = np.abs(np.diff(arr, axis=0))
    frac_change = (d > 0).mean(axis=0)
    mean_delta = d.mean(axis=0)
    # smoothness: among changing frames, how small is the step (ball=small, counter=±1 or jumpy)
    rng_span = arr.max(axis=0) - arr.min(axis=0)
    ndist = np.array([len(np.unique(arr[:, b])) for b in range(128)])
    print(f"T={T}")
    print("byte  frac_chg  mean|d|  span  ndist   (sorted by frac_change*span)")
    score = frac_change * rng_span
    order = np.argsort(-score)
    for b in order[:18]:
        tag = " <-PADDLE" if b in PADDLE_BYTES else ""
        print(f"{b:4d}  {frac_change[b]:.3f}   {mean_delta[b]:5.2f}   {rng_span[b]:4d}  {ndist[b]:4d}{tag}")
    return arr


def trajectory(seed=0, nsteps=120, bytes_=(90, 95, 99, 101, 109, 125)):
    rng = np.random.RandomState(seed)
    e = warlords_v3.parallel_env(obs_type="ram")
    o, _ = e.reset(seed=seed)
    rows = []
    while e.agents and len(rows) < nsteps:
        acts = {a: int(rng.randint(6)) for a in e.agents}
        o, *_ = e.step(acts)
        ram = next(iter(o.values()))
        rows.append([int(ram[b]) for b in bytes_])
    e.close()
    print("\ntrajectory (bytes %s):" % (bytes_,))
    for r in rows[::3]:
        print(r)


if __name__ == "__main__":
    a = scan(0)
    scan(5)
