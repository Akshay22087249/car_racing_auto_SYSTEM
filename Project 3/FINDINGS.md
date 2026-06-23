# Warlords: game-understanding findings (working notes)

Last updated: 2026-06-21. Status: investigation in progress. Several prior project
conclusions (in `report/report.md`, `README.md`, old memory) are now known to be
**wrong** because they were based on a flawed evaluation and wrong RAM bytes.
This file is the corrected source of truth to resume from.

## TL;DR
1. The real tournament has **no `max_cycles`** (default = 100000) and instantiates
   agents with **no arguments** (so the agent must self-detect its corner).
2. The real ball coordinates are **byte 4 (X)** and **byte 14 (Y)**. The old agents
   tracked **byte 90 (a frame counter)** and **byte 95 (serve-only)** = useless.
3. Winning = last castle standing within 100000 cycles. A *perfect* defender
   traps/loses the ball, the game never resolves, and **nobody wins** (timeout).
   So the best agent is an **active, slightly leaky** interceptor, not a turtle.
4. The previously claimed "rule agent ~0.38, fair-share ceiling" is an **artifact**
   of an eval that truncated at `max_cycles=30000`. Under faithful conditions the
   old agents are actually **at or below random**.

## How the tournament actually scores (verified from the notebook)
- `warlords_v3.env(obs_type="ram", render_mode="rgb_array")` — **no max_cycles** →
  default **100000 cycles**.
- Agents created as `Agent1()` … `Agent4()` with **no args**. Fixed seats:
  Agent1→first_0, Agent2→second_0, Agent3→third_0, Agent4→fourth_0.
- 10 games. Each step, `scores[agent]+=reward`; if `reward>0` it counts as a win.
  Last castle standing gets +1. **Most wins across the 10 games = tournament winner.**
- Games resolve naturally to one winner in ~5000 cycles (~19k agent-steps) with
  all-random play — UNLESS the ball gets stuck (then it's a timeout, no winner).

## RAM map — CORRECTED (verified by probing the live env)
| What | Byte(s) | Notes |
|---|---|---|
| Paddle of player i | **91 + i** | UP(2) decreases, DOWN(5) increases; range ~[0,84]; ~12/cycle (fast) |
| Ball X | **4** | range ~5–158, bounces/reverses |
| Ball Y | **14** | range ~4–88, bounces/reverses |
| Frame counter | 90 | monotonic +1, wraps — NOT the ball (old agents used this) |
| Serve-only value | 95 | only moves during the serve — useless in play |
| Castle health | 19/20 (c0 bricks), 43/44 (c1 bricks), 108 (c2), 109 (c3) | **re-verify tomorrow** |

- **FIRE (action 1) == NOOP (action 0)**, byte-for-byte over 400 steps. No catch /
  aim mechanic. The whole game is paddle positioning. (Re-confirmed this session.)
- There is only **one ball** (scanned all 128 bytes; nothing else is ball-like).

## The central dynamic: defense can BACKFIRE (verified)
- A perfect wall keeps the ball off its own castle, but the ball can get trapped
  bouncing harmlessly OR **fully deactivate**: it parks at a fixed spot (saw
  `(byte4,byte14)=(62,84)`) and stays dead for hundreds of cycles (permanently in
  one demo game). No castle can be hit → guaranteed timeout → **no winner**.
- Parking happens most when **multiple strong, synchronized walls** keep batting
  the ball cleanly. Against chaotic/random opponents the ball keeps re-serving.
- Demo of a never-ending game: `results/videos/stalemate_no_winner.mp4` (two strong
  walls + two random; the ball parks at cycle ~1356 and all 4 castles survive).

## Measured win rates — faithful (max_cycles=100000, vs 3 random, all 4 seats, 96 games)
| agent | win | loss | timeout | note |
|---|---|---|---|---|
| random | 0.229 | 0.771 | 0.00 | fair share = 0.25 |
| noop (frozen paddle) | 0.177 | 0.823 | 0.00 | paddle does matter |
| **baseline_rule (old)** | 0.177 | — | 0.00 | == frozen (tracks byte 90 = useless) |
| **sweep (recent "best")** | 0.156 | 0.70 | **0.15** | below random |
| **old rule_agent** | 0.302 | 0.70 | 0.00 | above random, but not 0.38 |
| **PPO (submission_agent)** | 0.083 | 0.53 | **0.385** | turtles → times out |
| track_xinvy (correct-ball diagonal) | 0.281 | 0.72 | 0.00 | leaky+active sweet spot |
| static | 0.208 | 0.79 | 0.00 | |

## New agent (IN PROGRESS — do not consider final)
- Per-corner velocity-aware interceptor:
  `target = clip(a + bx*Xn + by*Yn + vx*dXn + vy*dYn) * paddle_range`, tuned per
  corner directly on win rate (timeouts count as losses, so turtling is punished).
- **Corner 0 tuned & validated: win 0.42, 0 timeouts** (vs 0.23 random, 0.30 old).
  Corners 1–3 not yet tuned (stopped early to spare the laptop).
- Files: `agents/warlord_agent.py` (self-detect + per-corner params), `best_params.json`.

## What still needs UNDERSTANDING (focus for tomorrow, before more tuning)
1. **Corner geometry.** Which player = which screen corner; map the L-shaped paddle
   arc; derive the exact ball→paddle intercept mapping per corner (principled, not
   curve-fitted). Linear+velocity is only an approximation.
2. **Ball parking / re-serve.** What exactly triggers deactivation? Can our policy
   avoid *causing* it, to minimise our own timeouts?
3. **Health/brick bytes.** Verify per-corner; build a reliable "our castle is about
   to be hit" signal.
4. **Deflection physics.** Does paddle position/velocity change the ball's outgoing
   angle? If so, is any aiming/offense possible despite FIRE being a no-op?
5. **Self-detection.** Measure zero-arg corner-detection reliability and harden it.
6. **Opponents.** Tournament rivals are likely better than random; sanity-check vs
   the rule baseline and check robustness, not just vs-random win rate.

## Corrections owed to the formal deliverables (do later, once findings are final)
- `report/report.md` and `README.md` repeat the wrong ball bytes and the "~0.38 /
  fair-share ceiling" claim. Fix after the new agent is finalised.

## Investigation scripts created this session
- `probe_truth.py` — max_cycles, FIRE vs NOOP, action→paddle, natural resolution.
- `calib.py`, `rescan.py` — find the real ball bytes & paddle ranges.
- `tourney.py` + `controllers.py` — faithful harness + candidate control laws.
- `bakeoff.py` — compare control laws.
- `measure.py` — win/loss/timeout/outlast under faithful conditions (the key tool).
- `optimize.py` — per-corner win-rate tuning (multiprocessing; pass fewer procs!).
- `bench_final.py` — final benchmark + self-detection accuracy.
- `make_stall_video.py` — record the no-winner timeout demo.

## Running gently (avoid the fan storm)
`optimize.py` / `measure.py` use a multiprocessing Pool sized to all cores. Reduce
it (e.g. 2–3 procs). NOTE: killing the parent process leaves **orphaned workers**
running at 100% — they appear as Homebrew `python -c` (the venv symlinks to it), so
kill those too, not just `*optimize.py*`.
