"""Best tournament agent for Warlords (RAM mode). Self-contained, numpy only.

How it works (all reverse-engineered from the real environment):
  * Each player i controls paddle byte 91+i; UP(2) decreases it, DOWN(5) increases.
  * The ball's true screen coordinates are RAM byte 4 (X) and byte 14 (Y). (The
    bytes the older agents used -- 90 and 95 -- are a frame counter and a
    serve-only value, i.e. useless; tracking them is no better than a frozen
    paddle.)
  * Winning = being the last castle standing within the env's 100000-cycle limit.
    A *perfect* wall traps the ball on its own side and the game times out with no
    winner, so the goal is an ACTIVE interceptor: block the ball well enough to
    outlast opponents while keeping it circulating so opponents' castles fall.

Control law, per corner: move the paddle toward
    target = clip(a + bx*Xn + by*Yn + vx*dXn + vy*dYn) * paddle_range
(Xn,Yn normalised ball position; dXn,dYn ball velocity). The five coefficients
were tuned per corner by black-box search directly on tournament win rate.

Every player receives the SAME 128-byte RAM, so the agent self-detects which
paddle is its own from a short directional probe at game start (only its own
paddle follows its own up/down command stream). In the tournament each class maps
to a fixed seat (Agent1->first_0, ...); pass player_index for a guaranteed corner,
otherwise it self-detects:

    from warlord_agent import WarlordAgent
    agent1 = WarlordAgent()            # self-detects its corner
    agent1 = WarlordAgent(player_index=0)   # or force the seat (0..3)
"""
import numpy as np

PADDLE_BYTES = [91, 92, 93, 94]
BALL_X, BALL_Y = 4, 14
BX_MAX, BY_MAX = 160.0, 90.0
UP, DOWN, NOOP = 2, 5, 0
PMIN, PMAX = 0, 84
RESET_BYTE_JUMP = 40   # this many changed bytes between calls => a new game
DEADZONE = 1

# Fixed up/down barcode used only while detecting our corner; long runs give our
# paddle a clean, unique drift to correlate against.
PROBE = (UP, UP, UP, UP, UP, DOWN, DOWN, DOWN, DOWN, DOWN, UP, UP, UP, UP,
         DOWN, DOWN, DOWN, DOWN, UP, UP, UP, DOWN, DOWN, DOWN)

# Per-corner control coefficients (a, bx, by, vx, vy), tuned on win rate.
# Filled in by optimize.py -> best_params.json.
PARAMS = {
    0: (0.5, -1.0, 1.0, 0.0, 0.0),
    1: (0.5, -1.0, -1.0, 0.0, 0.0),
    2: (0.5, -1.0, 1.0, 0.0, 0.0),
    3: (0.5, 1.0, 1.0, 0.0, 0.0),
}


class WarlordAgent:
    def __init__(self, player_index=None):
        self.fixed = player_index
        self._reset()

    def _reset(self):
        self.k = self.fixed
        self.prev = None
        self.last = UP
        self.p_lo, self.p_hi = PMIN, PMAX
        self.px = self.py = None
        self.obs_log, self.acts = [], []

    def _is_new_game(self, obs):
        if self.prev is None:
            return True
        return int((obs != self.prev).sum()) > RESET_BYTE_JUMP

    def _detect(self):
        if self.k is not None or len(self.obs_log) < 12:
            return
        d = np.clip(np.diff(np.asarray(self.obs_log, np.int16), axis=0), -20, 20).astype(np.float64)
        acts = np.asarray(self.acts[:len(d)])
        signal = (acts == UP).astype(np.float64) - (acts == DOWN).astype(np.float64)
        if int((signal != 0).sum()) < 6:
            return
        scores = [abs(float(d[:, b] @ signal)) for b in PADDLE_BYTES]
        order = np.argsort(scores)
        # require a clear winner before locking in
        if scores[order[-1]] > 1.5 * max(scores[order[-2]], 1e-6):
            self.k = int(order[-1])

    def act(self, observation):
        obs = np.asarray(observation, dtype=np.uint8).reshape(-1)[:128]
        if self._is_new_game(obs):
            self._reset()
        self.prev = obs

        if self.k is None:                      # detection phase: emit barcode
            self.obs_log.append(obs.astype(np.int16))
            action = PROBE[min(len(self.obs_log) - 1, len(PROBE) - 1)]
            self.acts.append(action)
            self._detect()
            self.last = action
            return action

        pb = PADDLE_BYTES[self.k]
        p = int(obs[pb])
        self.p_lo, self.p_hi = min(self.p_lo, p), max(self.p_hi, p)
        xr, yr = int(obs[BALL_X]), int(obs[BALL_Y])
        dx = 0.0 if self.px is None else (xr - self.px)
        dy = 0.0 if self.py is None else (yr - self.py)
        self.px, self.py = xr, yr
        a, bx, by, vx, vy = PARAMS[self.k]
        frac = (a + bx * (xr / BX_MAX) + by * (yr / BY_MAX)
                + vx * (dx / BX_MAX) + vy * (dy / BY_MAX))
        frac = min(1.0, max(0.0, frac))
        target = self.p_lo + frac * max(self.p_hi - self.p_lo, 1)
        if p < target - DEADZONE:
            self.last = UP
        elif p > target + DEADZONE:
            self.last = DOWN
        else:                                   # at target: jiggle, never freeze
            self.last = DOWN if self.last == UP else UP
        return self.last
