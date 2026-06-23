"""Candidate paddle control laws for Warlords, plus opponent/agent factories.
Kept importable (top-level functions/classes) so multiprocessing spawn can pickle
the factories passed to tourney.evaluate.
"""
import numpy as np

from baselines import RandomAgent, RuleBasedAgent

NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
PADDLE_BYTES = [91, 92, 93, 94]
PMIN, PMAX = 0, 84
# The true ball coordinates (reverse-engineered): byte 4 = X, byte 14 = Y.
# byte 90 is a frame counter, byte 95 only moves during the serve -> both useless.
BALL_X, BALL_Y = 4, 14
BX_MAX, BY_MAX = 160.0, 90.0
HEALTH_BYTES = [108, 109]


class LawAgent:
    """Move our corner's paddle toward a target derived from a named control law.
    Fixed corner (player_index). Active: never sits still when it should move."""

    def __init__(self, player_index, law="track95", deadzone=1):
        self.k = player_index
        self.pb = PADDLE_BYTES[player_index]
        self.law = law
        self.deadzone = deadzone
        self.last = UP
        self.sweep_dir = UP
        self.p_lo, self.p_hi = PMIN, PMAX
        self.prev = None

    def _move_to(self, p, target):
        if p < target - self.deadzone:
            self.last = UP
        elif p > target + self.deadzone:
            self.last = DOWN
        else:  # at target: jiggle so the shield never freezes
            self.last = DOWN if self.last == UP else UP
        return self.last

    def _sweep(self, p):
        if p >= self.p_hi - 2:
            self.sweep_dir = DOWN
        if p <= self.p_lo + 2:
            self.sweep_dir = UP
        self.last = self.sweep_dir
        return self.last

    def act(self, observation):
        obs = np.asarray(observation, dtype=np.uint8).reshape(-1)[:128]
        p = int(obs[self.pb])
        self.p_lo, self.p_hi = min(self.p_lo, p), max(self.p_hi, p)
        x = int(obs[BALL_X]) / BX_MAX
        y = int(obs[BALL_Y]) / BY_MAX
        law = self.law
        if law == "sweep":
            return self._sweep(p)
        if law == "static":
            mid = (self.p_lo + self.p_hi) // 2
            return self._move_to(p, mid)
        if law == "random":
            self.last = UP if (int(obs[90])) % 2 else DOWN
            return self.last
        # tracking laws on the true ball coords (X=byte4, Y=byte14)
        if law == "trackx":
            frac = x
        elif law == "trackxinv":
            frac = 1.0 - x
        elif law == "tracky":
            frac = y
        elif law == "trackyinv":
            frac = 1.0 - y
        elif law == "track_xy":
            frac = 0.5 * (x + y)
        elif law == "track_xyinv":
            frac = 0.5 * ((1 - x) + y)
        elif law == "track_xinvy":
            frac = 0.5 * (x + (1 - y))
        elif law == "track_xinvyinv":
            frac = 0.5 * ((1 - x) + (1 - y))
        else:
            frac = 0.5
        frac = min(1.0, max(0.0, frac))
        target = self.p_lo + frac * max(self.p_hi - self.p_lo, 1)
        return self._move_to(p, target)


class LinearInterceptor:
    """target_frac = clip(a + bx*X + by*Y); move paddle toward it. Params tuned
    per corner by black-box search on the real win/survival objective."""

    def __init__(self, player_index, a=0.5, bx=0.0, by=0.0, deadzone=1):
        self.k = player_index
        self.pb = PADDLE_BYTES[player_index]
        self.a, self.bx, self.by = a, bx, by
        self.deadzone = deadzone
        self.last = UP
        self.p_lo, self.p_hi = PMIN, PMAX

    def act(self, observation):
        obs = np.asarray(observation, dtype=np.uint8).reshape(-1)[:128]
        p = int(obs[self.pb])
        self.p_lo, self.p_hi = min(self.p_lo, p), max(self.p_hi, p)
        x = int(obs[BALL_X]) / BX_MAX
        y = int(obs[BALL_Y]) / BY_MAX
        frac = self.a + self.bx * x + self.by * y
        frac = min(1.0, max(0.0, frac))
        target = self.p_lo + frac * max(self.p_hi - self.p_lo, 1)
        if p < target - self.deadzone:
            self.last = UP
        elif p > target + self.deadzone:
            self.last = DOWN
        else:
            self.last = DOWN if self.last == UP else UP
        return self.last


class NoopAgent:
    """Frozen paddle: does nothing. Tests whether paddle control matters at all."""
    def act(self, observation):
        return NOOP


class VelInterceptor:
    """target_frac = clip(a + bx*X + by*Y + vx*dX + vy*dY); move toward it.
    Includes ball velocity (dX,dY) so the paddle can lead the ball. Params tuned
    per corner on the real win-rate objective."""

    def __init__(self, player_index, a=0.5, bx=0.0, by=0.0, vx=0.0, vy=0.0, deadzone=1):
        self.k = player_index
        self.pb = PADDLE_BYTES[player_index]
        self.a, self.bx, self.by, self.vx, self.vy = a, bx, by, vx, vy
        self.deadzone = deadzone
        self.last = UP
        self.p_lo, self.p_hi = PMIN, PMAX
        self.px = self.py = None

    def act(self, observation):
        obs = np.asarray(observation, dtype=np.uint8).reshape(-1)[:128]
        p = int(obs[self.pb])
        self.p_lo, self.p_hi = min(self.p_lo, p), max(self.p_hi, p)
        xr = int(obs[BALL_X]); yr = int(obs[BALL_Y])
        dx = 0.0 if self.px is None else (xr - self.px)
        dy = 0.0 if self.py is None else (yr - self.py)
        self.px, self.py = xr, yr
        frac = (self.a + self.bx * (xr / BX_MAX) + self.by * (yr / BY_MAX)
                + self.vx * (dx / BX_MAX) + self.vy * (dy / BY_MAX))
        frac = min(1.0, max(0.0, frac))
        target = self.p_lo + frac * max(self.p_hi - self.p_lo, 1)
        if p < target - self.deadzone:
            self.last = UP
        elif p > target + self.deadzone:
            self.last = DOWN
        else:
            self.last = DOWN if self.last == UP else UP
        return self.last


# ---- factories (top-level so they pickle under spawn) ----

def random_opp(seat, seed):
    return RandomAgent(seed=(seed * 7 + seat * 101 + 1) & 0x7fffffff)


def rule_opp(seat, seed):
    return RuleBasedAgent(player_index=seat)


def make_law_agent(seat, seed, law="track95"):
    return LawAgent(player_index=seat, law=law)
