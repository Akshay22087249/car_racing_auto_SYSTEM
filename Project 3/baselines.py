"""Baseline policies for Warlords. Each exposes act(observation) -> int and is
safe to reuse across consecutive games (it re-detects a fresh game itself).

All four players receive the same 128-byte RAM, so the rule-based agent first
works out which paddle byte is its own (only its own actions move its paddle),
then steers that paddle to shadow the ball.
"""
import numpy as np

from ram_map import PADDLE_BYTES, BALL_BYTES, detect_self_paddle, UP, DOWN, NOOP

RESET_BYTE_JUMP = 40  # this many changed bytes between calls => a new game started


class _GameTracker:
    """Shared bookkeeping: detect a new game and infer our own paddle byte."""

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.prev_obs = None
        self.acts, self.obs_log = [], []
        self.paddle_byte = None
        self.t = 0

    def is_new_game(self, obs):
        if self.prev_obs is None:
            return True
        return int((obs != self.prev_obs).sum()) > RESET_BYTE_JUMP

    def update(self, obs, last_action):
        if self.is_new_game(obs):
            self.reset_state()
        self.prev_obs = obs.copy()
        self.t += 1
        if self.paddle_byte is None:  # log (kept aligned) only while still detecting
            self.obs_log.append(obs.astype(np.int16))
            if last_action is not None:
                self.acts.append(last_action)
            _, byte = detect_self_paddle(self.acts, self.obs_log)
            if byte is not None:
                self.paddle_byte = byte


class RandomAgent:
    """Lower bound: uniform random actions."""

    def __init__(self, n_actions=6, seed=None):
        self.n_actions = n_actions
        self.rng = np.random.RandomState(seed)

    def act(self, observation):
        return int(self.rng.randint(self.n_actions))


class RuleBasedAgent:
    """Detect our paddle, then steer it to track the ball's coordinate.

    Until the paddle is identified it sweeps up/down (active blocking, already
    better than random). The ball<->paddle coordinate ranges are learned online
    so no hand-tuned screen geometry is needed.
    """

    def __init__(self, player_index=None, ball_byte=BALL_BYTES[0]):
        self.ball_byte = ball_byte
        self.tr = _GameTracker()
        self.fixed_byte = PADDLE_BYTES[player_index] if player_index is not None else None
        self._last = None
        self._probe_rng = np.random.RandomState(np.random.randint(1 << 30))
        self._p_lo, self._p_hi = 255, 0
        self._b_lo, self._b_hi = 255, 0

    def act(self, observation):
        obs = np.asarray(observation, dtype=np.uint8)
        self.tr.update(obs, self._last)
        pb = self.fixed_byte if self.fixed_byte is not None else self.tr.paddle_byte

        if pb is None:
            # unique random up/down probe so detect_self_paddle can spot the
            # paddle that follows OUR command stream (not a colliding opponent's)
            self._last = UP if self._probe_rng.rand() < 0.5 else DOWN
            return self._last

        paddle = int(obs[pb])
        ball = int(obs[self.ball_byte])
        self._p_lo, self._p_hi = min(self._p_lo, paddle), max(self._p_hi, paddle)
        self._b_lo, self._b_hi = min(self._b_lo, ball), max(self._b_hi, ball)

        # map the ball coordinate into the paddle's observed range
        b_span = max(self._b_hi - self._b_lo, 1)
        p_span = max(self._p_hi - self._p_lo, 1)
        target = self._p_lo + (ball - self._b_lo) / b_span * p_span

        if paddle < target - 2:
            self._last = UP
        elif paddle > target + 2:
            self._last = DOWN
        else:
            self._last = NOOP
        return self._last
