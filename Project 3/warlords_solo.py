"""Single-agent view of Warlords for opponent-diversity training.

Naive parameter-shared self-play collapses here: with four equally-good copies
the game just lasts long for everyone, so there is no pressure to actually
outplay anyone, and the result is brittle against real opponents. Instead we let
the learner control ONE randomly chosen corner each episode and drive the other
three with scripted opponents sampled from a pool (random / rule-based / a frozen
snapshot of the learner). The learner is rewarded for defending its own castle
and outlasting the others, and because its corner is randomised it learns to
defend any corner -- which the tournament requires.

The observation is a compact, control-relevant feature vector (paddle, ball,
castle healths) read from the reverse-engineered RAM rather than the raw 128
bytes, and STACK_K successive frames are stacked so the policy sees the ball's
velocity. This is what lets a small MLP actually learn to intercept.
"""
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.atari import warlords_v3

from baselines import RandomAgent, RuleBasedAgent
from ram_map import castle_health, extract_features, N_FEATURES

STACK_K = 3
OBS_DIM = STACK_K * N_FEATURES + 4


def build_obs(feat_frames, corner):
    """feat_frames: iterable of STACK_K feature vectors (oldest..newest)."""
    v = np.zeros(OBS_DIM, dtype=np.float32)
    for i, f in enumerate(feat_frames):
        v[i * N_FEATURES:(i + 1) * N_FEATURES] = f
    v[STACK_K * N_FEATURES + corner] = 1.0
    return v


class FrozenPolicyOpponent:
    """Runs a frozen numpy MLP snapshot of the learner from a known corner."""

    def __init__(self, weights, corner):
        self.w0, self.b0 = weights["w0"], weights["b0"]
        self.w1, self.b1 = weights["w1"], weights["b1"]
        self.w2, self.b2 = weights["w2"], weights["b2"]
        self.corner = corner
        self.frames = deque(maxlen=STACK_K)

    def act(self, ram):
        f = extract_features(ram, self.corner)
        if not self.frames:
            for _ in range(STACK_K):
                self.frames.append(f)
        else:
            self.frames.append(f)
        x = build_obs(self.frames, self.corner)
        h = np.tanh(self.w0 @ x + self.b0)
        h = np.tanh(self.w1 @ h + self.b1)
        return int(np.argmax(self.w2 @ h + self.b2))


class WarlordsSoloEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponent_pool=("random", "rule"), max_cycles=8000,
                 outlast_bonus=2.0, win_scale=10.0, def_weight=1.5, off_weight=2.0,
                 time_penalty=0.0, stalemate_loss=False, frozen_path=None):
        self.observation_space = spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Discrete(6)
        self.opponent_pool = list(opponent_pool)
        self.max_cycles = max_cycles
        self.outlast_bonus = outlast_bonus
        self.win_scale = win_scale
        self.def_weight = def_weight   # penalty per unit of own-castle damage (defense)
        self.off_weight = off_weight   # reward per unit of opponent-castle damage (offense)
        self.time_penalty = time_penalty  # per-step cost: punishes stalling, forces decisive play
        self.stalemate_loss = stalemate_loss  # truncation while alive but not winner counts as a loss
        self.frozen_path = frozen_path
        self._frozen = None
        self.frames = deque(maxlen=STACK_K)
        self.env = warlords_v3.parallel_env(obs_type="ram", max_cycles=max_cycles)
        self.rng = np.random.RandomState()

    def _load_frozen(self):
        if self.frozen_path is None:
            return None
        try:
            return dict(np.load(self.frozen_path))
        except Exception:
            return None

    def _make_opp(self, name, corner):
        if name == "rule":
            return RuleBasedAgent()
        if name == "self" and self._frozen is not None:
            return FrozenPolicyOpponent(self._frozen, corner)
        return RandomAgent(seed=self.rng.randint(1 << 30))

    def _push(self, ram):
        self.frames.append(extract_features(ram, self.k))
        return build_obs(self.frames, self.k)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self._frozen = self._load_frozen()
        obs, _ = self.env.reset(seed=int(self.rng.randint(1 << 30)))
        self.agents = list(self.env.agents)
        self.k = int(self.rng.randint(4))
        self.me = self.agents[self.k]
        self.opps = {a: self._make_opp(self.rng.choice(self.opponent_pool), i)
                     for i, a in enumerate(self.agents) if a != self.me}
        self.last_obs = obs
        self.prev_health = castle_health(obs[self.me])
        self.frames.clear()
        f = extract_features(obs[self.me], self.k)
        for _ in range(STACK_K):
            self.frames.append(f)
        return build_obs(self.frames, self.k), {}

    def step(self, action):
        actions = {}
        for a in self.env.agents:
            actions[a] = int(action) if a == self.me else self.opps[a].act(self.last_obs[a])
        obs, rew, term, trunc, info = self.env.step(actions)
        self.last_obs = obs

        me_done = term.get(self.me, False) or trunc.get(self.me, False) or self.me not in self.env.agents
        newly_dead = sum(1 for a in rew if term.get(a, False) and a != self.me)
        native = rew.get(self.me, 0.0)
        ram = obs[self.me] if self.me in obs else self.last_obs[self.me]

        # dense per-step shaping: penalise damage to our castle, reward damage to others
        health = castle_health(ram)
        dmg = np.clip(self.prev_health - health, 0.0, None)
        self.prev_health = health
        reward = native * self.win_scale
        reward += -self.def_weight * dmg[self.k] + self.off_weight * (dmg.sum() - dmg[self.k])
        if not me_done:
            reward += self.outlast_bonus * newly_dead - self.time_penalty
        elif self.stalemate_loss and trunc.get(self.me, False) and native <= 0:
            reward -= self.win_scale  # stalling to the time limit is as bad as losing
        won = native > 0  # +1 only goes to the last player standing
        done = me_done or len(self.env.agents) == 0

        out = self._push(ram)
        if done:
            info = dict(info, won=bool(won), corner=self.k)
        return out, float(reward), bool(done), False, info
