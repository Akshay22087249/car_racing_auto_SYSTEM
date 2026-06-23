"""Tournament agent for Warlords (RAM mode). Self-contained: depends only on
numpy, so it runs even where torch is unavailable.

It runs a small MLP policy (trained by self-play PPO and exported to
policy_weights.npz next to this file). Because every player receives the same
128-byte console RAM, the agent first works out which paddle is its own -- only
its own actions move its own paddle -- then feeds the policy a compact feature
vector (its paddle, the ball, all castle healths) read from the RAM, with the
last STACK_K frames stacked so the ball's velocity is visible. This matches the
observation used during training exactly.

Drop submission_agent.py and policy_weights.npz into the tournament folder and
import it for your slot. In the tournament notebook each agent maps to a fixed
seat (Agent1 -> first_0, Agent2 -> second_0, ...), so pass that seat for a
guaranteed-correct corner; runtime detection is only a fallback:

    from submission_agent import WarlordsAgent
    agent1 = WarlordsAgent(player_index=0)   # 0,1,2,3 for first/second/third/fourth_0
"""
import os
import sys
from collections import deque

import numpy as np

PADDLE_BYTES = [91, 92, 93, 94]    # paddle position per corner
BALL_BYTES = [90, 95, 111]         # ball position / motion
HEALTH_BYTES = [108, 109]          # third/fourth castle health (only fall during play)
DOWN, UP = 5, 2
# A fixed up/down "barcode" of runs: the runs give our paddle a strong, clean
# drift to correlate against, and the specific pattern is unique to us, so an
# opponent that is also probing follows a different stream and is not mistaken
# for our paddle.
PROBE_SEQ = (UP, UP, UP, UP, DOWN, DOWN, DOWN, DOWN, UP, UP, UP,
             DOWN, DOWN, DOWN, DOWN, UP, UP, UP, UP, DOWN, DOWN, DOWN)
STACK_K = 3
N_FEATURES = 11
OBS_DIM = STACK_K * N_FEATURES + 4
WEIGHTS = os.path.join(os.path.dirname(__file__), "policy_weights.npz")

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int16)
_HEALTH = [("bricks", (19, 20), 8.0), ("bricks", (43, 44), 8.0),
           ("value", 108, 212.0), ("value", 109, 98.0)]


def _castle_health(ram):
    out = np.empty(4, dtype=np.float32)
    for i, (kind, idx, full) in enumerate(_HEALTH):
        out[i] = (_POPCOUNT[ram[list(idx)]].sum() / full) if kind == "bricks" else ram[idx] / full
    return np.clip(out, 0.0, 1.0)


def _features(ram, corner):
    ram = ram.astype(np.float32)
    health = _castle_health(ram.astype(np.uint8))
    feat = [ram[91 + corner] / 255.0, health[corner]]
    feat += [ram[b] / 255.0 for b in BALL_BYTES]
    for j in range(4):
        if j != corner:
            feat += [ram[91 + j] / 255.0, health[j]]
    return np.asarray(feat, dtype=np.float32)


class WarlordsAgent:
    def __init__(self, player_index=None, weights_path=WEIGHTS, deterministic=True):
        self.fixed_index = player_index
        self.deterministic = deterministic
        self.rng = np.random.RandomState(0)
        try:
            w = np.load(weights_path)
            self.w0, self.b0 = w["w0"], w["b0"]
            self.w1, self.b1 = w["w1"], w["b1"]
            self.w2, self.b2 = w["w2"], w["b2"]
            self.have_policy = True
        except Exception:
            self.have_policy = False
            sys.stderr.write("WarlordsAgent: no policy weights found, using ball-tracking fallback\n")
        self._reset_game()

    def _reset_game(self):
        self.frames = deque(maxlen=STACK_K)
        self.obs_log, self.acts = [], []
        self.index = self.fixed_index
        self._last = None
        self.prev = None

    def _is_new_game(self, obs):
        if self.prev is None:
            return True
        return any(int(obs[b]) > int(self.prev[b]) + 8 for b in HEALTH_BYTES)

    def _probe_action(self):
        return PROBE_SEQ[min(len(self.obs_log) - 1, len(PROBE_SEQ) - 1)]

    def _detect_index(self):
        # Correlate each paddle byte's signed movement with our own up/down stream.
        if self.index is not None or len(self.obs_log) < 18:
            return
        obs = np.asarray(self.obs_log, dtype=np.int16)
        d = np.clip(np.diff(obs, axis=0), -20, 20).astype(np.float64)
        acts = np.asarray(self.acts[:len(d)])
        signal = (acts == UP).astype(np.float64) - (acts == DOWN).astype(np.float64)
        if int((signal != 0).sum()) < 6:
            return
        scores = [abs(float(d[:, b] @ signal)) for b in PADDLE_BYTES]
        self.index = int(np.argmax(scores))

    def _push_frame(self, ram, corner):
        f = _features(ram, corner)
        if not self.frames:
            for _ in range(STACK_K):
                self.frames.append(f)
        else:
            self.frames.append(f)

    def _policy_action(self, corner):
        x = np.zeros(OBS_DIM, dtype=np.float32)
        for i, f in enumerate(self.frames):
            x[i * N_FEATURES:(i + 1) * N_FEATURES] = f
        x[STACK_K * N_FEATURES + corner] = 1.0
        h = np.tanh(self.w0 @ x + self.b0)
        h = np.tanh(self.w1 @ h + self.b1)
        logits = self.w2 @ h + self.b2
        if self.deterministic:
            return int(np.argmax(logits))
        p = np.exp(logits - logits.max()); p /= p.sum()
        return int(self.rng.choice(len(p), p=p))

    def _fallback_action(self, ram):
        if self.index is None:
            return self._probe_action()
        paddle, ball = int(ram[PADDLE_BYTES[self.index]]), int(ram[BALL_BYTES[0]])
        return UP if paddle < ball - 2 else DOWN if paddle > ball + 2 else 0

    def act(self, observation):
        obs = np.asarray(observation, dtype=np.uint8).reshape(-1)[:128]
        if self._is_new_game(obs):
            self._reset_game()
        self.prev = obs
        if self.index is None:  # only log during the detection phase
            self.obs_log.append(obs.astype(np.int16))
            if self._last is not None:
                self.acts.append(self._last)
            self._detect_index()

        if self.have_policy and self.index is not None:
            self._push_frame(obs, self.index)
            action = self._policy_action(self.index)
        elif self.have_policy:
            action = self._probe_action()  # directional probe until detected
        else:
            action = self._fallback_action(obs)
        self._last = action
        return action
