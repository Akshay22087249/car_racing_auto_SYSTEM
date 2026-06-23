"""Reverse-engineered Warlords RAM layout (obs_type="ram", 128 bytes).

Found by isolating each player's effect on the RAM: run two identical
environments that differ only in one player's action and diff the bytes.
The bytes that move are caused solely by that player.

Findings:
  RAM[91 + i]  -> paddle position of player i  (i = 0..3)
  RAM[90], RAM[111] change every frame -> ball coordinates / motion
All four players receive the *same* 128-byte console RAM, so an agent must
work out which paddle is its own at runtime (see detect_self_paddle).
"""
import numpy as np

PLAYER_NAMES = ["first_0", "second_0", "third_0", "fourth_0"]
PADDLE_BYTES = [91, 92, 93, 94]   # paddle position per player index
BALL_BYTES = [90, 111]            # every-frame movers (ball position)

# Castle health per corner, found by correlating decreasing RAM bytes with each
# player's elimination. third/fourth expose a clean summary byte; first/second
# are read from their brick-bitmask bytes (popcount). All normalised to [0,1].
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int16)
_HEALTH = [("bricks", (19, 20), 8.0),     # first_0
           ("bricks", (43, 44), 8.0),     # second_0
           ("value", 108, 212.0),         # third_0
           ("value", 109, 98.0)]          # fourth_0


def castle_health(ram):
    """Return the four corners' castle health as a [0,1] array."""
    ram = np.asarray(ram, dtype=np.uint8)
    out = np.empty(4, dtype=np.float32)
    for i, (kind, idx, full) in enumerate(_HEALTH):
        if kind == "bricks":
            out[i] = _POPCOUNT[ram[list(idx)]].sum() / full
        else:
            out[i] = ram[idx] / full
    return np.clip(out, 0.0, 1.0)


BALL_FEATURE_BYTES = [90, 95, 111]  # ball position / motion candidates
N_FEATURES = 11


def extract_features(ram, corner):
    """Compact, control-relevant observation for the given corner, built from the
    reverse-engineered RAM. Raw 128-byte RAM forces the network to first find the
    few useful bytes; handing it the paddle/ball/health directly makes the control
    problem easy to learn. Stacking successive feature vectors supplies velocity.

    Layout: [my paddle, my health, ball x/y/z, then (paddle, health) per opponent].
    """
    ram = np.asarray(ram, dtype=np.uint8).astype(np.float32)
    health = castle_health(ram)
    feat = [ram[91 + corner] / 255.0, health[corner]]
    feat += [ram[b] / 255.0 for b in BALL_FEATURE_BYTES]
    for j in range(4):
        if j != corner:
            feat += [ram[91 + j] / 255.0, health[j]]
    return np.asarray(feat, dtype=np.float32)

NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5


def paddle_byte(player_index: int) -> int:
    return PADDLE_BYTES[player_index]


def detect_self_paddle(action_history, obs_history):
    """Infer which paddle byte belongs to us from our own action/observation log.

    Only our own paddle follows OUR command stream. We correlate each paddle
    byte's signed movement with our own up(+1)/down(-1) signal, so the byte that
    tracks our (unique, pseudo-random) probe is ours. Correlating with our own
    signal rather than a fixed schedule makes this robust even when opponents are
    also probing -- their paddles follow their command stream, not ours.

    action_history[i]: the action we took right after observing obs_history[i].
    Returns (player_index, paddle_byte) or (None, None) if undecided.
    """
    n = min(len(obs_history), len(action_history))
    if n < 10:
        return None, None
    obs = np.asarray(obs_history[:n], dtype=np.int16)
    d = np.clip(np.diff(obs, axis=0), -20, 20).astype(np.float64)  # signed, clip RAM wrap
    acts = np.asarray(action_history[:len(d)])
    signal = (acts == UP).astype(np.float64) - (acts == DOWN).astype(np.float64)
    if int((signal != 0).sum()) < 6:
        return None, None
    scores = [abs(float(d[:, b] @ signal)) for b in PADDLE_BYTES]
    best = int(np.argmax(scores))
    return best, PADDLE_BYTES[best]
