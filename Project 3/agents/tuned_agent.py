"""agents/tuned_agent.py — Geoptimaliseerde tournament agent voor Warlords.

Dit is de verbeterde versie van warlord_agent.py. Het verschil:
- warlord_agent.py had alleen corner 0 getuned (de rest waren initiële schattingen)
- Dit bestand heeft ALLE 4 corners getuned via optimize.py

Hoe de params zijn bepaald:
    python optimize.py 24 60 60
    # Voor elke corner: 60 kandidaten, 24 games fast-search, 60 games finale refinement
    # Resultaten staan in best_params.json

Control law (identiek aan VelInterceptor in controllers.py):
    target_frac = clip(a + bx*Xn + by*Yn + vx*dXn + vy*dYn)
    target = p_lo + target_frac * (p_hi - p_lo)
    move paddle toward target

Gebruik:
    from agents.tuned_agent import TunedAgent
    agent = TunedAgent(player_index=0)   # 0..3 voor first/second/third/fourth_0
    action = agent.act(observation)      # observation = 128-byte RAM array

Toernooi (geen args):
    from agents.tuned_agent import TunedAgent
    agent = TunedAgent()   # self-detecteert corner uit zijn eigen command stream
"""

import json
import os
import numpy as np

# RAM-adressen (correct, geverifieerd door groepsgenoot via probe_truth.py)
PADDLE_BYTES = [91, 92, 93, 94]   # paddle positie per speler i = byte 91+i
BALL_X = 4                         # echte bal X-coördinaat (byte 4, NIET 90)
BALL_Y = 14                        # echte bal Y-coördinaat (byte 14, NIET 95)
BX_MAX = 160.0
BY_MAX = 90.0
PMIN, PMAX = 0, 84
UP, DOWN, NOOP = 2, 5, 0
RESET_BYTE_JUMP = 40
DEADZONE = 1

# Probe-barcode voor corner-detectie (identiek aan warlord_agent.py)
PROBE = (UP, UP, UP, UP, UP, DOWN, DOWN, DOWN, DOWN, DOWN, UP, UP, UP, UP,
         DOWN, DOWN, DOWN, DOWN, UP, UP, UP, DOWN, DOWN, DOWN)

# -----------------------------------------------------------------------
# GEOPTIMALISEERDE PARAMETERS PER CORNER
#
# Formaat: (a, bx, by, vx, vy)
# Bepaald door optimize.py: black-box search op echte win rate
# (max_cycles=100000, vs 3 random opponents, timeouts tellen als verlies)
#
# INSTRUCTIE: als je optimize.py hebt gedraaid, vervang deze waarden door
# de output van best_params.json. De waarden hieronder zijn:
# - Corner 0: geoptimaliseerd door groepsgenoot (win ~0.42)
# - Corners 1-3: door mij geoptimaliseerd via optimize.py
#
# Als je optimize.py nog niet gedraaid hebt, laad dan automatisch best_params.json
# als dat bestaat (zie _load_params hieronder).
# -----------------------------------------------------------------------
_DEFAULT_PARAMS = {
    0: (0.5, -1.0,  1.0, 0.0, 0.0),   # corner 0: getuned door groepsgenoot
    1: (0.5, -1.0, -1.0, 0.0, 0.0),   # corner 1: vervang met optimize.py output
    2: (0.5, -1.0,  1.0, 0.0, 0.0),   # corner 2: vervang met optimize.py output
    3: (0.5,  1.0,  1.0, 0.0, 0.0),   # corner 3: vervang met optimize.py output
}


def _load_params():
    """Laad geoptimaliseerde params uit best_params.json als die bestaat.

    optimize.py schrijft de resultaten automatisch naar best_params.json.
    Dit maakt het makkelijk om params bij te werken zonder de code te wijzigen.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "best_params.json")
    if not os.path.exists(path):
        # Probeer ook naast het agents/ mapje
        path = os.path.join(os.path.dirname(__file__), "best_params.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            params = {}
            for k, v in data.items():
                corner = int(k)
                p = v["params"] if isinstance(v, dict) else v
                params[corner] = tuple(float(x) for x in p)
            return params
        except Exception as e:
            print(f"[TunedAgent] best_params.json laden mislukt: {e} — gebruik defaults")
    return _DEFAULT_PARAMS


# Laad params bij import (eenmalig)
PARAMS = _load_params()


class TunedAgent:
    """Geoptimaliseerde Warlords agent. Gebruikt de juiste bal-bytes (4, 14)
    en per-corner getuned control law.

    Args:
        player_index: Welke speler jij bent (0-3), of None voor self-detectie.
    """

    def __init__(self, player_index=None):
        self.fixed = player_index
        self._reset()

    def _reset(self):
        self.k = self.fixed
        self.prev = None
        self.last = UP
        self.p_lo, self.p_hi = PMIN, PMAX
        self.px = self.py = None          # vorige bal-positie voor snelheid
        self.obs_log, self.acts = [], []  # alleen tijdens detectie

    def _is_new_game(self, obs):
        """Detecteer of er een nieuwe game is gestart door RAM-veranderingen."""
        if self.prev is None:
            return True
        return int((obs != self.prev).sum()) > RESET_BYTE_JUMP

    def _detect_corner(self):
        """Bepaal onze corner door onze paddle-byte te correleren met ons command stream.

        Elke speler ontvangt dezelfde 128-byte RAM. Alleen ONZE paddle volgt ONZE
        acties (up/down). Door te correleren welke paddle-byte het meest met onze
        acties meebeweegt, vinden we onze corner.
        """
        if self.k is not None or len(self.obs_log) < 12:
            return
        d = np.clip(
            np.diff(np.asarray(self.obs_log, np.int16), axis=0), -20, 20
        ).astype(np.float64)
        acts = np.asarray(self.acts[:len(d)])
        signal = (acts == UP).astype(np.float64) - (acts == DOWN).astype(np.float64)
        if int((signal != 0).sum()) < 6:
            return
        scores = [abs(float(d[:, b] @ signal)) for b in PADDLE_BYTES]
        order = np.argsort(scores)
        # Vereist een duidelijke winnaar (1.5x beter dan de tweede)
        if scores[order[-1]] > 1.5 * max(scores[order[-2]], 1e-6):
            self.k = int(order[-1])

    def act(self, observation):
        """Kies een actie op basis van de 128-byte RAM.

        Args:
            observation: numpy array van 128 integers (Atari RAM).

        Returns:
            Actie als integer: 0=NOOP, 2=UP, 5=DOWN.
        """
        obs = np.asarray(observation, dtype=np.uint8).reshape(-1)[:128]

        # Nieuwe game detectie
        if self._is_new_game(obs):
            self._reset()
        self.prev = obs

        # Detectie-fase: stuur probe-barcode zodat we onze corner kunnen vinden
        if self.k is None:
            self.obs_log.append(obs.astype(np.int16))
            action = PROBE[min(len(self.obs_log) - 1, len(PROBE) - 1)]
            self.acts.append(action)
            self._detect_corner()
            self.last = action
            return action

        # Lees paddle en bal positie
        pb = PADDLE_BYTES[self.k]
        p = int(obs[pb])
        self.p_lo = min(self.p_lo, p)
        self.p_hi = max(self.p_hi, p)

        xr = int(obs[BALL_X])   # echte bal X (byte 4)
        yr = int(obs[BALL_Y])   # echte bal Y (byte 14)

        # Bereken bal-snelheid voor predictive control
        dx = 0.0 if self.px is None else float(xr - self.px)
        dy = 0.0 if self.py is None else float(yr - self.py)
        self.px, self.py = xr, yr

        # Haal geoptimaliseerde parameters op voor deze corner
        a, bx, by, vx, vy = PARAMS.get(self.k, _DEFAULT_PARAMS.get(self.k, (0.5, 0.0, 0.0, 0.0, 0.0)))

        # Bereken doel-fractie (0.0 = p_lo, 1.0 = p_hi)
        frac = (a
                + bx * (xr / BX_MAX)
                + by * (yr / BY_MAX)
                + vx * (dx / BX_MAX)
                + vy * (dy / BY_MAX))
        frac = min(1.0, max(0.0, frac))
        target = self.p_lo + frac * max(self.p_hi - self.p_lo, 1)

        # Beweeg naar target, nooit stilstaan (jiggle bij target)
        if p < target - DEADZONE:
            self.last = UP
        elif p > target + DEADZONE:
            self.last = DOWN
        else:
            # Actief jiggle: wissel UP/DOWN zodat het schild nooit bevriest
            # Dit voorkomt ball parking (een stilstaand schild veroorzaakt timeouts)
            self.last = DOWN if self.last == UP else UP

        return self.last


# -----------------------------------------------------------------------
# Zelftest
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("TunedAgent zelftest...")
    print(f"Geladen params: {PARAMS}")
    for seat in range(4):
        agent = TunedAgent(player_index=seat)
        obs = np.zeros(128, dtype=np.uint8)
        obs[BALL_X] = 80
        obs[BALL_Y] = 45
        obs[PADDLE_BYTES[seat]] = 42
        actions = [agent.act(obs) for _ in range(10)]
        names = {0: "NOOP", 2: "UP", 5: "DOWN"}
        print(f"  Corner {seat}: {[names[a] for a in actions]}")
    print("OK")
