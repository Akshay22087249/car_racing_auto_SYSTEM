"""evaluate_final.py — Finale evaluatie onder correcte tournament-condities.

Dit script vervangt/verbetert evaluate.py op twee punten:

1. CORRECTE condities: max_cycles=100000 (geen afkap), exact zoals de tournament
   notebook. De oude evaluate.py gebruikte max_cycles=30000 voor sommige vergelijkingen,
   waardoor rule_agent er beter uitzag dan hij werkelijk is.

2. WIN / LOSS / TIMEOUT apart: een game zonder winnaar is een TIMEOUT, geen verlies.
   Dit is cruciaal voor het rapport: PPO "verloor" nooit, het veroorzaakte timeouts.
   Dat verschil vertelt het echte verhaal (turtling vs. actief spelen).

Gebruik:
    python evaluate_final.py --games 40
    python evaluate_final.py --games 20 --quick     # snelle test, 4 seats * 20 games
    python evaluate_final.py --plots-only           # alleen grafieken van bestaande CSV

Output:
    results/final_comparison.csv   — win/loss/timeout per agent
    results/final_comparison.png   — grafiek voor het rapport
"""

import argparse
import csv
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

# Voeg de project-root toe aan het pad zodat imports werken
sys.path.insert(0, os.path.dirname(__file__))

RESULTS = os.path.join(os.path.dirname(__file__), "results")
MODELS  = os.path.join(os.path.dirname(__file__), "models")


# -----------------------------------------------------------------------
# Agent factories (top-level voor multiprocessing pickling)
# -----------------------------------------------------------------------

def _make_random(seat, seed):
    from baselines import RandomAgent
    return RandomAgent(seed=(seed * 5 + seat * 999 + 1) & 0x7fffffff)


def _make_noop(seat, seed):
    """Bevroren paddle — ondergrens voor control."""
    from controllers import NoopAgent
    return NoopAgent()


def _make_rule_agent(seat, seed):
    """De nieuwe rule_agent (volgt echte bal-bytes)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))
    from rule_agent import RuleAgent
    return RuleAgent(player_index=seat)


def _make_ppo(seat, seed):
    """Exported PPO agent (submission_agent.py)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))
    from submission_agent import WarlordsAgent
    return WarlordsAgent(player_index=seat)


def _make_tuned(seat, seed):
    """Mijn nieuwe getuned agent (alle 4 corners geoptimaliseerd)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))
    from tuned_agent import TunedAgent
    return TunedAgent(player_index=seat)


# -----------------------------------------------------------------------
# Spelloop
# -----------------------------------------------------------------------

def play_one_game(agents_by_seat, seed, max_cycles=100000):
    """Speel één Warlords game. Geeft per seat: won, timeout, survived terug.

    Args:
        agents_by_seat: lijst van 4 agents, index = seat (0=first_0, etc.)
        seed: Random seed voor reproduceerbaarheid.
        max_cycles: Maximaal aantal cycles (100000 = tournament standaard).

    Returns:
        dict: seat_index -> dict(won, timeout, survived, reward)
    """
    from pettingzoo.atari import warlords_v3
    env = warlords_v3.env(obs_type="ram", max_cycles=max_cycles)
    env.reset(seed=seed)
    names = list(env.agents)
    mapping = {names[i]: agents_by_seat[i] for i in range(len(names))}
    reward = {n: 0.0 for n in names}
    survived = {n: 0 for n in names}

    for ag in env.agent_iter():
        obs, r, term, trunc, info = env.last()
        reward[ag] += r
        if term or trunc:
            action = None
        else:
            survived[ag] += 1
            action = mapping[ag].act(obs)
        env.step(action)
    env.close()

    # Bepaal uitkomst: iemand wint alleen als reward > 0 (laatste kasteel staand)
    someone_won = max(reward.values()) > 0
    result = {}
    for i, n in enumerate(names):
        won = bool(reward[n] > 0)
        timeout = bool(not someone_won)
        result[i] = dict(
            won=won,
            timeout=timeout,
            lost=bool(not won and not timeout),
            survived=survived[n],
            reward=float(reward[n]),
        )
    return result


def evaluate_agent(name, agent_factory, opponent_factory, games, seats=(0, 1, 2, 3),
                   base_seed=1000, max_cycles=100000):
    """Evalueer één agent vs een opponent over alle 4 seats.

    Args:
        name: Naam voor logging.
        agent_factory: callable(seat, seed) -> agent
        opponent_factory: callable(seat, seed) -> agent
        games: Aantal games per seat.
        seats: Welke seats de test-agent inneemt.
        base_seed: Startseed.
        max_cycles: Maximale cyclus-limiet (gebruik altijd 100000).

    Returns:
        dict met win/loss/timeout rates en mean_survival.
    """
    wins = losses = timeouts = 0
    total_survived = 0
    total = 0

    for seat in seats:
        for g in range(games):
            seed = base_seed + g
            agents = [opponent_factory(i, seed) for i in range(4)]
            agents[seat] = agent_factory(seat, seed)
            try:
                result = play_one_game(agents, seed=seed, max_cycles=max_cycles)
                me = result[seat]
                wins += me["won"]
                losses += me["lost"]
                timeouts += me["timeout"]
                total_survived += me["survived"]
                total += 1
            except Exception as e:
                print(f"  [WARN] game mislukt (seat={seat}, g={g}): {e}")

    n = max(total, 1)
    return dict(
        agent=name,
        win_rate=round(wins / n, 3),
        loss_rate=round(losses / n, 3),
        timeout_rate=round(timeouts / n, 3),
        mean_survival=round(total_survived / n),
        games=total,
    )


# -----------------------------------------------------------------------
# Visualisatie
# -----------------------------------------------------------------------

def plot_results(rows, output_path):
    """Maak een gestapeld staafdiagram: win / timeout / loss per agent.

    Timeouts apart weergeven is cruciaal voor het rapport: PPO's "verlies"
    is eigenlijk een timeout (turtling), niet een directe defeat.

    Args:
        rows: Lijst van dicts uit evaluate_agent.
        output_path: Pad voor output PNG.
    """
    names = [r["agent"] for r in rows]
    wins     = [r["win_rate"]     for r in rows]
    timeouts = [r["timeout_rate"] for r in rows]
    losses   = [r["loss_rate"]    for r in rows]
    survival = [r["mean_survival"] for r in rows]

    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Linkergrafiek: win/timeout/loss gestapeld
    ax1.bar(x, wins,     0.55, label="Win",     color="#4CAF50")
    ax1.bar(x, timeouts, 0.55, label="Timeout", color="#FFC107", bottom=wins)
    bottom2 = [w + t for w, t in zip(wins, timeouts)]
    ax1.bar(x, losses,   0.55, label="Loss",    color="#F44336", bottom=bottom2)
    ax1.axhline(0.25, ls="--", c="gray", lw=1.5, label="Fair share (0.25)")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylabel("Rate"); ax1.set_title("Win / Timeout / Loss per agent\n(max_cycles=100000, faithful)")
    ax1.legend(fontsize=9); ax1.set_ylim(0, 1.05)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.grid(alpha=0.25, axis="y")

    # Rechtergrafiek: survival (hoe lang overleeft de agent)
    bars = ax2.bar(x, survival, 0.55, color="#5C6BC0")
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylabel("Mean survival (steps)")
    ax2.set_title("Overlevingsduur\n(langer = sterker verdediger, maar let op timeouts)")
    for bar, val in zip(bars, survival):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.grid(alpha=0.25, axis="y")

    plt.suptitle("Warlords Agent Vergelijking — Correcte Tournament Condities",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafiek opgeslagen: {output_path}")


# -----------------------------------------------------------------------
# Hoofd
# -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Finale evaluatie onder correcte tournament-condities."
    )
    p.add_argument("--games", type=int, default=40,
                   help="Aantal games per seat per agent (default: 40)")
    p.add_argument("--quick", action="store_true",
                   help="Snelle test: alleen 10 games, 2 seats")
    p.add_argument("--plots-only", action="store_true",
                   help="Genereer alleen grafieken van bestaande CSV")
    p.add_argument("--max-cycles", type=int, default=100000,
                   help="Max cycles per game (default: 100000 = tournament standaard)")
    args = p.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    csv_path = os.path.join(RESULTS, "final_comparison.csv")
    png_path = os.path.join(RESULTS, "final_comparison.png")

    if args.plots_only:
        if not os.path.exists(csv_path):
            print(f"Geen {csv_path} gevonden. Draai eerst zonder --plots-only.")
            return
        rows = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                rows.append({k: (float(v) if k != "agent" else v) for k, v in row.items()})
        plot_results(rows, png_path)
        return

    games = 10 if args.quick else args.games
    seats = (0, 1) if args.quick else (0, 1, 2, 3)

    # Welke agents evalueren we?
    # tuned_agent is jouw bijdrage — als agents/tuned_agent.py bestaat, voeg hem toe
    candidates = [
        ("random",      _make_random),
        ("rule_agent",  _make_rule_agent),
        ("ppo_agent",   _make_ppo),
        ("tuned_agent", _make_tuned),
    ]

    print(f"Finale evaluatie: {games} games/seat, seats={seats}, max_cycles={args.max_cycles}\n")
    print(f"{'Agent':15s} | {'Win':>6} {'Loss':>6} {'Timeout':>8} | {'Survival':>9}")
    print("-" * 58)

    rows = []
    for name, factory in candidates:
        try:
            result = evaluate_agent(
                name=name,
                agent_factory=factory,
                opponent_factory=_make_random,
                games=games,
                seats=seats,
                max_cycles=args.max_cycles,
            )
            rows.append(result)
            print(f"{name:15s} | {result['win_rate']:>6.3f} {result['loss_rate']:>6.3f} "
                  f"{result['timeout_rate']:>8.3f} | {result['mean_survival']:>9.0f}")
        except Exception as e:
            print(f"{name:15s} | FOUT: {e}")

    print()

    # Sla op als CSV
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        print(f"CSV opgeslagen: {csv_path}")
        plot_results(rows, png_path)


if __name__ == "__main__":
    main()
