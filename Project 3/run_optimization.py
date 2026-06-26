"""run_optimization.py — Tune de VelInterceptor voor alle 4 corners.

Dit is een wrapper om optimize.py te draaien voor alle corners en de beste
parameters automatisch te laden in tuned_agent.py via best_params.json.

Gebruik:
    python run_optimization.py              # alle 4 corners, standaard settings
    python run_optimization.py --corner 1   # alleen corner 1 (resume)
    python run_optimization.py --fast       # snelle test (minder games, minder kandidaten)

Verwachte output:
    best_params.json  — wordt automatisch aangemaakt door optimize.py
    Terminal output per corner met de beste gevonden parameters

Tijdsinschatting:
    Per corner ~20-40 minuten op een normale laptop (met --procs 2-3)
    Alle 4 corners: 1.5-3 uur totaal
    Met --fast: ~5-10 min per corner

Tip: verminder --procs als de laptop oververhit (optimize.py gebruikt standaard
alle CPU-kernen). Aanbevolen: --procs 3
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))


def tune_corner(seat: int, games: int, n_cand: int, refine_games: int, procs: int) -> dict:
    """Tune één corner en geef de beste params terug.

    Args:
        seat: Corner-index (0-3).
        games: Aantal games voor initiële screening.
        n_cand: Aantal kandidaat-configuraties.
        refine_games: Aantal games voor finale refinement.
        procs: Aantal CPU-processen.

    Returns:
        dict met 'params', 'win', 'timeout'.
    """
    from optimize import optimize_corner

    print(f"\n{'='*60}")
    print(f"Corner {seat} tuning: {n_cand} kandidaten, {games} screen-games, "
          f"{refine_games} refine-games, {procs} processen")
    print(f"{'='*60}")

    t0 = time.time()
    results = optimize_corner(
        seat=seat,
        n_cand=n_cand,
        games=games,
        refine_games=refine_games,
        procs=procs,
    )
    elapsed = time.time() - t0

    print(f"\nCorner {seat} — top 5 (params | win | timeout):")
    for params, win, timeout in results[:5]:
        print(f"  {params}  win={win:.3f}  to={timeout:.2f}")

    best_params, best_win, best_timeout = results[0]
    print(f"\n→ Beste: {best_params}  win={best_win:.3f}  timeout={best_timeout:.2f}")
    print(f"  Tijd: {elapsed/60:.1f} minuten")

    return {"params": list(best_params), "win": best_win, "timeout": best_timeout}


def main():
    p = argparse.ArgumentParser(
        description="Tune VelInterceptor voor alle corners van Warlords."
    )
    p.add_argument("--corner", type=int, default=None,
                   help="Tune alleen deze corner (0-3). Standaard: alle 4.")
    p.add_argument("--games", type=int, default=24,
                   help="Screening-games per kandidaat (default: 24)")
    p.add_argument("--n-cand", type=int, default=60,
                   help="Aantal kandidaat-configuraties (default: 60)")
    p.add_argument("--refine-games", type=int, default=60,
                   help="Refinement-games voor de top kandidaten (default: 60)")
    p.add_argument("--procs", type=int, default=3,
                   help="CPU-processen (default: 3, verhoog voorzichtig)")
    p.add_argument("--fast", action="store_true",
                   help="Snelle test: 20 kandidaten, 12 games, 20 refine-games")
    args = p.parse_args()

    if args.fast:
        args.games = 12
        args.n_cand = 20
        args.refine_games = 20
        print("FAST mode: minder nauwkeurig maar sneller (~5-10 min per corner)")

    corners = [args.corner] if args.corner is not None else [0, 1, 2, 3]

    # Laad bestaande params (als we resuming)
    params_path = os.path.join(os.path.dirname(__file__), "best_params.json")
    best = {}
    if os.path.exists(params_path):
        with open(params_path) as f:
            existing = json.load(f)
        # Converteer string-keys naar int
        best = {int(k): v for k, v in existing.items()}
        print(f"Bestaande params geladen voor corners: {sorted(best.keys())}")

    total_t0 = time.time()

    for seat in corners:
        if seat in best and not args.fast:
            print(f"\nCorner {seat} al getuned (win={best[seat].get('win', '?'):.3f}) — overgeslagen")
            print("  Gebruik --corner {seat} om opnieuw te tunen")
            continue

        result = tune_corner(
            seat=seat,
            games=args.games,
            n_cand=args.n_cand,
            refine_games=args.refine_games,
            procs=args.procs,
        )
        best[seat] = result

        # Sla na elke corner op zodat we kunnen resumens
        with open(params_path, "w") as f:
            json.dump({str(k): v for k, v in best.items()}, f, indent=2)
        print(f"\n  best_params.json bijgewerkt met corner {seat}")

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"Optimalisatie klaar! Totale tijd: {total_elapsed/60:.1f} minuten")
    print(f"\nBeste params per corner:")
    for seat in sorted(best.keys()):
        b = best[seat]
        params = b["params"] if isinstance(b, dict) else b
        win = b.get("win", "?") if isinstance(b, dict) else "?"
        print(f"  Corner {seat}: {params}  (win={win:.3f})")
    print(f"\nOpgeslagen in: {params_path}")
    print("TunedAgent laadt deze params automatisch bij de volgende import.")
    print("\nVolgende stap: python evaluate_final.py --games 40")


if __name__ == "__main__":
    main()
