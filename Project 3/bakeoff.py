"""Bake-off: which control law wins most, per corner, vs 3 random opponents.
This grounds the interceptor design in the real win objective."""
import sys
from functools import partial

from tourney import evaluate
from controllers import make_law_agent, random_opp, rule_opp

LAWS = ["sweep", "static", "trackx", "trackxinv", "tracky", "trackyinv",
        "track_xy", "track_xyinv", "track_xinvy", "track_xinvyinv"]


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    opp = random_opp if (len(sys.argv) < 3 or sys.argv[2] == "random") else rule_opp
    print(f"games/seat={games}  opponents={opp.__name__}")
    print(f"{'law':14s} | overall | per-seat (c0 c1 c2 c3)")
    rows = []
    for law in LAWS:
        tf = partial(make_law_agent, law=law)
        overall, per = evaluate(tf, opp, games=games, per_seat=True)
        rows.append((law, overall, per))
        print(f"{law:14s} |  {overall:.3f}  | "
              + " ".join(f"{per[s]:.2f}" for s in (0, 1, 2, 3)))
    rows.sort(key=lambda r: -r[1])
    print("\nranked:", [(r[0], round(r[1], 3)) for r in rows])


if __name__ == "__main__":
    main()
