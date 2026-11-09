"""Stage 5: replay the score from the serves (singles: the winner of a rally serves the next one).

Usage: python s5_score.py <serves.json> <score.json> [--start-score 0 0]

The winner of point N is the server of point N+1; the last point's winner is unknown unless
--last-winner R|O is given. Free consistency check: the server must stand in his RIGHT court
when his score is even, LEFT when odd. Every serve that breaks the rule is flagged.

Output: {"points": [{"n", "t", "server", "side", "expected_side", "side_ok", "before": "you-opp",
          "winner", "after"}, ...], "contradictions": [n, ...]}
"""
import argparse

from bball.io import load_json, save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("serves")
    ap.add_argument("out")
    ap.add_argument("--start-score", type=int, nargs=2, default=(0, 0), metavar=("YOU", "OPP"))
    ap.add_argument("--last-winner", choices=("R", "O"), help="winner of the last point, if known")
    args = ap.parse_args()

    serves = load_json(args.serves)["serves"]
    score = {"R": args.start_score[0], "O": args.start_score[1]}
    points, bad = [], []
    for i, s in enumerate(serves):
        own = score[s["server"]]
        expected = "right" if own % 2 == 0 else "left"
        winner = serves[i + 1]["server"] if i + 1 < len(serves) else args.last_winner
        before = f"{score['R']}-{score['O']}"
        if winner:
            score[winner] += 1
        p = {"n": i + 1, "t": s["t"], "server": s["server"], "side": s["side"], "expected_side": expected,
             "side_ok": s["side"] == expected, "side_margin": s.get("margin"),
             "before": before, "winner": winner, "after": f"{score['R']}-{score['O']}" if winner else None}
        points.append(p)
        if not p["side_ok"]:
            bad.append(p["n"])

    save_json({"start_score": list(args.start_score), "points": points, "contradictions": bad}, args.out)
    name = {"R": "you", "O": "opp", None: "?"}
    print(f"{len(points)} points -> {args.out}   (score: you-opp)")
    for p in points:
        mark = "ok" if p["side_ok"] else f"CONTRADICTION (expected {p['expected_side']})"
        print(f"  #{p['n']:2d}  {p['t']:6.1f}s  {p['before']:>5s}  serve {name[p['server']]:3s} from {p['side']:5s}"
              f" {mark:30s} won by {name[p['winner']]:3s} -> {p['after'] or '?'}")
    print(f"service-court rule: {len(points) - len(bad)}/{len(points)} serves agree, "
          f"{len(bad)} contradictions{': ' + str(bad) if bad else ''}")
    won = sum(p["winner"] == "R" for p in points)
    lost = sum(p["winner"] == "O" for p in points)
    print(f"points won {won}, lost {lost}, unknown {len(points) - won - lost}")


if __name__ == "__main__":
    main()
