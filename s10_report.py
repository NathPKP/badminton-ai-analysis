"""Stage 10: every number in one place, split by points won vs lost and by rally length.

Usage: python s10_report.py <work dir> [--out work/report.json] [--md work/report.md]

Reads players.json, points.json, review.json, motion.json, pose_stats.json, pose_frames.json and
pickups.json from the work dir. The coaching itself (lessons with timestamps) is written by the AI
from these numbers into work/coaching.json; this script only gathers and checks the evidence.
"""
import argparse
from pathlib import Path

import numpy as np

from bball import court
from bball.io import load_json, save_json
from bball.track import positions, rolling_median
from s7_movement import BASE, interp_gaps, moving_mean

ENDING = {(True, "O"): "ton coup gagnant (ou sa faute au filet)", (True, "R"): "sa faute (volant dehors ou service)",
          (False, "R"): "son coup gagnant (ou ta faute au filet)", (False, "O"): "ta frappe sortie"}


def pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def val(d, *keys):
    """d[k1][k2]... or None when any level is missing / None."""
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def bucket(d):
    return "<=5s" if d <= 5 else "5-10s" if d <= 10 else ">10s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work")
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    w = Path(args.work)
    out_json = Path(args.out or w / "report.json")
    out_md = Path(args.md or w / "report.md")

    frames = load_json(w / "players.json")["frames"]
    t = np.array([f["t"] for f in frames])
    P = moving_mean(rolling_median(interp_gaps(positions(frames, "r"), 10), 5), 7)
    pts = load_json(w / "points.json")["points"]
    motion = {m["n"]: m for m in load_json(w / "motion.json")["points"]}
    msum = load_json(w / "motion.json")["summary"]
    pose = load_json(w / "pose_stats.json")
    pframes = load_json(w / "pose_frames.json")["frames"]
    picks = {p["n"]: p for p in load_json(w / "pickups.json")["points"]}

    rows = []
    for p in pts:
        r = {k: p[k] for k in ("n", "t", "server", "side", "before", "after", "winner", "r1", "rdur")}
        if p["rdur"] is None:
            r["note"] = "la vidéo s'arrête pendant l'échange"
            rows.append(r)
            continue
        r["bucket"] = bucket(p["rdur"])
        eye, auto = p.get("shuttle"), picks.get(p["n"], {}).get("shuttle")
        r["shuttle"], r["shuttle_source"] = (eye, "lecture visuelle") if eye else (auto, "ramassage détecté") if auto else (None, None)
        if r["shuttle"] and r["winner"]:
            r["ending"] = ENDING[(r["winner"] == "R", r["shuttle"])]
        k = int(np.argmin(np.abs(t - (p["r1"] - 0.25))))
        if not np.isnan(P[k, 0]):
            r["you_at_end"] = {"x": round(float(P[k, 0]), 2), "depth": round(float(P[k, 1]), 2),
                               "from_base": round(float(np.hypot(P[k, 0] - BASE[0], P[k, 1] - BASE[1])), 2)}
        m = motion.get(p["n"], {}).get("R", {})
        r["you_move"] = {k2: m.get(k2) for k2 in ("distance_m", "avg_speed", "peak_speed", "back_share", "trips")}
        r["you_move"]["trips"] = len(m.get("trips", []))
        rows.append(r)

    done = [r for r in rows if r.get("rdur") is not None and r["winner"]]
    won = [r for r in done if r["winner"] == "R"]

    def wl(sel):
        return {"won": sum(r["winner"] == "R" for r in sel), "lost": sum(r["winner"] == "O" for r in sel)}
    agg = {"points": wl(done),
           "on_your_serve": wl([r for r in done if r["server"] == "R"]),
           "on_his_serve": wl([r for r in done if r["server"] == "O"]),
           "by_length": {b: wl([r for r in done if r["bucket"] == b]) for b in ("<=5s", "5-10s", ">10s")},
           "rally_s": {k: float(np.median(v)) if v else None for k, v in
                       (("median", [r["rdur"] for r in done]), ("won_median", [r["rdur"] for r in won]),
                        ("lost_median", [r["rdur"] for r in done if r["winner"] == "O"]))},
           "endings": {}}
    for r in done:
        e = r.get("ending", "inconnu")
        agg["endings"].setdefault(e, []).append(r["n"])
    end_depth = {k: [r["you_at_end"]["depth"] for r in done if r["winner"] == k and "you_at_end" in r] for k in "RO"}
    agg["you_depth_at_end"] = {"won": end_depth["R"], "lost": end_depth["O"]}
    agg["movement_you"] = {k: msum["R"].get(k) for k in ("all", "you_won", "you_lost", "<=5s", "5-10s", ">10s")}
    agg["pose_you"] = pose["stats"]

    # example moments for the lessons: back-court frames with the racket down / up, by result
    winner = {p["n"]: p["winner"] for p in pts}
    rally = {}
    for p in pts:
        if p["r1"]:
            for k in np.where((t >= p["r0"] + 0.3) & (t <= p["r1"]))[0]:
                rally[round(float(t[k]), 2)] = p["n"]
    examples = {"rear_racket_down_lost": [], "rear_racket_up_won": []}
    for f in pframes:
        n = rally.get(round(f["t"], 2))
        if not n or f.get("wrist_up") is None or f["trunc"]:
            continue
        k = int(np.argmin(np.abs(t - f["t"])))
        if P[k, 1] > 1.6:
            continue
        if winner[n] == "O" and f["wrist_up"] <= 0:
            examples["rear_racket_down_lost"].append((n, f["t"]))
        if winner[n] == "R" and f["wrist_up"] > 0:
            examples["rear_racket_up_won"].append((n, f["t"]))
    agg["examples"] = examples

    meta = load_json(w / "analysis.json") if (w / "analysis.json").exists() else {}
    limits = []
    if meta.get("params", {}).get("camera", "sol") == "sol":
        limits.append("profondeur de l'adversaire biaisée (caméra au sol) : ses zones ne sont pas utilisées")
    if len(done) < 40:
        limits.append(f"{len(done)} échanges seulement : des tendances, pas des certitudes")
    guessed = [r["n"] for r in done if r.get("shuttle_source") == "ramassage détecté"]
    if guessed:
        limits.append("côté du volant déduit du ramassage (pas vu directement) pour le(s) point(s) "
                      + ", ".join(map(str, guessed)))
    save_json({"points": rows, "summary": agg, "limits": limits}, out_json)

    # ---- markdown, in French for the player
    L = ["# Analyse du match", "",
         f"**{agg['points']['won']} points gagnés, {agg['points']['lost']} perdus** sur {len(done)} échanges terminés "
         f"(score {rows[0]['before']} → {done[-1]['after']}).", "",
         "| # | Service | Score | Durée | Gagné par | Volant tombé | Fin du point |", "|---|---|---|---|---|---|---|"]
    who = {"R": "toi", "O": "adversaire", None: "?"}
    for r in rows:
        dur = f"{r['rdur']:.1f} s" if r.get("rdur") else "—"
        L.append(f"| {r['n']} | {r['t']:.1f} s, {who[r['server']]} | {r['before']} → {r['after'] or '?'} | {dur} | "
                 f"{who[r['winner']]} | {who[r.get('shuttle')] if r.get('shuttle') else '?'} | {r.get('ending', r.get('note', '?'))} |")
    s, mv = agg, agg["movement_you"]
    bl = s["by_length"]
    na = lambda x: "n/a" if x is None else x
    L += ["", "## Ce qui ressort", "",
          f"- Sur ton service : {s['on_your_serve']['won']} gagné(s), {s['on_your_serve']['lost']} perdu(s). "
          f"Sur le sien : {s['on_his_serve']['won']} gagné(s), {s['on_his_serve']['lost']} perdu(s).",
          f"- Échanges de 5 s ou moins : {bl['<=5s']['won']}-{bl['<=5s']['lost']}. "
          f"De 5 à 10 s : {bl['5-10s']['won']}-{bl['5-10s']['lost']}. Plus de 10 s : {bl['>10s']['won']}-{bl['>10s']['lost']}.",
          f"- Vitesse moyenne en jeu : {na(val(mv, 'all', 'avg_speed'))} m/s ; 1 s après une sortie de position, "
          f"tu es encore à {na(val(mv, 'you_won', 'median_dist_1s_after_trip'))} m de ta base sur les points gagnés, "
          f"{na(val(mv, 'you_lost', 'median_dist_1s_after_trip'))} m sur les points perdus.",
          f"- Raquette haute au fond du court : {pct(val(s['pose_you'], 'REAR', 'won', 'racket_up'))} du temps sur les points gagnés, "
          f"{pct(val(s['pose_you'], 'REAR', 'lost', 'racket_up'))} sur les points perdus.",
          "", "## Limites", ""] + [f"- {x}" for x in limits]
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"-> {out_json}, {out_md}")
    print(f"points {agg['points']}, your serve {agg['on_your_serve']}, his serve {agg['on_his_serve']}")
    print(f"by length {agg['by_length']}, rally medians {agg['rally_s']}")
    print(f"endings {agg['endings']}")
    print(f"your depth at rally end: won {end_depth['R']}  lost {end_depth['O']}")
    for r in done:
        print(f"  #{r['n']:2d} {r['t']:6.1f}s srv {r['server']} {r['winner']} {r['rdur']:4.1f}s  end at {r.get('you_at_end')}  "
              f"move {r['you_move']}  {r.get('ending', '')}")
    print(f"examples rear racket down (lost): {examples['rear_racket_down_lost'][:12]}")
    print(f"examples rear racket up (won): {examples['rear_racket_up_won'][:12]}")


if __name__ == "__main__":
    main()
