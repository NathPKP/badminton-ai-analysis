"""Stage 10: every number in one place, split by points won vs lost and by rally length.

Usage: python s10_report.py <analysis dir> [--lang en|fr] [--out report.json] [--md report.md]

Reads players.json, points.json, review.json, motion.json, pose_stats.json, pose_frames.json and
pickups.json from the analysis dir. The coaching itself (lessons with timestamps) is written from
these numbers into coaching.json, by an AI assistant or by s10_coach.py; this script only gathers
and checks the evidence. Texts (report.md, endings, limits) follow --lang (bball/i18n.py).
"""
import argparse
from pathlib import Path

import numpy as np

from bball import court
from bball.io import load_json, save_json
from bball.track import positions, rolling_median
from bball.i18n import Text, analysis_lang
from s7_movement import BASE, interp_gaps, moving_mean

# (you won?, side where the shuttle came down) -> how the point ended
ENDING = {(True, "O"): "my_winner", (True, "R"): "his_fault", (False, "R"): "his_winner", (False, "O"): "my_out"}


T = Text("en")


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
    ap.add_argument("--lang", choices=("en", "fr"), help="default: the analysis' language")
    args = ap.parse_args()
    w = Path(args.work)
    global T
    T = Text(args.lang or analysis_lang(w))
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
            r["note"] = T("note_unfinished")
            rows.append(r)
            continue
        r["bucket"] = bucket(p["rdur"])
        eye, auto = p.get("shuttle"), picks.get(p["n"], {}).get("shuttle")
        r["shuttle"], r["shuttle_source"] = (eye, "eye") if eye else (auto, "pickup") if auto else (None, None)
        if r["shuttle"] and r["winner"]:
            r["ending_code"] = ENDING[(r["winner"] == "R", r["shuttle"])]
            r["ending"] = T("end_second")[r["ending_code"]]
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
        limits.append(T("limit_depth"))
    if len(done) < 40:
        limits.append(T("limit_few", n=len(done)))
    guessed = [r["n"] for r in done if r.get("shuttle_source") == "pickup"]
    if guessed:
        limits.append(T("limit_guessed", list=", ".join(map(str, guessed))))
    save_json({"lang": T.lang, "points": rows, "summary": agg, "limits": limits}, out_json)

    # ---- markdown, for the player
    L = [T("md_title"), "",
         T("md_points", won=agg["points"]["won"], lost=agg["points"]["lost"], n=len(done), a=rows[0]["before"],
           b=done[-1]["after"]), "",
         T("md_head"), "|---|---|---|---|---|---|---|"]
    who = T("md_who")
    for r in rows:
        dur = f"{r['rdur']:.1f} s" if r.get("rdur") else "—"
        L.append(f"| {r['n']} | {r['t']:.1f} s, {who[r['server']]} | {r['before']} → {r['after'] or '?'} | {dur} | "
                 f"{who[r['winner']]} | {who[r.get('shuttle')] if r.get('shuttle') else '?'} | {r.get('ending', r.get('note', '?'))} |")
    s, mv = agg, agg["movement_you"]
    bl = s["by_length"]
    na = lambda x: "n/a" if x is None else x
    wl = lambda d: f"{d['won']}-{d['lost']}"
    L += ["", T("md_findings"), "",
          T("md_serve", w1=s["on_your_serve"]["won"], l1=s["on_your_serve"]["lost"], w2=s["on_his_serve"]["won"],
            l2=s["on_his_serve"]["lost"]),
          T("md_length", a=wl(bl["<=5s"]), b=wl(bl["5-10s"]), c=wl(bl[">10s"])),
          T("md_speed", v=na(val(mv, "all", "avg_speed")), w=na(val(mv, "you_won", "median_dist_1s_after_trip")),
            l=na(val(mv, "you_lost", "median_dist_1s_after_trip"))),
          T("md_racket", w=pct(val(s["pose_you"], "REAR", "won", "racket_up")),
            l=pct(val(s["pose_you"], "REAR", "lost", "racket_up"))),
          "", T("md_limits"), ""] + [f"- {x}" for x in limits]
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
