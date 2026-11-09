"""Reference numbers: movement per rally across several analyses (e.g. pro TV matches vs your own).

Usage: python s12_benchmark.py <analysis> [<analysis> ...] [--out benchmark.json] [--label pro]

Per rally and per player: duration, distance, average and peak speed, back / front share, trips out of
position and the distance to the base 1 s later, computed exactly as stage 7 does (same smoothing,
same s7_movement.rally_stats), so the numbers compare directly.
Rallies come from points.json (rallies read by eye) when the analysis has one; for TV footage
(footage=tv) they are found automatically: the main court view is on screen from the serve preparation
until shortly after the rally ends, so each view run of 4 s or more holds one rally, from the end of the
still serve preparation (the faster player above 1 m/s for 1 s) to the last fast movement (above 1.2 m/s)
+ 0.3 s. A rally whose start or end is cut by the camera is kept for movement but left out of durations.
Which players count: both players on TV footage (camera high up); only you ("R") on your own footage
filmed from the floor (camera=sol), where the far player's depth is biased.
"""
import argparse
from pathlib import Path

import numpy as np

from bball.io import load_json, save_json
from bball.track import positions, rolling_median
from s7_movement import interp_gaps, moving_mean, own_half, rally_stats


def tv_rallies(t, view, speed, min_run=4.0):
    """[(t0, t1, start_cut, end_cut)] from runs of the main view and the faster player's speed."""
    runs, start = [], None
    for i, v in enumerate(np.r_[view, False]):
        if v and start is None:
            start = i
        if not v and start is not None:
            if runs and t[start] - t[runs[-1][1]] <= 0.5:          # short glitches in the view check
                runs[-1] = (runs[-1][0], i - 1)
            else:
                runs.append((start, i - 1))
            start = None
    out = []
    for a, b in runs:
        if t[b] - t[a] < min_run:
            continue
        s = np.nan_to_num(speed[a:b + 1])
        moving = np.convolve(s >= 1.0, np.ones(10), "valid") >= 8          # 1 s mostly above 1 m/s ...
        moving &= s[:len(moving)] >= 1.0                                   # ... starting with this frame
        if not moving.any():
            continue
        i0 = int(np.argmax(moving))
        fast = np.where(s >= 1.2)[0]
        i1 = min(int(fast[-1]) + 3, len(s) - 1)
        if t[a + i1] - t[a + i0] < 1.5:
            continue
        out.append((float(t[a + i0]), float(t[a + i1]), i0 < 5, i1 >= len(s) - 4))
    return out


def analyse(folder):
    folder = Path(folder)
    meta = load_json(folder / "analysis.json")
    params = meta.get("params", {})
    data = load_json(folder / "players.json")
    frames, fps = data["frames"], data["fps"]
    t = np.array([f["t"] for f in frames])
    tv = params.get("footage") == "tv"
    raised = params.get("camera") == "haute" or tv
    P = {"r": moving_mean(rolling_median(interp_gaps(positions(frames, "r"), int(fps)), 5), 7),
         "o": moving_mean(rolling_median(interp_gaps(positions(frames, "o"), int(fps)), 5 if raised else 9), 7)}
    if tv:
        view = np.array([f.get("v", False) for f in frames])
        sp = {}
        for k, Q in P.items():
            s = np.full(len(Q), np.nan)
            s[2:-2] = np.hypot(*(Q[4:] - Q[:-4]).T) / (4 / fps)
            sp[k] = s
        rallies = tv_rallies(t, view, np.nanmax(np.c_[sp["r"], sp["o"]], 1))
    else:
        rallies = [(p["r0"] + 0.3, p["r1"], False, False) for p in load_json(folder / "points.json")["points"] if p["r1"]]
    keys = ("r", "o") if raised else ("r",)
    rows = []
    for t0, t1, cut0, cut1 in rallies:
        for k in keys:
            st = rally_stats(own_half(P[k], k), t, fps, t0, t1)
            if st.get("coverage", 0) < 0.6 or st.get("avg_speed") is None:
                continue
            rows.append({"t0": round(t0, 2), "t1": round(t1, 2), "dur": round(t1 - t0, 2), "cut": cut0 or cut1,
                         "player": k, **{x: st[x] for x in ("distance_m", "avg_speed", "peak_speed", "back_share",
                                                            "front_share", "live_s")},
                         "trips": len(st["trips"]),
                         "after_1s": [tr["after_1s_m"] for tr in st["trips"] if tr["after_1s_m"] is not None]})
    return {"analysis": folder.name, "footage": "tv" if tv else "own", "rallies": len(rallies), "rows": rows}


def summary(rows):
    def med(x):
        x = [v for v in x if v is not None]
        return round(float(np.median(x)), 2) if x else None

    def iqr(x):
        return [round(float(v), 2) for v in np.percentile(x, [25, 75])] if len(x) >= 4 else None
    durs = sorted({(r["t0"], r["dur"]) for r in rows if not r["cut"]})
    live = sum(r["live_s"] for r in rows)
    return {"rallies": len({r["t0"] for r in rows}), "player_rallies": len(rows),
            "rally_s": {"median": med([d for _, d in durs]), "iqr": iqr([d for _, d in durs]), "n": len(durs)},
            "avg_speed": {"median": med([r["avg_speed"] for r in rows]), "iqr": iqr([r["avg_speed"] for r in rows])},
            "peak_speed": {"median": med([r["peak_speed"] for r in rows])},
            "distance_per_10s": round(10 * sum(r["distance_m"] for r in rows) / live, 1) if live else None,
            "back_share": med([r["back_share"] for r in rows]),
            "front_share": med([r["front_share"] for r in rows]),
            "trips_per_10s": round(10 * sum(r["trips"] for r in rows) / live, 2) if live else None,
            "after_1s_m": med([x for r in rows for x in r["after_1s"]])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analyses", nargs="+")
    ap.add_argument("--out", default="benchmark.json")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent / "analyses"
    res = []
    for a in args.analyses:
        folder = Path(a) if Path(a).is_dir() else next(p for p in root.iterdir() if a in p.name)
        r = analyse(folder)
        r["summary"] = summary(r["rows"])
        res.append(r)
        s = r["summary"]
        print(f"{r['analysis']:34s} {r['footage']:4s} rallies {s['rallies']:3d} (timed {s['rally_s']['n']:2d}, median "
              f"{s['rally_s']['median']} s)  speed {s['avg_speed']['median']} m/s  peak {s['peak_speed']['median']}  "
              f"{s['distance_per_10s']} m/10 s  back {s['back_share']}  trips {s['trips_per_10s']}/10 s  "
              f"1 s after a trip {s['after_1s_m']} m")
    groups = {}
    for r in res:
        groups.setdefault(r["footage"], []).extend(r["rows"])
    save_json({"analyses": [{k: v for k, v in r.items() if k != "rows"} for r in res],
               "groups": {g: summary(rows) for g, rows in groups.items()}}, args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
