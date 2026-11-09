"""Stage 6: rally ends -> points.json (one record per point, with rally duration).

Usage: python s6_rallies.py <proxy.mp4> <players.json> <score.json> <review.json> <points.json>
           [--pad 0.25] [--sheets work/contact/rally_ends]

Rally ends are read by eye (review.json "rally_ends": last moment of live play = last hit,
lunge or the shuttle landing). Racket sounds are not used: other courts' hits make rallies look
too long. r0 = serve contact, r1 = rally end + pad, rdur = r1 - r0. The last rally may have no
end (video stops): r1 = rdur = null.

Output: {"points": [{"n", "t", "next_t", "server", "side", "before", "winner", "after",
          "r0", "r1", "rdur", "shuttle"}, ...]}
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from bball.io import load_json, save_json
from bball.video import contact_sheet, read_frame, with_zoom
from s4_serves import far_box


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("players")
    ap.add_argument("score")
    ap.add_argument("review")
    ap.add_argument("out")
    ap.add_argument("--pad", type=float, default=0.25, help="added to the last live moment (s)")
    ap.add_argument("--sheets", default="work/contact/rally_ends")
    ap.add_argument("--blind", action="store_true", help="do not mark the current rally end on the sheets "
                                                         "(for an independent re-read)")
    args = ap.parse_args()

    frames = load_json(args.players)["frames"]
    pts = load_json(args.score)["points"]
    ends = sorted(load_json(args.review)["rally_ends"], key=lambda e: e["t"])
    out, unused = [], list(ends)
    for i, p in enumerate(pts):
        nxt = pts[i + 1]["t"] if i + 1 < len(pts) else None
        mine = [e for e in unused if e["t"] > p["t"] and (nxt is None or e["t"] < nxt)]
        if len(mine) > 1:
            raise SystemExit(f"point {p['n']}: {len(mine)} rally ends between {p['t']} and {nxt}")
        end = mine[0] if mine else None
        if end:
            unused.remove(end)
        r1 = round(end["t"] + args.pad, 2) if end else None
        out.append({"n": p["n"], "t": p["t"], "next_t": nxt, "server": p["server"], "side": p["side"],
                    "before": p["before"], "winner": p["winner"], "after": p["after"],
                    "r0": p["t"], "r1": r1, "rdur": round(r1 - p["t"], 2) if r1 else None,
                    "shuttle": end["shuttle"] if end else None})
    if unused:
        print(f"WARNING: rally ends matched to no point: {[e['t'] for e in unused]}")
    save_json({"pad": args.pad, "points": out}, args.out)

    name = {"R": "you", "O": "opp", None: "?"}
    print(f"{len(out)} points -> {args.out}")
    for p in out:
        dur = f"{p['rdur']:5.2f} s" if p["rdur"] else "    ? "
        print(f"  #{p['n']:2d}  serve {p['r0']:6.1f}  end {p['r1'] or float('nan'):6.2f}  rally {dur}  "
              f"shuttle on {name[p['shuttle']]:3s} side  won by {name[p['winner']]:3s}  {p['before']} -> {p['after']}")
    d = np.array([p["rdur"] for p in out if p["rdur"]])
    print(f"rally length: median {np.median(d):.1f} s, min {d.min():.1f}, max {d.max():.1f}; "
          f"<=5 s: {(d <= 5).sum()}, 5-10 s: {((d > 5) & (d <= 10)).sum()}, >10 s: {(d > 10).sum()}")

    Path(args.sheets).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    for p in out:
        if not p["r1"]:
            continue
        tiles = []
        for t in np.arange(p["r1"] - 1.75, p["r1"] + 1.01, 0.25):
            img = read_frame(cap, t)
            if img is not None:
                mark = "  <- rally end" if abs(t - p["r1"]) < 0.01 and not args.blind else ""
                tiles.append(with_zoom(img, far_box(frames, t), f"#{p['n']} t={t:.2f}s{mark}"))
        cv2.imwrite(str(Path(args.sheets) / f"end_{p['n']:02d}.jpg"),
                    contact_sheet(tiles, cols=2, width=tiles[0].shape[1]))
    print(f"contact sheets around each rally end -> {args.sheets}/end_NN.jpg")


if __name__ == "__main__":
    main()
