"""Stage 9: who bends to pick up the shuttle after each rally -> which side it came down on.

Usage: python s9_pickups.py <proxy.mp4> <players.json> <points.json> <pickups.json>
           [--review work/review.json] [--sheet work/contact/s9_pickups.jpg]

Standing height model per player: an upright person's box height grows linearly with how far
below the horizon his feet are in the image (perspective), h = a * feet_row + b, fitted on his
still, untruncated frames (in pixels, so the far player's depth bias does not matter). A box
under 82 % of that height (72 % in the original write-up, filmed from higher) while moving
slowly, for at least 0.3 s, is a bend. Re-check the threshold with --review on new footage. The first bend
between the rally end and the next serve says on which side the shuttle came down.

What it means (from your side): won + shuttle on their side = your winner (or their shot into
the net); won + your side = their shot went out (or a serve fault); lost + your side = their
winner (or your shot into the net); lost + their side = your shot went out.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from bball.io import load_json, save_json
from bball.track import positions, rolling_median, speed
from bball.video import contact_sheet, read_frame, with_zoom
from s4_serves import far_box

TRUNC_Y = 0.985


def height_model(frames, key, v, size):
    """Robust linear fit of upright box height (px) vs feet row (px)."""
    w, h = size
    rows, hs = [], []
    for f, s in zip(frames, v):
        b = f[key + "b"]
        if b is None or b[3] > TRUNC_Y or not (s < 0.5):
            continue
        rows.append(b[3] * h)
        hs.append((b[3] - b[1]) * h)
    rows, hs = np.array(rows), np.array(hs)
    m = np.ones(len(rows), bool)
    for _ in range(4):
        a, c = np.polyfit(rows[m], hs[m], 1)
        res = hs - (a * rows + c)
        # bends and partial boxes only make boxes shorter: keep the upper part of the cloud
        m = res > -1.5 * np.std(res[m])
    return a, c, int(m.sum())


def bend_runs(ratio, slow, thr, min_len):
    """[(start, end)] runs where ratio < thr and the player is slow, at least min_len frames."""
    flag = (ratio < thr) & slow
    out, start = [], None
    for i, f in enumerate(np.append(flag, False)):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("players")
    ap.add_argument("points")
    ap.add_argument("out")
    # 0.72 in the original write-up (raised camera); with a camera on the floor a bend only shrinks the
    # box to ~76-80 %. 0.78-0.85 all agreed with the eye reading (6/6-7/7); 0.82 covered the most rallies.
    ap.add_argument("--thr", type=float, default=0.82, help="bend = box under this share of the standing height")
    ap.add_argument("--slow", type=float, default=1.5,
                    help="m/s; bending players move slowly, but the box jump while bending inflates the speed")
    ap.add_argument("--min-bend", type=float, default=0.3, help="s")
    ap.add_argument("--window", type=float, default=8.0, help="look this long after the rally end (s)")
    ap.add_argument("--review", help="review.json: compare with the shuttle sides read by eye")
    ap.add_argument("--sheet", default="work/contact/s9_pickups.jpg")
    args = ap.parse_args()

    data = load_json(args.players)
    frames, fps = data["frames"], data["fps"]
    t = np.array([f["t"] for f in frames])
    cap = cv2.VideoCapture(args.video)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    pts = load_json(args.points)["points"]

    ratio, slow, model = {}, {}, {}
    for key, win in (("r", 5), ("o", 9)):
        v = speed(rolling_median(positions(frames, key), win), fps)
        a, c, n = height_model(frames, key, v, size)
        model[key] = {"a": round(float(a), 4), "b": round(float(c), 2), "frames": n}
        r = np.full(len(frames), np.nan)
        for i, f in enumerate(frames):
            b = f[key + "b"]
            if b is not None and b[3] <= TRUNC_Y:
                r[i] = (b[3] - b[1]) * size[1] / (a * b[3] * size[1] + c)
        ratio[key] = r
        # speed is NaN while bending more often than not (box jumps): treat NaN as slow
        slow[key] = ~(v > args.slow)
        print(f"standing height model {'you' if key == 'r' else 'opp'}: h = {a:.3f} * feet_row + {c:.1f} "
              f"({n} frames)")

    min_len = max(1, int(round(args.min_bend * fps)))
    runs = {k: bend_runs(ratio[k], slow[k], args.thr, min_len) for k in ("r", "o")}
    review = {}
    if args.review:
        for e in load_json(args.review)["rally_ends"]:
            review[round(e["t"], 2)] = e["shuttle"]

    out, tiles = [], []
    name = {"r": "R", "o": "O"}
    for i, p in enumerate(pts):
        if not p["r1"]:
            continue
        end = min(p["r1"] + args.window, p["next_t"] or 1e9)
        bends = []
        for k in ("r", "o"):
            for a, b in runs[k]:
                if p["r1"] - 0.25 <= t[a] <= end:
                    bends.append({"who": name[k], "t": round(float(t[a]), 2), "dur": round(float(t[b] - t[a]) + 1 / fps, 2),
                                  "min_ratio": round(float(np.nanmin(ratio[k][a:b + 1])), 2),
                                  "pos": frames[a][k]})
        bends.sort(key=lambda x: x["t"])
        side = bends[0]["who"] if bends else None
        rv = review.get(round(p["r1"] - 0.25, 2), "n/a") if review else "n/a"
        rec = {"n": p["n"], "r1": p["r1"], "winner": p["winner"], "bends": bends, "shuttle": side,
               "review_shuttle": rv}
        if side and p["winner"]:
            you_won = p["winner"] == "R"
            rec["meaning"] = {(True, "O"): "your winner (or their shot into the net)",
                              (True, "R"): "their shot went out (or a serve fault)",
                              (False, "R"): "their winner (or your shot into the net)",
                              (False, "O"): "your shot went out"}[(you_won, side)]
        out.append(rec)
        if bends:
            b0 = bends[0]
            img = read_frame(cap, b0["t"])
            k = int(np.argmin(np.abs(t - b0["t"])))
            box = frames[k]["rb" if b0["who"] == "R" else "ob"]
            hh, ww = img.shape[:2]
            cv2.rectangle(img, (int(box[0] * ww), int(box[1] * hh)), (int(box[2] * ww), int(box[3] * hh)),
                          (0, 0, 255), 2)
            tiles.append(with_zoom(img, far_box(frames, b0["t"]),
                                   f"#{p['n']} t={b0['t']:.1f} {'YOU' if b0['who'] == 'R' else 'OPP'} bends "
                                   f"({b0['min_ratio']:.0%} of standing)"))

    agree = [o for o in out if o["review_shuttle"] in ("R", "O") and o["shuttle"]]
    ok = sum(o["shuttle"] == o["review_shuttle"] for o in agree)
    save_json({"model": model, "thr": args.thr, "points": out,
               "vs_review": {"compared": len(agree), "agree": ok}}, args.out)
    print(f"{len(out)} rallies -> {args.out}")
    for o in out:
        bs = ", ".join(f"{b['who']}@{b['t']:.1f}({b['min_ratio']:.0%})" for b in o["bends"][:3]) or "none"
        print(f"  #{o['n']:2d} end {o['r1']:6.2f}  bends: {bs:40s} -> shuttle {o['shuttle'] or '?'} "
              f"(eye: {o['review_shuttle'] or '?'})  {o.get('meaning', '')}")
    if args.review:
        print(f"agreement with the eye reading: {ok}/{len(agree)} rallies where both have a side")
    if tiles:
        Path(args.sheet).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.sheet, contact_sheet(tiles, cols=2, width=tiles[0].shape[1]))
        print(f"contact sheet (first bend after each rally) -> {args.sheet}")


if __name__ == "__main__":
    main()
