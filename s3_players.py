"""Stage 3: identify both players and map their feet to court metres.

Usage: python s3_players.py <proxy.mp4> <dets.json> <calib.json> <players.json>
           [--start 12] [--sheet work/contact/s3_players.jpg] [--plot work/figures/s3_tracks.png]

Near player (you, "r") = the near-half person whose shorts band (48-62 % of the box height)
is blue; nobody blue enough means a gap, never a guess. Far player ("o") = the most
confident detection inside the far half.

Output: {"fps": 10.0, "frames": [{"t", "r": [x, y] | null, "rb": box | null, "rt": feet out of frame,
         "rf": filled by gap fill, "o": [x, y] | null, "ob": box | null, "of": filled}, ...]}
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from bball import court
from bball.io import load_json, save_json
from bball.video import contact_sheet, label

NEAR_Y = (-1.2, 6.9)
FAR_Y = (6.7, 15.5)
X_MARGIN = (-1.0, court.WIDTH + 1.0)
TRUNC_Y = 0.985          # box bottom below this fraction of the frame = feet out of frame


def shorts_blue(frame, box, hue=(100, 135)):
    """Fraction of shorts-coloured pixels (OpenCV hue lo-hi, 0-179; lo > hi wraps through red) in the
    band at 48-62 % of the box height (central half). Default = blue."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    bh, bw = y2 - y1, x2 - x1
    band = frame[int(y1 + 0.48 * bh):int(y1 + 0.62 * bh), int(x1 + 0.25 * bw):int(x2 - 0.25 * bw)]
    if band.size == 0:
        return 0.0
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    lo, hi = hue
    in_hue = (hsv[:, 0] >= lo) & (hsv[:, 0] <= hi) if lo <= hi else (hsv[:, 0] >= lo) | (hsv[:, 0] <= hi)
    blue = in_hue & (hsv[:, 1] >= 40) & (hsv[:, 2] >= 25)
    return float(blue.mean())


def candidates(people, H, size):
    """Each detection -> dict with box, feet pixel, court position and flags."""
    out = []
    for p in people:
        fx, fy = (p[0] + p[2]) / 2 * size[0], p[3] * size[1]
        x, y = court.to_court(H, [[fx, fy]])[0]
        out.append({"box": p[:4], "score": p[4], "pos": (float(x), float(y)),
                    "trunc": p[3] > TRUNC_Y,
                    "in_x": X_MARGIN[0] <= x <= X_MARGIN[1]})
    return out


def in_half(c, half):
    lo, hi = half
    return c["in_x"] and lo <= c["pos"][1] <= hi


def gap_fill(track, cands, key, half, max_dist=0.8, window=5, passes=3):
    """Fill missing frames with the in-half person within max_dist of a known position nearby."""
    n = len(track)
    for _ in range(passes):
        known = [t[key] for t in track]
        for i in range(n):
            if track[i][key] is not None or track[i].get(key + "t"):
                continue
            refs = [known[j] for j in range(max(0, i - window), min(n, i + window + 1))
                    if j != i and known[j] is not None]
            best, best_d = None, max_dist
            for c in cands[i]:
                if not in_half(c, half) or c["trunc"]:
                    continue
                d = min(np.hypot(c["pos"][0] - r[0], c["pos"][1] - r[1]) for r in refs) if refs else 1e9
                if d < best_d:
                    best, best_d = c, d
            if best is not None:
                track[i][key] = best["pos"]
                track[i][key + "b"] = best["box"]
                track[i][key + "f"] = True   # filled
    return track


def despike(track, key, thr):
    """Drop single frames > thr m from both neighbours when the neighbours agree with each other."""
    dropped = 0
    for i in range(1, len(track) - 1):
        a, b, c = track[i - 1][key], track[i][key], track[i + 1][key]
        if a is None or b is None or c is None:
            continue
        dab, dbc, dac = (np.hypot(b[0] - a[0], b[1] - a[1]), np.hypot(c[0] - b[0], c[1] - b[1]),
                         np.hypot(c[0] - a[0], c[1] - a[1]))
        if dab > thr and dbc > thr and dac < thr:
            track[i][key], track[i][key + "b"] = None, None
            dropped += 1
    return dropped


def draw_tile(frame, rec, H):
    img = court.draw_court(frame.copy(), H, (0, 200, 200), 1)
    h, w = img.shape[:2]
    for key, col in (("r", (0, 255, 0)), ("o", (0, 140, 255))):
        b = rec.get(key + "b")
        if b is None:
            continue
        cv2.rectangle(img, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)), col, 1)
        fx, fy = int((b[0] + b[2]) / 2 * w), int(b[3] * h)
        cv2.circle(img, (fx, min(fy, h - 4)), 6, col, 2)
        pos = rec.get(key)
        txt = "feet out of frame" if rec.get(key + "t") else f"{pos[0]:.1f},{pos[1]:.1f}" if pos else "?"
        label(img, f"{'YOU' if key == 'r' else 'OPP'} {txt}", (int(b[0] * w), max(int(b[1] * h) - 6, 14)), 0.45)
    mm = court.minimap([(rec["r"] or (None, None), (0, 255, 0)), (rec["o"] or (None, None), (0, 140, 255))])
    img[8:8 + mm.shape[0], w - 8 - mm.shape[1]:w - 8] = mm
    label(img, f"t={rec['t']:.1f}s")
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("dets")
    ap.add_argument("calib")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0.0, help="ignore everything before this time (warm-up)")
    ap.add_argument("--min-blue", type=float, default=0.35, help="min share of shorts-coloured pixels in the band")
    ap.add_argument("--shorts-hue", type=int, nargs=2, default=(100, 135), metavar=("LO", "HI"),
                    help="your shorts' OpenCV hue range (0-179): blue 100 135, red 170 10, green 40 85")
    ap.add_argument("--sheet", default="work/contact/s3_players.jpg")
    ap.add_argument("--plot", default="work/figures/s3_tracks.png")
    args = ap.parse_args()

    dets = load_json(args.dets)
    H_at = court.load_calib(args.calib)
    cap = cv2.VideoCapture(args.video)
    vfps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames = [f for f in dets["frames"] if f["t"] >= args.start]
    wanted = {int(round(f["t"] * vfps)): k for k, f in enumerate(frames)}

    track, cands = [], []
    sheet_k = set(np.linspace(0, len(frames) - 1, 16).astype(int))
    sheet_frames = {}
    i = 0
    while wanted:
        if not cap.grab():
            break
        k = wanted.pop(i, None)
        i += 1
        if k is None:
            continue
        _, frame = cap.retrieve()
        f = frames[k]
        H = H_at(f["t"])
        rec = {"t": f["t"], "r": None, "rb": None, "rt": False, "o": None, "ob": None}
        cs = candidates(f["p"], H, size) if H is not None else []
        cands.append(cs)
        if k in sheet_k:
            sheet_frames[k] = frame
        near = [c for c in cs if in_half(c, NEAR_Y) or c["trunc"] and c["in_x"]]
        for c in near:
            c["blue"] = shorts_blue(frame, c["box"], args.shorts_hue)
        blue = [c for c in near if c["blue"] >= args.min_blue]
        if blue:
            me = max(blue, key=lambda c: c["blue"])
            rec["rb"], rec["rt"] = me["box"], me["trunc"]
            rec["r"] = None if me["trunc"] else me["pos"]
        far = [c for c in cs if in_half(c, FAR_Y) and not c["trunc"] and c["box"] is not rec["rb"]]
        if far:
            opp = max(far, key=lambda c: c["score"])
            rec["o"], rec["ob"] = opp["pos"], opp["box"]
        track.append(rec)

    n = len(track)
    direct_r = sum(t["r"] is not None for t in track)
    gap_fill(track, cands, "r", NEAR_Y)
    gap_fill(track, cands, "o", FAR_Y)
    # far depth is ~5x less precise than near depth (low camera), so it gets a wider spike threshold
    sp_r, sp_o = despike(track, "r", 0.9), despike(track, "o", 1.5)

    out = []
    for t in track:
        rec = {"t": t["t"],
               "r": [round(v, 3) for v in t["r"]] if t["r"] else None, "rb": t["rb"], "rt": bool(t["rt"]),
               "rf": bool(t.get("rf")),
               "o": [round(v, 3) for v in t["o"]] if t["o"] else None, "ob": t["ob"],
               "of": bool(t.get("of"))}
        out.append(rec)
    save_json({"fps": dets["fps"], "start": args.start, "frames": out}, args.out)

    has_r = sum(t["r"] is not None for t in out)
    trunc = sum(t["rt"] for t in out)
    has_o = sum(t["o"] is not None for t in out)
    print(f"{n} frames from t={args.start:.1f}s -> {args.out}")
    print(f"you      : {has_r / n:6.1%} placed ({direct_r} by shorts colour, {has_r - direct_r} by gap fill), "
          f"{trunc / n:.1%} feet out of frame, {sp_r} spikes dropped")
    print(f"opponent : {has_o / n:6.1%} placed, {sp_o} spikes dropped")
    for key, name in (("r", "you"), ("o", "opponent")):
        p = np.array([t[key] for t in out if t[key]])
        print(f"  {name:8s} x {p[:, 0].min():5.2f}..{p[:, 0].max():5.2f}  y {p[:, 1].min():5.2f}..{p[:, 1].max():5.2f}")
        steps = [np.hypot(b[key][0] - a[key][0], b[key][1] - a[key][1]) * dets["fps"]
                 for a, b in zip(out, out[1:]) if a[key] and b[key]]
        print(f"  {name:8s} raw frame-to-frame speed m/s: median {np.median(steps):.2f}, "
              f"p90 {np.percentile(steps, 90):.2f}, max {max(steps):.1f}")

    tiles = []
    for k in sorted(sheet_frames):
        tiles.append(draw_tile(sheet_frames[k], out[k], H_at(out[k]["t"])))
    Path(args.sheet).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.sheet, contact_sheet(tiles, cols=4, width=600))
    print(f"contact sheet -> {args.sheet}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    tt = np.array([t["t"] for t in out])
    for ax, j, name in ((axes[0], 1, "y along court (m)"), (axes[1], 0, "x across (m)")):
        for key, col, lab in (("r", "tab:green", "you"), ("o", "tab:orange", "opponent")):
            v = np.array([t[key][j] if t[key] else np.nan for t in out])
            ax.plot(tt, v, ".", ms=2, color=col, label=lab)
        if j == 1:
            ax.axhline(court.NET_Y, color="k", lw=0.8)
            tr = np.array([t["rt"] for t in out])
            ax.plot(tt[tr], np.zeros(tr.sum()) - 0.5, "|", color="red", label="your feet out of frame")
        ax.set_ylabel(name)
        ax.legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("time (s)")
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=110)
    print(f"track plot -> {args.plot}")


if __name__ == "__main__":
    main()
