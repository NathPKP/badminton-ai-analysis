"""Stage 4: serve detection = both players still in their service areas for >= 0.8 s.

Usage: python s4_serves.py <proxy.mp4> <players.json> <serves.json> [--review work/review.json]
           [--sheets work/contact/serves]

Side, from the two x positions (server and receiver stand diagonally):
  near player on the camera's right + far player on the camera's left = both in their RIGHT
  service courts -> "right" (server's score even); the mirror -> "left" (odd).
  The far player faces the camera, so his right court is on the camera's left.
Who serves cannot be read from positions alone. Without --review, "server" is a guess (the
receiver is usually the one who moves first). With --review, the serves read by eye on the
timeline sheets (timeline_sheets.py) are used, and the automatic detection is scored against them.

Output: {"serves": [{"n", "t", "server", "side", "diag_ok", "margin", "checked", "r": [x, y],
          "o": [x, y]}, ...], "auto": [automatic candidates]}
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from bball import court
from bball.io import load_json, save_json
from bball.track import positions, rolling_median, speed
from bball.video import contact_sheet, read_frame, with_zoom


def runs(mask):
    """[(start, end_inclusive), ...] of consecutive True values."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def side_of(r, o, server=None):
    """Service side ("right" = server's score even) from the two positions.

    Each player's own court: near player's right = camera right (x > 3.05); the far player faces
    the camera, so his right = camera left. The server often stands on the centre line, the
    receiver clearly inside his court, so the receiver decides (or, server unknown, whoever is
    further from the centre line). Returns (side, diagonal_ok, margin_m).
    """
    cx = court.CENTER_X
    r_right = bool(r[0] > cx) if not np.isnan(r[0]) else None
    o_right = bool(o[0] < cx) if not np.isnan(o[0]) else None
    dr = abs(r[0] - cx) if r_right is not None else -1.0
    do = abs(o[0] - cx) if o_right is not None else -1.0
    if server == "R":
        use_o = do >= 0.15 or dr < 0
    elif server == "O":
        use_o = dr < 0.15 and do >= 0
    else:
        use_o = do > dr
    right, margin = (o_right, do) if use_o else (r_right, dr)
    return ("right" if right else "left"), bool(r_right == o_right), round(float(margin), 2)


def far_box(frames, t):
    """Far player's box at the frame nearest t, or the closest known one within 1 s."""
    k = min(range(len(frames)), key=lambda i: abs(frames[i]["t"] - t))
    for d in range(0, 11):
        for j in (k - d, k + d):
            if 0 <= j < len(frames) and frames[j]["ob"]:
                return frames[j]["ob"]
    return None


def serve_closeup(cap, frames, t0, t1, step, out_path, cols=6):
    """Fine-grained sheet: for each time, a zoom on the near player next to a zoom on the far player."""
    tiles = []
    for t in np.arange(t0, t1 + 1e-6, step):
        img = read_frame(cap, t)
        if img is None:
            continue
        h, w = img.shape[:2]
        k = min(range(len(frames)), key=lambda i: abs(frames[i]["t"] - t))
        parts = []
        for key, half in (("r", None), ("o", 60)):
            b = frames[k][key + "b"] or far_box(frames, t) if key == "o" else frames[k]["rb"]
            if b is None:
                parts.append(np.zeros((240, 240, 3), np.uint8))
                continue
            cx, cy = (b[0] + b[2]) / 2 * w, (b[1] + b[3]) / 2 * h
            hh = half or max((b[3] - b[1]) * h, (b[2] - b[0]) * w) * 0.6
            x0, y0 = int(np.clip(cx - hh, 0, max(w - 2 * hh, 0))), int(np.clip(cy - hh, 0, max(h - 2 * hh, 0)))
            crop = img[y0:y0 + int(2 * hh), x0:x0 + int(2 * hh)]
            parts.append(cv2.resize(crop, (240, 240), interpolation=cv2.INTER_CUBIC))
        tile = np.hstack(parts)
        cv2.putText(tile, f"{t:.1f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(tile, f"{t:.1f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(tile)
    cv2.imwrite(str(out_path), contact_sheet(tiles, cols=cols, width=tiles[0].shape[1]))


def serve_sheet(cap, frames, s, out_path):
    tiles = []
    for dt in (-1.5, -1.0, -0.5, 0.0, 0.3, 0.6, 1.0, 1.5):
        t = s["t"] + dt
        img = read_frame(cap, t)
        if img is None:
            continue
        tiles.append(with_zoom(img, far_box(frames, t),
                               f"serve {s['n']}  t={t:.1f}s ({dt:+.1f})  side={s['side']}"))
    cv2.imwrite(str(out_path), contact_sheet(tiles, cols=2, width=tiles[0].shape[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("players")
    ap.add_argument("out")
    ap.add_argument("--min-still", type=float, default=0.8, help="seconds both players must be still")
    ap.add_argument("--v-near", type=float, default=0.7)
    ap.add_argument("--v-far", type=float, default=0.9)
    ap.add_argument("--near-y", type=float, nargs=2, default=(1.0, 4.9))
    ap.add_argument("--far-y", type=float, nargs=2, default=(7.9, 12.4),
                    help="far service area; the low camera puts far feet ~0.5 m too close to the net, "
                         "so the lower bound is 7.9 instead of 8.5")
    ap.add_argument("--x", type=float, nargs=2, default=(0.2, 5.9))
    ap.add_argument("--far-smooth", type=float, default=0.9, help="far player median window (s)")
    ap.add_argument("--max-break", type=float, default=0.3, help="still windows may contain breaks this long (s)")
    ap.add_argument("--merge", type=float, default=6.0, help="merge same-side serves closer than this (s)")
    ap.add_argument("--review", help="serves read by eye (review.json): used instead of the automatic ones")
    ap.add_argument("--sheets", default="work/contact/serves")
    ap.add_argument("--closeups", help="also write 0.1 s close-ups of both players around every serve here")
    args = ap.parse_args()

    data = load_json(args.players)
    frames, fps = data["frames"], data["fps"]
    t = np.array([f["t"] for f in frames])
    half = int(round(0.5 * fps)) | 1                       # 0.5 s window, odd
    R = rolling_median(positions(frames, "r"), half)
    # far depth jumps ~0.5 m between frames (low camera): smooth the far player over 0.9 s
    O = rolling_median(positions(frames, "o"), int(round(args.far_smooth * fps)) | 1)
    vr, vo = speed(R, fps), speed(O, fps)

    with np.errstate(invalid="ignore"):
        ok = ((vr < args.v_near) & (vo < args.v_far)
              & (R[:, 1] >= args.near_y[0]) & (R[:, 1] <= args.near_y[1])
              & (O[:, 1] >= args.far_y[0]) & (O[:, 1] <= args.far_y[1])
              & (R[:, 0] >= args.x[0]) & (R[:, 0] <= args.x[1])
              & (O[:, 0] >= args.x[0]) & (O[:, 0] <= args.x[1]))
    # close short breaks (a jitter frame, the far player hidden behind you for a moment)
    gap = int(round(args.max_break * fps))
    for a, b in zip(runs(ok), runs(ok)[1:]):
        if b[0] - a[1] - 1 <= gap:
            ok[a[1] + 1:b[0]] = True

    cands = []
    for a, b in runs(ok):
        if (b - a + 1) / fps < args.min_still:
            continue
        r, o = np.median(R[a:b + 1], 0), np.median(O[a:b + 1], 0)
        side, diag_ok, margin = side_of(r, o)
        # receiver usually moves first: compare displacement in the 1 s after the stillness
        e = min(b + int(fps), len(frames) - 1)
        dr = np.nan_to_num(np.hypot(*(R[e] - R[b])), nan=0.0)
        do = np.nan_to_num(np.hypot(*(O[e] - O[b])), nan=0.0)
        cands.append({"t": round(float(t[b]), 2), "t_still": round(float(t[a]), 2), "side": side,
                      "diag_ok": diag_ok, "margin": margin, "server_guess": "R" if dr < do else "O",
                      "r": [round(float(v), 2) for v in r], "o": [round(float(v), 2) for v in o]})

    auto = []
    for c in cands:   # a pause before the same serve: keep the last still window
        if auto and c["side"] == auto[-1]["side"] and c["t_still"] - auto[-1]["t"] < args.merge:
            c["t_still"] = auto[-1]["t_still"]
            auto[-1] = c
        else:
            auto.append(c)
    print(f"automatic: {len(cands)} still windows -> {len(auto)} serve candidates")

    if args.review:
        # serves read by eye are the truth; the side is measured from the positions just before contact
        serves = []
        for rv in load_json(args.review)["serves"]:
            m = (t >= rv["t"] - 1.0) & (t <= rv["t"] - 0.1)
            r, o = np.nanmedian(R[m], 0), np.nanmedian(O[m], 0)
            side, diag_ok, margin = side_of(r, o, rv["server"])
            serves.append({"t": rv["t"], "server": rv["server"], "side": side, "diag_ok": diag_ok,
                           "margin": margin, "checked": True,
                           "r": [round(float(v), 2) for v in r], "o": [round(float(v), 2) for v in o]})
        hit = [any(-2.5 <= c["t"] - s["t"] <= 2.0 for c in auto) for s in serves]
        false = [c for c in auto if not any(-2.5 <= c["t"] - s["t"] <= 2.0 for s in serves)]
        print(f"vs review: automatic detection found {sum(hit)}/{len(serves)} serves, "
              f"{len(false)} false ({', '.join(f'{c['t']:.1f}s' for c in false) or 'none'}); "
              f"missed: {', '.join(f'{s['t']:.1f}s' for s, h in zip(serves, hit) if not h) or 'none'}")
    else:
        serves = [dict(c, server=c["server_guess"], checked=False) for c in auto]

    for n, s in enumerate(serves, 1):
        s["n"] = n
    save_json({"serves": serves, "auto": auto, "params": vars(args)}, args.out)
    print(f"{len(serves)} serves ({'read by eye' if args.review else 'automatic, server = guess'}) -> {args.out}")
    for s in serves:
        flag = "" if s["diag_ok"] else "   <-- not diagonal"
        flag += "   <-- side unsure (< 0.15 m from the centre line)" if s["margin"] < 0.15 else ""
        print(f"  #{s['n']:2d}  t={s['t']:6.1f}s  server={s['server']}  side={s['side']:5s} "
              f"({s['margin']:.2f} m)  you=({s['r'][0]:.1f},{s['r'][1]:.1f}) "
              f"opp=({s['o'][0]:.1f},{s['o'][1]:.1f}){flag}")

    Path(args.sheets).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    for s in serves:
        serve_sheet(cap, frames, s, Path(args.sheets) / f"serve_{s['n']:02d}.jpg")
    print(f"contact sheets -> {args.sheets}/serve_NN.jpg")
    if args.closeups:
        Path(args.closeups).mkdir(parents=True, exist_ok=True)
        end = frames[-1]["t"]
        for s in serves:
            serve_closeup(cap, frames, round(s["t"] - 1.5, 1), min(round(s["t"] + 1.2, 1), end), 0.1,
                          Path(args.closeups) / f"closeup_{s['n']:02d}_{s['t']:05.1f}.jpg")
        print(f"0.1 s close-ups (read these to confirm the server) -> {args.closeups}/closeup_NN_T.jpg")


if __name__ == "__main__":
    main()
