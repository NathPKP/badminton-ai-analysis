"""Stage 8: body position of the near player (MediaPipe Pose Landmarker heavy).

Usage: python s8_pose.py <proxy.mp4> <players.json> <points.json> <model.task> <pose_stats.json>
           [--frames work/pose_frames.json] [--sheet work/contact/s8_pose.jpg]

Pose runs on a square crop around your box (1.35x its size), 10 fps. With the camera behind you,
2D knee angles look straighter than they are, so these measures are used instead (all symmetric,
because left/right can swap in a back view):
  crouch       = 1 - (hip-to-ankle height / torso length) / (the same ratio when you stand upright
                 between points); 0 = upright, 0.2 = hips 20 % lower
  stance width = ankle-to-ankle distance / shoulder width (measured in torso lengths, then divided
                 by your square-on shoulder width: the 2D shoulder width collapses when you turn)
  racket up    = a wrist above the shoulder line (prepared); can fire falsely in deep lunges
Checked against 3 blind human-style ratings of 24 crops: hand-up 21/24, stance width well ordered,
crouch only weakly (it separates upright from bent, not slight from deep).
Contexts during rallies: READY = slower than 0.8 m/s and within 1.6 m of the base point (waiting
for the shot), FRONT = past your short service line, REAR = last 1.6 m before your baseline.
"""
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from bball import court
from bball.io import load_json, save_json
from bball.track import positions, rolling_median
from bball.video import contact_sheet, label
from s7_movement import BASE, BACK_M, interp_gaps, moving_mean

CROP = 384
SH, HIP, WR, KN, AN = (11, 12), (23, 24), (15, 16), (25, 26), (27, 28)
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28)]


def crop_box(frame, b, scale=1.35):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = b[0] * w, b[1] * h, b[2] * w, b[3] * h
    cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * scale
    s = CROP / side
    M = np.float32([[s, 0, CROP / 2 - cx * s], [0, s, CROP / 2 - cy * s]])
    return cv2.warpAffine(frame, M, (CROP, CROP), borderValue=(0, 0, 0))


def measures(p):
    """p: (33, 4) x, y, z, visibility in crop fractions -> dict of raw measures (None if not visible)."""
    vis = p[:, 3]
    sh, hip = p[list(SH), :2].mean(0), p[list(HIP), :2].mean(0)
    torso = float(np.linalg.norm(sh - hip))
    out = {"torso": torso, "vis_upper": float(vis[list(SH + HIP)].min()),
           "vis_ankles": float(vis[list(AN)].min()), "vis_wrists": float(vis[list(WR)].max())}
    if torso < 0.03:
        return out
    out["shoulders"] = float(np.linalg.norm(p[SH[0], :2] - p[SH[1], :2])) / torso
    if out["vis_ankles"] > 0.5:
        ank = p[list(AN), :2]
        out["hip_ankle"] = float(ank[:, 1].mean() - hip[1]) / torso
        # in torso lengths: shoulder width collapses when you turn sideways, torso length barely changes
        out["ankles"] = float(np.linalg.norm(ank[0] - ank[1])) / torso
    wr = p[list(WR)]
    # 0.3: a raised racket hand is often blurred; checked against blind ratings (21/24 agree, the
    # misses are lunges where the bent-over shoulder line drops below a forward-reaching hand)
    up = [(sh[1] - y) / torso for (x, y, z, v) in wr if v > 0.3]
    out["wrist_up"] = float(max(up)) if up else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("players")
    ap.add_argument("points")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--frames", default="work/pose_frames.json")
    ap.add_argument("--sheet", default="work/contact/s8_pose.jpg")
    ap.add_argument("--ready-speed", type=float, default=0.8)
    ap.add_argument("--ready-dist", type=float, default=1.6)
    args = ap.parse_args()

    data = load_json(args.players)
    frames, fps = data["frames"], data["fps"]
    t = np.array([f["t"] for f in frames])
    pts = load_json(args.points)["points"]
    P = moving_mean(rolling_median(interp_gaps(positions(frames, "r"), int(fps)), 5), 7)
    v = np.full(len(P), np.nan)
    v[2:-2] = np.hypot(*(P[4:] - P[:-4]).T) / (4 / fps)
    d_base = np.hypot(P[:, 0] - BASE[0], P[:, 1] - BASE[1])

    rally_of = np.zeros(len(t), int)              # point number, 0 = between points
    for p in pts:
        if p["r1"]:
            rally_of[(t >= p["r0"] + 0.3) & (t <= p["r1"])] = p["n"]
    winner = {p["n"]: p["winner"] for p in pts}

    opts = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=args.model),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    lm = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(args.video)
    vfps = cap.get(cv2.CAP_PROP_FPS)
    want = {int(round(f["t"] * vfps)): i for i, f in enumerate(frames) if f["rb"]}
    recs = [None] * len(frames)
    crops = {}
    i = 0
    while want:
        if not cap.grab():
            break
        k = want.pop(i, None)
        i += 1
        if k is None:
            continue
        _, img = cap.retrieve()
        crop = crop_box(img, frames[k]["rb"])
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), int(t[k] * 1000))
        if not res.pose_landmarks:
            continue
        p = np.array([[q.x, q.y, q.z, q.visibility] for q in res.pose_landmarks[0]])
        m = measures(p)
        m["trunc"] = bool(frames[k]["rt"])
        recs[k] = {"t": t[k], "lm": np.round(p[:, [0, 1, 3]], 3).tolist(), **m}
        if rally_of[k]:
            crops[k] = crop
        if k % 200 == 0:
            print(f"  {t[k]:6.1f} s", flush=True)

    # upright reference: between points, standing still, ankles visible; upper part of the distribution
    between = [r["hip_ankle"] for k, r in enumerate(recs)
               if r and rally_of[k] == 0 and "hip_ankle" in r and v[k] < 0.6 and not r["trunc"]]
    upright = float(np.percentile(between, 80))
    # your shoulder width seen square-on (back to the camera): upper part of the between-point distribution
    shoulders = float(np.percentile([r["shoulders"] for k, r in enumerate(recs)
                                     if r and rally_of[k] == 0 and r["vis_upper"] > 0.5], 80))
    print(f"pose found on {sum(r is not None for r in recs)}/{sum(f['rb'] is not None for f in frames)} frames; "
          f"upright hip-to-ankle = {upright:.2f} torso, shoulder width = {shoulders:.2f} torso "
          f"({len(between)} between-point frames)")

    rows = []
    for k, r in enumerate(recs):
        if not r or not rally_of[k] or r["vis_upper"] < 0.5:
            continue
        ctx = []
        if v[k] < args.ready_speed and d_base[k] <= args.ready_dist:
            ctx.append("READY")
        if P[k, 1] >= court.SHORT_SERVICE_Y[0]:
            ctx.append("FRONT")
        if P[k, 1] <= BACK_M or r["trunc"]:
            ctx.append("REAR")
        rows.append({"k": k, "t": r["t"], "n": int(rally_of[k]), "won": None if winner[rally_of[k]] is None else winner[rally_of[k]] == "R", "ctx": ctx,
                     "crouch": None if "hip_ankle" not in r or r["trunc"] else round(1 - r["hip_ankle"] / upright, 3),
                     "stance": None if "ankles" not in r or r["trunc"] else round(r["ankles"] / shoulders, 2),
                     "racket_up": None if r["wrist_up"] is None else bool(r["wrist_up"] > 0)})

    def summarise(sel):
        c = [x["crouch"] for x in sel if x["crouch"] is not None]
        s = [x["stance"] for x in sel if x["stance"] is not None]
        u = [x["racket_up"] for x in sel if x["racket_up"] is not None]
        return {"frames": len(sel), "crouch": round(float(np.mean(c)), 3) if c else None,
                "stance": round(float(np.mean(s)), 2) if s else None,
                "racket_up": round(float(np.mean(u)), 2) if u else None,
                "n_crouch": len(c), "n_racket": len(u)}

    stats = {}
    for ctx in ("ALL", "READY", "FRONT", "REAR"):
        sel = [x for x in rows if ctx == "ALL" or ctx in x["ctx"]]
        stats[ctx] = {"all": summarise(sel), "won": summarise([x for x in sel if x["won"] is True]),
                      "lost": summarise([x for x in sel if x["won"] is False])}
    save_json({"upright_ratio": round(upright, 3), "shoulder_torso": round(shoulders, 3), "stats": stats,
               "per_point": {n: summarise([x for x in rows if x["n"] == n and "READY" in x["ctx"]])
                             for n in sorted({x["n"] for x in rows})}}, args.out)
    save_json({"fps": fps, "frames": [r for r in recs if r]}, args.frames)

    print(f"{len(rows)} rally frames with pose -> {args.out}")
    print("  context  part  frames  crouch  stance  racket-up")
    for ctx, d in stats.items():
        for part in ("all", "won", "lost"):
            s = d[part]
            f = lambda x, fmt: format(x, fmt) if x is not None else "   -"
            print(f"  {ctx:6s} {part:5s} {s['frames']:6d}  {f(s['crouch'], '6.3f')}  {f(s['stance'], '6.2f')}  "
                  f"{f(s['racket_up'], '8.0%')}")

    # contact sheet: skeletons on 16 rally frames spread over the match, READY first
    ready = [x for x in rows if "READY" in x["ctx"]]
    other = [x for x in rows if "READY" not in x["ctx"]]
    pick = [ready[j] for j in np.linspace(0, len(ready) - 1, min(10, len(ready))).astype(int)] if ready else []
    pick += [other[j] for j in np.linspace(0, len(other) - 1, min(6, len(other))).astype(int)] if other else []
    tiles = []
    for x in sorted(pick, key=lambda x: x["t"]):
        img = crops[x["k"]].copy()
        p = np.array(recs[x["k"]]["lm"])
        for a, b in BONES:
            if p[a, 2] > 0.5 and p[b, 2] > 0.5:
                cv2.line(img, tuple((p[a, :2] * CROP).astype(int)), tuple((p[b, :2] * CROP).astype(int)), (0, 255, 0), 2)
        sh_y = int(p[list(SH), 1].mean() * CROP)
        cv2.line(img, (0, sh_y), (CROP, sh_y), (0, 200, 255), 1)
        label(img, f"#{x['n']} {x['t']:.1f}s {'/'.join(x['ctx']) or 'MOVING'}", (6, 20), 0.5)
        label(img, f"crouch {x['crouch'] if x['crouch'] is not None else '-'}  stance {x['stance'] or '-'}  "
                   f"racket {'UP' if x['racket_up'] else 'down' if x['racket_up'] is not None else '-'}", (6, 42), 0.45)
        tiles.append(img)
    Path(args.sheet).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.sheet, contact_sheet(tiles, cols=4, width=CROP))
    print(f"contact sheet -> {args.sheet}  (orange line = shoulder line; a wrist above it = racket up)")


if __name__ == "__main__":
    main()
