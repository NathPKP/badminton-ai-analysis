"""Stage 13: shuttle tracking with TrackNetV3 (ONNX, CPU), one position per frame.

Usage: python s13_shuttle.py <proxy.mp4> <tracknet.onnx> <shuttle.json> [--inpaint inpaintnet.onnx]
                             [--t0 12] [--t1 150] [--sheet contact/s13_shuttle.jpg] [--check-video check.mp4]

TrackNetV3 (qaz812345/TrackNetV3, MIT, weights included) reads 8 frames at a time, resized to 512x288, after
a median background image, and outputs one heatmap per frame. The shuttle is the centre of the largest blob
above 0.5. InpaintNet then fills short gaps inside a flight (the shuttle hidden by a player or blurred), as in
the original predict.py. The ONNX files come from download_models.py --shuttle.

The original takes one background for the whole video, which assumes a fixed camera. Here the camera is
scanned first (cuts, camera movement): each still stretch of a shot gets its own background, a moving camera
one per second, and no 8-frame window, filled-in gap or flight crosses a cut. --bg global restores the original.
On broadcast footage, --view drops the close-ups and replays (where the real shuttle is seen too, out of play).

Output: {"fps": 30.0, "t0": 12.0, "w": 848, "h": 478, "model": "TrackNetV3",
         "p": [[x, y, how] or null, ...],   one entry per frame from t0, x and y as fractions of the frame,
                                            how = 1 seen by TrackNet, 2 filled in by InpaintNet; null = not seen
         "flights": [{"t0", "t1", "n", "a": [x, y], "b": [x, y]}, ...],   moving runs (bball.shuttle.flights)
         "camera": [{"t0", "t1", "shot", "moving"}, ...]}   shots (numbered from 0) and camera movement
"""
import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from bball.io import load_json, save_json
from bball.shuttle import flights
from bball.video import contact_sheet, label

H, W = 288, 512
SEQ = 8                                  # TrackNet frames per sample (from the checkpoint)
SEQ_INPAINT = 16
COOR_TH = 50 / math.sqrt(H ** 2 + W ** 2)   # normalised coordinates below this on both axes mean "not seen"


def small_rgb(frame):
    """BGR frame -> (3, 288, 512) uint8 RGB, resized like the original (PIL bicubic with antialiasing)."""
    img = Image.fromarray(frame[:, :, ::-1]).resize((W, H), Image.BICUBIC)
    return np.asarray(img).transpose(2, 0, 1)


def read_range(cap, fps, t0, t1):
    first = int(round(t0 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    last = int(round(t1 * fps)) if t1 else None
    i = first
    while last is None or i < last:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1


def blob_centre(heat):
    """Centre (x, y) in heatmap pixels of the largest blob above 0.5, or None (as predict_location)."""
    mask = (heat > 0.5).astype(np.uint8)
    if not mask.any():
        return None
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in cnts]
    x, y, w, h = max(rects, key=lambda r: r[2] * r[3])      # first of equal areas, like the original
    return int(x + w / 2), int(y + h / 2)


def scan(video, fps, t0, t1):
    """Camera behaviour per frame from t0: (shot number, camera moving).

    A cut is a jump of the grey-level histogram (total variation > 0.25) or of the mean pixel change (> 40) between
    two frames. The camera moves when the image shifts by more than 0.3 % of its width in 0.2 s (phase correlation
    on 160x90 grey frames), or when no shift fits at all (zoom, fast pan); smoothed over 1 s.
    """
    cap = cv2.VideoCapture(str(video))
    lag = max(1, int(round(0.2 * fps)))
    win = cv2.createHanningWindow((160, 90), cv2.CV_32F)
    shots, shift, hist, recent = [], [], None, []
    shot = 0
    for i, frame in read_range(cap, fps, t0, t1):
        g = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        h = cv2.calcHist([g], [0], None, [32], [0, 256]).ravel()
        h /= h.sum()
        if hist is not None and (0.5 * np.abs(h - hist).sum() > 0.25
                                 or np.abs(g.astype(np.float32) - recent[-1]).mean() > 40):
            shot += 1
            recent = []
        hist = h
        recent = (recent + [g.astype(np.float32)])[-(lag + 1):]
        if len(recent) == lag + 1 and len(shift) % 3 == 0:
            # OpenCV 5 multiplies its inputs by the window in place: pass copies
            (dx, dy), resp = cv2.phaseCorrelate(recent[0].copy(), recent[-1].copy(), win)
            shift.append(np.hypot(dx, dy) if resp > 0.1 else 99.0)
        else:
            shift.append(shift[-1] if shift and len(recent) > 1 else 0.0)
        shots.append(shot)
    cap.release()
    moving = np.array(shift) > 0.005 * 160
    k = int(fps) | 1                                   # 1 s majority vote
    pad = np.pad(moving.astype(np.float32), k // 2, mode="edge")
    moving = np.convolve(pad, np.ones(k) / k, mode="valid") > 0.5
    return np.array(shots), moving


def segments(shots, moving, fps, mode):
    """Background segments as (first, last, sample_first, sample_last) frame offsets.

    mode "global": one background for everything (the original method, for a fixed camera).
    mode "shot": one background per still stretch of each shot; when the camera moves, one per second, from the
    frames within 1 s around it in the same shot.
    """
    n = len(shots)
    if mode == "global":
        return [(0, n - 1, 0, n - 1)]
    out, sec = [], max(1, int(round(fps)))
    i = 0
    while i < n:
        j = i
        while j + 1 < n and shots[j + 1] == shots[i] and moving[j + 1] == moving[i]:
            j += 1
        s0 = i
        while s0 > 0 and shots[s0 - 1] == shots[i]:
            s0 -= 1
        s1 = j
        while s1 + 1 < n and shots[s1 + 1] == shots[i]:
            s1 += 1
        if not moving[i]:
            out.append((i, j, i, j))
        else:
            for a in range(i, j + 1, sec):
                b = min(a + sec - 1, j)
                out.append((a, b, max(s0, a - sec), min(s1, b + sec)))
        i = j + 1
    return out


def backgrounds(video, fps, t0, t1, segs, samples):
    """Median image of each segment's sample range (the court without the players), read in one pass."""
    need = {}
    for k, (_, _, a, b) in enumerate(segs):
        step = max(3, (b - a + 1) // samples)            # 10 samples/s at most at 30 fps
        for f in range(a, b + 1, step):
            need.setdefault(f, []).append(k)
    last = {}
    for f, ks in need.items():
        for k in ks:
            last[k] = max(last.get(k, -1), f)
    pending, done = {}, {}
    cap = cv2.VideoCapture(str(video))
    first = int(round(t0 * fps))
    for i, frame in read_range(cap, fps, t0, t1):
        f = i - first
        if f not in need:
            continue
        img = small_rgb(frame)
        for k in need[f]:
            pending.setdefault(k, []).append(img)
            if last[k] == f:
                done[k] = np.median(np.stack(pending.pop(k)), 0).astype(np.uint8)
    cap.release()
    for k in list(pending):                             # video shorter than announced
        done[k] = np.median(np.stack(pending.pop(k)), 0).astype(np.uint8)
    return done


def track(session, video, fps, t0, t1, shots, segs, bgs, batch=2):
    """TrackNet on 8-frame windows that never straddle a cut, each with the background of its first frame.
    Returns a list of (x, y) in heatmap pixels or None, one per frame from t0."""
    seg_of = np.zeros(len(shots), dtype=int)
    for k, (a, b, _, _) in enumerate(segs):
        seg_of[a:b + 1] = k
    cap = cv2.VideoCapture(str(video))
    first = int(round(t0 * fps))
    out, window, windows = [], [], []
    started = time.time()

    def close():
        # pad an incomplete window (before a cut, at the end) with its last frame
        n = len(window)
        windows.append((window[0][0], [w[1] for w in window] + [window[-1][1]] * (SEQ - n), n))

    def flush():
        x = np.stack([np.concatenate([bgs[seg_of[f0]]] + fr).astype(np.float32) / 255 for f0, fr, _ in windows])
        heat = session.run(None, {session.get_inputs()[0].name: x})[0]
        for (_, _, n), hm in zip(windows, heat):
            out.extend(blob_centre(hm[k]) for k in range(n))
        windows.clear()

    for i, frame in read_range(cap, fps, t0, t1):
        f = i - first
        if f >= len(shots):
            break
        if window and shots[f] != shots[window[0][0]]:
            close()
            window = []
        window.append((f, small_rgb(frame)))
        if len(window) == SEQ:
            close()
            window = []
        if len(windows) >= batch:
            flush()
        if (f + 1) % 900 == 0:
            print(f"  {i / fps:6.1f} s   ({time.time() - started:.0f} s elapsed)", flush=True)
    if window:
        close()
    if windows:
        flush()
    cap.release()
    return out


def inpaint_mask(ys, seen, th_h, max_gap=SEQ_INPAINT - 1, shots=None):
    """Frames to fill in: gaps between two sightings low enough in the image (generate_inpaint_mask).

    Unlike the original, gaps longer than InpaintNet's window are left empty: over a close-up or a replay
    the network would otherwise invent a flight."""
    n = len(seen)
    mask = np.zeros(n, dtype=np.float32)
    i = j = 0
    while j < n:
        while i < n - 1 and seen[i]:
            i += 1
        j = i
        while j < n - 1 and not seen[j]:
            j += 1
        if j == i:
            break
        if i == 0 and ys[j] > th_h and j <= max_gap:
            mask[:j] = 1
        elif i > 1 and ys[i - 1] > th_h and ys[j] > th_h and j - i <= max_gap and (
                shots is None or shots[i - 1] == shots[j]):
            mask[i:j] = 1
        i = j
    return mask


def inpaint(session, pts, shots=None):
    """InpaintNet over sliding 16-frame windows, weighted towards the window centre (eval_mode 'weight')."""
    n = len(pts)
    xy = (np.array([p if p else (0, 0) for p in pts], dtype=np.float32) / np.float32([W, H])).astype(np.float32)
    seen = np.array([p is not None for p in pts])
    mask = inpaint_mask(xy[:, 1] * H, seen, H * 0.05, shots=shots)
    if n < SEQ_INPAINT or not mask.any():
        return xy, seen, mask
    wt = np.array([min(k + 1, SEQ_INPAINT - k) for k in range(SEQ_INPAINT)], dtype=np.float32)
    acc = np.zeros((n, 2), dtype=np.float32)
    tot = np.zeros(n, dtype=np.float32)
    starts = list(range(0, n - SEQ_INPAINT + 1))
    for b in range(0, len(starts), 256):
        st = starts[b:b + 256]
        c = np.stack([xy[s:s + SEQ_INPAINT] for s in st])
        m = np.stack([mask[s:s + SEQ_INPAINT, None] for s in st])
        res = session.run(None, {"coords": c, "mask": m})[0]
        res = res * m + c * (1 - m)
        for s, r in zip(st, res):
            acc[s:s + SEQ_INPAINT] += r * wt[:, None]
            tot[s:s + SEQ_INPAINT] += wt
    out = acc / tot[:, None]
    gone = (out[:, 0] < COOR_TH) & (out[:, 1] < COOR_TH)
    out[gone] = 0
    return out, seen, mask


def draw_check(video, fps, t0, res, sheet=None, out_video=None, n_sheet=12):
    """Contact sheet of frames where the shuttle was found (with its last 8 positions), and/or a check video."""
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    found = [k for k, p in enumerate(res) if p]
    pick = set(found[int(j)] for j in np.linspace(0, len(found) - 1, n_sheet)) if (sheet and found) else set()
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if out_video else None
    tiles = []
    for i, frame in read_range(cap, fps, t0, t0 + len(res) / fps):
        k = i - int(round(t0 * fps))
        if k >= len(res) or not (writer or k in pick):
            continue
        for j in range(max(0, k - 7), k + 1):
            p = res[j]
            if p:
                col = (0, 255, 255) if p[2] == 1 else (255, 0, 255)
                cv2.circle(frame, (int(p[0] * w), int(p[1] * h)), 5 if j == k else 2, col, 2 if j == k else -1)
        label(frame, f"t={i / fps:.2f}s")
        if writer:
            writer.write(frame)
        if k in pick:
            tiles.append(frame)
    cap.release()
    if writer:
        writer.release()
    if sheet and tiles:
        Path(sheet).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sheet), contact_sheet(tiles, cols=4))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("model", help="tracknet.onnx")
    ap.add_argument("out")
    ap.add_argument("--inpaint", help="inpaintnet.onnx: fill short gaps in each flight")
    ap.add_argument("--view", help="players.json of broadcast footage (s3 --check-view): keep the shuttle only "
                                   "where the main court view is on screen, not in close-ups and replays")
    ap.add_argument("--t0", type=float, default=0.0, help="start (s)")
    ap.add_argument("--t1", type=float, help="end (s)")
    ap.add_argument("--median-samples", type=int, default=150, help="most frames per background image")
    ap.add_argument("--bg", choices=["shot", "global"], default="shot",
                    help="shot: one background per still stretch of each shot, per second while the camera moves "
                         "(default); global: one for the whole range, as the original (fixed camera only)")
    ap.add_argument("--threads", type=int, default=0, help="onnxruntime threads (0 = automatic)")
    ap.add_argument("--sheet", help="save a 12-frame contact sheet with the shuttle drawn")
    ap.add_argument("--check-video", help="save a check video with the shuttle drawn (slow, large)")
    args = ap.parse_args()

    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads
    net = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    started = time.time()
    shots, moving = scan(args.video, fps, args.t0, args.t1)
    segs = segments(shots, moving, fps, args.bg)
    print(f"camera: {shots[-1] + 1} shot(s), moving {100 * moving.mean():.0f}% of the time, "
          f"{len(segs)} background image(s)  ({time.time() - started:.0f} s)", flush=True)
    bgs = backgrounds(args.video, fps, args.t0, args.t1, segs, args.median_samples)
    pts = track(net, args.video, fps, args.t0, args.t1, shots, segs, bgs)
    shots, moving = shots[:len(pts)], moving[:len(pts)]
    if args.view:
        main_view = {round(f["t"], 1): f["v"] for f in load_json(args.view)["frames"] if "v" in f}
        off = [k for k in range(len(pts)) if not main_view.get(round(args.t0 + k / fps, 1), True)]
        for k in off:
            pts[k] = None
        print(f"main view: {100 * (1 - len(off) / max(1, len(pts))):.0f}% of the frames kept")
    res = [[round(p[0] / W, 4), round(p[1] / H, 4), 1] if p else None for p in pts]
    if args.inpaint:
        xy, seen, mask = inpaint(ort.InferenceSession(args.inpaint, providers=["CPUExecutionProvider"]), pts, shots)
        for k in np.nonzero(mask)[0]:
            if xy[k].any():
                res[k] = [round(float(xy[k, 0]), 4), round(float(xy[k, 1]), 4), 2]
    fl = flights(res, fps, args.t0, w, h, shots=shots)
    cam = []
    for k, (a, b, _, _) in enumerate(segments(shots, moving, fps, "shot")):
        if cam and cam[-1]["shot"] == int(shots[a]) and cam[-1]["moving"] == bool(moving[a]):
            cam[-1]["t1"] = round(args.t0 + b / fps, 2)
        else:
            cam.append({"t0": round(args.t0 + a / fps, 2), "t1": round(args.t0 + b / fps, 2),
                        "shot": int(shots[a]), "moving": bool(moving[a])})
    save_json({"fps": round(fps, 3), "t0": args.t0, "w": w, "h": h, "model": "TrackNetV3", "bg": args.bg,
               "p": res, "flights": fl, "camera": cam}, args.out)

    seen = sum(1 for p in res if p and p[2] == 1)
    filled = sum(1 for p in res if p and p[2] == 2)
    print(f"{len(res)} frames ({len(res) / fps:.0f} s) in {time.time() - started:.0f} s -> {args.out}")
    print(f"shuttle seen on {seen} frames ({100 * seen / max(1, len(res)):.0f}%), filled in on {filled}; "
          f"{len(fl)} moving flights")
    if args.sheet or args.check_video:
        draw_check(args.video, fps, args.t0, res, args.sheet, args.check_video)
        for p in (args.sheet, args.check_video):
            if p:
                print(f"check -> {p}")


if __name__ == "__main__":
    main()
