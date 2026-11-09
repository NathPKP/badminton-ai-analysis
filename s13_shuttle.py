"""Stage 13: shuttle tracking with TrackNetV3 (ONNX, CPU), one position per frame.

Usage: python s13_shuttle.py <proxy.mp4> <tracknet.onnx> <shuttle.json> [--inpaint inpaintnet.onnx]
                             [--t0 12] [--t1 150] [--sheet contact/s13_shuttle.jpg] [--check-video check.mp4]

TrackNetV3 (qaz812345/TrackNetV3, MIT, weights included) reads 8 frames at a time, resized to 512x288, after
a median background image of the video, and outputs one heatmap per frame. The shuttle is the centre of the
largest blob above 0.5. InpaintNet then fills short gaps inside a flight (the shuttle hidden by a player or
blurred), as in the original predict.py. Frames are read once, in order, so memory does not grow with the
video length. The ONNX files come from download_models.py --shuttle.

Output: {"fps": 30.0, "t0": 12.0, "w": 848, "h": 478, "model": "TrackNetV3",
         "p": [[x, y, how] or null, ...],   one entry per frame from t0, x and y as fractions of the frame,
                                            how = 1 seen by TrackNet, 2 filled in by InpaintNet; null = not seen
         "flights": [{"t0", "t1", "n", "a": [x, y], "b": [x, y]}, ...]}   moving runs (bball.shuttle.flights)
"""
import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from bball.io import save_json
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


def median_background(video, fps, t0, t1, samples):
    """Median of up to `samples` frames spread over the range (the empty court, players removed)."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(round(t0 * fps))
    last = min(total, int(round(t1 * fps))) if t1 else total
    step = max(1, (last - first) // samples)
    keep = []
    i = first
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    while i < last:
        if not cap.grab():
            break
        if (i - first) % step == 0:
            keep.append(small_rgb(cap.retrieve()[1]))
        i += 1
    cap.release()
    return np.median(np.stack(keep), 0).astype(np.uint8)


def blob_centre(heat):
    """Centre (x, y) in heatmap pixels of the largest blob above 0.5, or None (as predict_location)."""
    mask = (heat > 0.5).astype(np.uint8)
    if not mask.any():
        return None
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in cnts]
    x, y, w, h = max(rects, key=lambda r: r[2] * r[3])      # first of equal areas, like the original
    return int(x + w / 2), int(y + h / 2)


def track(session, video, fps, t0, t1, median, batch=2):
    """TrackNet on non-overlapping 8-frame windows. Returns a list of (x, y) in heatmap pixels or None."""
    cap = cv2.VideoCapture(str(video))
    med = median.astype(np.float32) / 255
    out, window, windows = [], [], []
    started, n = time.time(), 0

    def flush():
        x = np.stack([np.concatenate([med] + [f.astype(np.float32) / 255 for f in w]) for w in windows])
        heat = session.run(None, {session.get_inputs()[0].name: x})[0]
        for w, hm in zip(windows, heat):
            out.extend(blob_centre(hm[k]) for k in range(w.n))
        windows.clear()

    class Win(list):
        n = SEQ

    for i, frame in read_range(cap, fps, t0, t1):
        window.append(small_rgb(frame))
        if len(window) == SEQ:
            w = Win(window)
            windows.append(w)
            window = []
            if len(windows) == batch:
                flush()
        n += 1
        if n % 900 == 0:
            print(f"  {i / fps:6.1f} s   ({time.time() - started:.0f} s elapsed)", flush=True)
    if window:                                   # last, incomplete window: pad with its last frame
        w = Win(window + [window[-1]] * (SEQ - len(window)))
        w.n = len(window)
        windows.append(w)
    if windows:
        flush()
    cap.release()
    return out


def inpaint_mask(ys, seen, th_h, max_gap=SEQ_INPAINT - 1):
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
        elif i > 1 and ys[i - 1] > th_h and ys[j] > th_h and j - i <= max_gap:
            mask[i:j] = 1
        i = j
    return mask


def inpaint(session, pts):
    """InpaintNet over sliding 16-frame windows, weighted towards the window centre (eval_mode 'weight')."""
    n = len(pts)
    xy = (np.array([p if p else (0, 0) for p in pts], dtype=np.float32) / np.float32([W, H])).astype(np.float32)
    seen = np.array([p is not None for p in pts])
    mask = inpaint_mask(xy[:, 1] * H, seen, H * 0.05)
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
    ap.add_argument("--t0", type=float, default=0.0, help="start (s)")
    ap.add_argument("--t1", type=float, help="end (s)")
    ap.add_argument("--median-samples", type=int, default=300, help="frames used for the background image")
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
    median = median_background(args.video, fps, args.t0, args.t1, args.median_samples)
    pts = track(net, args.video, fps, args.t0, args.t1, median)
    res = [[round(p[0] / W, 4), round(p[1] / H, 4), 1] if p else None for p in pts]
    if args.inpaint:
        xy, seen, mask = inpaint(ort.InferenceSession(args.inpaint, providers=["CPUExecutionProvider"]), pts)
        for k in np.nonzero(mask)[0]:
            if xy[k].any():
                res[k] = [round(float(xy[k, 0]), 4), round(float(xy[k, 1]), 4), 2]
    fl = flights(res, fps, args.t0, w, h)
    save_json({"fps": round(fps, 3), "t0": args.t0, "w": w, "h": h, "model": "TrackNetV3", "p": res,
               "flights": fl}, args.out)

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
