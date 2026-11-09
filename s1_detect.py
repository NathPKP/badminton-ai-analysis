"""Stage 1: person detection with YOLOX-S (ONNX, CPU) every 3rd frame (10 fps).

Usage: python s1_detect.py <proxy.mp4> <yolox_s.onnx> <dets.json> [--sheet work/contact/s1_dets.jpg]

Output: {"fps": 10.0, "frames": [{"t": 12.3, "p": [[x1, y1, x2, y2, score], ...]}, ...]}
with box coordinates as fractions of the frame width / height.
"""
import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort

from bball.io import save_json
from bball.video import contact_sheet, label

INPUT = 640
STRIDES = (8, 16, 32)


def make_grids():
    grids, strides = [], []
    for s in STRIDES:
        n = INPUT // s
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


def preprocess(frame):
    """Letterbox to 640x640 (top-left, pad 114), BGR, no normalisation (YOLOX >= 0.1.1rc0)."""
    h, w = frame.shape[:2]
    r = min(INPUT / h, INPUT / w)
    resized = cv2.resize(frame, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((INPUT, INPUT, 3), 114, dtype=np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    return padded.transpose(2, 0, 1)[None].astype(np.float32), r


def detect(session, frame, grids, strides, conf=0.3, iou=0.5):
    blob, r = preprocess(frame)
    out = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
    xy = (out[:, :2] + grids) * strides
    wh = np.exp(out[:, 2:4]) * strides
    score = out[:, 4] * out[:, 5]                       # objectness * person class
    keep = score >= conf
    xy, wh, score = xy[keep], wh[keep], score[keep]
    if len(score) == 0:
        return []
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r
    idx = cv2.dnn.NMSBoxes(
        [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
        score.tolist(), conf, iou)
    h, w = frame.shape[:2]
    res = []
    for i in np.array(idx).reshape(-1):
        x1, y1, x2, y2 = boxes[i]
        res.append([round(float(np.clip(x1 / w, 0, 1)), 4), round(float(np.clip(y1 / h, 0, 1)), 4),
                    round(float(np.clip(x2 / w, 0, 1)), 4), round(float(np.clip(y2 / h, 0, 1)), 4),
                    round(float(score[i]), 3)])
    return sorted(res, key=lambda b: -b[4])


def draw(frame, people, t):
    img = frame.copy()
    h, w = img.shape[:2]
    for x1, y1, x2, y2, s in people:
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        label(img, f"{s:.2f}", (p1[0], max(p1[1] - 4, 12)), 0.45)
    label(img, f"t={t:.1f}s  n={len(people)}")
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--step", type=int, default=3, help="run on every Nth frame (3 -> 10 fps at 30 fps)")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--sheet", help="save a 12-frame contact sheet with the boxes drawn")
    args = ap.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    grids, strides = make_grids()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames, kept = [], {}
    sheet_idx = set(np.linspace(0, total - 1, 12).astype(int) // args.step * args.step)
    t0, i = time.time(), 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % args.step == 0:
            _, frame = cap.retrieve()
            t = round(i / fps, 3)
            people = detect(session, frame, grids, strides, args.conf, args.iou)
            frames.append({"t": t, "p": people})
            if args.sheet and i in sheet_idx:
                kept[i] = draw(frame, people, t)
            if len(frames) % 100 == 0:
                print(f"  {t:6.1f} s / {total / fps:.1f} s   ({time.time() - t0:.0f} s elapsed)", flush=True)
        i += 1

    save_json({"fps": round(fps / args.step, 3), "frames": frames}, args.out)
    counts = np.array([len(f["p"]) for f in frames])
    print(f"{len(frames)} frames -> {args.out}")
    print(f"people per frame: mean {counts.mean():.2f}, 0: {(counts == 0).sum()}, 1: {(counts == 1).sum()}, "
          f"2: {(counts == 2).sum()}, 3+: {(counts >= 3).sum()}")
    if args.sheet:
        from pathlib import Path
        Path(args.sheet).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.sheet, contact_sheet([kept[k] for k in sorted(kept)], cols=4))
        print(f"contact sheet -> {args.sheet}")


if __name__ == "__main__":
    main()
