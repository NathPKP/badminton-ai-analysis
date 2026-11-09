"""Stage 2 (automatic): court calibration from the painted lines, no clicks.

Usage: python s2_autocalib.py <proxy.mp4> <calib.json> [--dets dets.json] [--t0 0] [--t1 END] [--check out.jpg]

1. Line mask: bright, thin, unsaturated pixels (white top-hat), on many frames spread over [t0, t1]; the
   frames that agree with each other (the main camera view, not replays or close-ups) are combined, which
   also removes the players.
2. Straight lines (Hough), split into lines across the court (baselines, service lines) and along it
   (sidelines).
3. Every pair of lines across x pair along, matched to every pair of court lines in the same order, gives a
   homography; it is scored by how much of the projected court falls on the mask.
4. The best one is refined on every court line, and saved like s2_calibrate.py does (named points + H),
   so `s2_calibrate.py` can open it to correct it by hand.
5. Checks: the share of the projected court lying on a painted line (--min-lines, 0.6), and with --dets
   (stage 1) the share of frames with people where somebody stands in each half of the court
   (--min-feet, 0.25), and the height people would have if the court were right: a standing person
   spans, in pixels, about as much as the same length of court laid across at their feet. 70 % of them
   must be 1.3-2.4 m tall, with a median of 1.45-2.0 m. Wrong courts give 0.6, 4 or 8 m (a crowd at the
   court side can put "somebody in each half" by chance; heights are much harder to fake). Lines may be
   partly hidden, but 60 % of the court must lie on painted lines (--min-lines): with less, wrong courts
   got through (a corner camera on a wooden floor gave a crooked court with 50 % on the lines and 97 %
   human heights). When in doubt the window opens: a few clicks cost less than a wrong court. Without --dets the line share
   must reach 0.8. Light and dark lines and up to 4 candidate views each are tried; the best by
   lines x plausible heights is kept. Exit code 2 when a check fails: the lines are not visible enough (camera on the
   floor, cluttered floor); calibrate by hand with s2_calibrate.py.
"""
import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np

from bball import court
from bball.io import load_json, probe, save_json
from s2_calibrate import check_image, solve

ACROSS = [0.0, court.DOUBLES_LONG_SERVICE_Y[0], court.SHORT_SERVICE_Y[0], court.SHORT_SERVICE_Y[1],
          court.DOUBLES_LONG_SERVICE_Y[1], court.LENGTH]
ALONG = [0.0, court.SINGLES_X[0], court.SINGLES_X[1], court.WIDTH]
PAINTED = [s for s in court.court_lines() if not (s[0][1] == court.NET_Y and s[1][1] == court.NET_Y)]


def line_mask(frame, dark=False):
    """Painted-line pixels: thin and lighter than their surroundings (white or light lines, unsaturated),
    or with dark=True darker than them (dark lines on a light wooden floor)."""
    h = frame.shape[0]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, h // 40) | 1,) * 2)
    if dark:
        return (cv2.morphologyEx(hsv[:, :, 2], cv2.MORPH_BLACKHAT, k) > 30).astype(np.uint8)
    tophat = cv2.morphologyEx(hsv[:, :, 2], cv2.MORPH_TOPHAT, k)
    return ((tophat > 25) & (hsv[:, :, 1] < 70)).astype(np.uint8)


def straightness(mask):
    """Total length of the long straight segments in a mask (a court view has many, a close-up few)."""
    h = mask.shape[0]
    segs = cv2.HoughLinesP(mask * 255, 1, np.pi / 360, threshold=40, minLineLength=h // 8, maxLineGap=h // 60)
    if segs is None:
        return 0.0
    segs = segs.reshape(-1, 4).astype(float)
    return float(np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]).sum())


def sample_frames(cap, fps, t0, t1, n=80):
    out = []
    for t in np.linspace(t0, t1, n + 2)[1:-1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, f = cap.read()
        if ok:
            out.append((float(t), f))
    return out


def sample_masks(samples, dark):
    """Line mask of every sampled frame (dilated by 1 px) and its long-straight-line content."""
    raw = [line_mask(f, dark) for _, f in samples]
    return [cv2.dilate(m, np.ones((3, 3), np.uint8)) for m in raw], np.array([straightness(m) for m in raw])


def view_masks(samples, masks, lines, k=4):
    """Up to k candidate views: (mask, reference frame, its time, frames in the view). For each, a
    reference frame with many long straight lines (not like an earlier pick) and the frames that look
    like it; the mask keeps the lines found in at least half of them, which removes the players."""
    small = np.array([cv2.resize(m, (160, 90), interpolation=cv2.INTER_AREA) > 0 for m in masks], np.float32)
    m = small.reshape(len(small), -1)
    inter = m @ m.T
    sizes = m.sum(1)
    iou = inter / np.maximum(sizes[:, None] + sizes[None, :] - inter, 1)
    out, taken = [], np.zeros(len(masks), bool)
    for ref in np.argsort(-lines):
        if len(out) == k or lines[ref] < 0.3 * lines.max():
            break
        if taken[ref]:
            continue
        others = np.delete(iou[ref], ref)
        same = np.where(iou[ref] >= 0.5 * (others.max() if len(others) else 1.0))[0]
        taken[same] = True
        mask = (np.mean([masks[i] for i in same], 0) >= 0.5).astype(np.uint8)
        out.append((mask, samples[ref][1], samples[ref][0], len(same)))
    return out


def in_view(H, samples, masks, min_share=0.35):
    """Times of the sampled frames whose own painted lines match this court (the main view is on)."""
    return [t for (t, _), m in zip(samples, masks) if score(H, m)[1] >= min_share]


def detect_lines(mask):
    """Merged straight lines as (vx, vy, x0, y0), split into across (near horizontal) and along."""
    h, w = mask.shape
    segs = cv2.HoughLinesP(mask * 255, 1, np.pi / 360, threshold=40, minLineLength=h // 12, maxLineGap=h // 60)
    if segs is None:
        return [], []
    segs = segs.reshape(-1, 4).astype(float)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    lines = []
    used = np.zeros(len(segs), bool)
    for i in np.argsort(-np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])):
        if used[i]:
            continue
        x1, y1, x2, y2 = segs[i]
        d = np.array([x2 - x1, y2 - y1]) / np.hypot(x2 - x1, y2 - y1)
        nrm = np.array([-d[1], d[0]])
        dist = np.abs((segs[:, :2] - [x1, y1]) @ nrm), np.abs((segs[:, 2:] - [x1, y1]) @ nrm)
        da = np.minimum(np.abs(ang - ang[i]), 180 - np.abs(ang - ang[i]))
        group = (~used) & (da < 3) & (dist[0] < 6) & (dist[1] < 6)
        used |= group
        pts = np.concatenate([segs[group, :2], segs[group, 2:]]).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        support = np.hypot(segs[group, 2] - segs[group, 0], segs[group, 3] - segs[group, 1]).sum()
        lines.append((vx, vy, x0, y0, ang[i], support))
    lines.sort(key=lambda l_: -l_[5])                       # longest painted lines first
    across = [l_[:4] for l_ in lines if min(l_[4], 180 - l_[4]) < 25]
    along = [l_[:4] for l_ in lines if min(l_[4], 180 - l_[4]) >= 25]
    return across, along


def intersect(a, b):
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = a, b
    m = np.array([[vx1, -vx2], [vy1, -vy2]])
    if abs(np.linalg.det(m)) < 1e-6:
        return None
    s = np.linalg.solve(m, [x2 - x1, y2 - y1])[0]
    return np.array([x1 + s * vx1, y1 + s * vy1])


def court_samples(step=0.1):
    pts = []
    for (x1, y1), (x2, y2) in PAINTED:
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step))
        s = np.linspace(0, 1, n)
        pts.append(np.stack([x1 + (x2 - x1) * s, y1 + (y2 - y1) * s], 1))
    return np.concatenate(pts)


SAMPLES = court_samples()


UNIT = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
MODEL_QUADS = [np.float32([[xa, ya], [xb, ya], [xa, yb], [xb, yb]])
               for ya, yb in itertools.combinations(ACROSS, 2) for xa, xb in itertools.combinations(ALONG, 2)]
# unit square -> court quad, inverted: H(court -> image) = H(unit -> image) @ MODEL_INV
MODEL_INV = np.array([np.linalg.inv(cv2.getPerspectiveTransform(UNIT, q)) for q in MODEL_QUADS])
SAMPLES_H = None


def score_many(Hs, mask):
    """score() for a batch of homographies (K, 3, 3) at once."""
    h, w = mask.shape
    p = Hs @ np.c_[SAMPLES, np.ones(len(SAMPLES))].T                      # (K, 3, N)
    z = p[:, 2]
    front = z > 1e-6
    x = np.where(front, p[:, 0] / np.where(front, z, 1), -1)
    y = np.where(front, p[:, 1] / np.where(front, z, 1), -1)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    hits = np.where(inside, mask[np.clip(y, 0, h - 1).astype(int), np.clip(x, 0, w - 1).astype(int)], 0).sum(1)
    n_in = inside.sum(1)
    prec = hits / np.maximum(n_in, 1)
    # a court squeezed into a few pixels sits "on" any line: 10 cm of court must span about a pixel or more
    step = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))
    pair = inside[:, 1:] & inside[:, :-1] & (step < 0.05 * max(h, w))
    spread = np.where(pair, step, 0).sum(1) / np.maximum(pair.sum(1), 1)
    # seen from behind a baseline: the near half is tall in the image and the sidelines are not flat
    key = Hs @ np.array([[court.CENTER_X, 0, 1], [court.CENTER_X, court.NET_Y, 1], [0, 0, 1], [0, court.NET_Y, 1],
                         [court.WIDTH, 0, 1], [court.WIDTH, court.NET_Y, 1]], float).T        # (K, 3, 6)
    kz = np.where(np.abs(key[:, 2]) > 1e-9, key[:, 2], 1e-9)
    kx, ky = key[:, 0] / kz, key[:, 1] / kz
    tall = (ky[:, 0] - ky[:, 1] >= 0.08 * h) & (key[:, 2] > 0).all(1)
    steep = [np.abs(ky[:, a] - ky[:, b]) >= np.tan(np.radians(20)) * np.abs(kx[:, a] - kx[:, b]) for a, b in ((2, 3), (4, 5))]
    near_w, net_w = kx[:, 4] - kx[:, 2], kx[:, 5] - kx[:, 3]
    wide = (near_w >= 0.25 * w) & (net_w >= 0.1 * w) & (net_w <= near_w)
    ok = (n_in >= 0.3 * len(SAMPLES)) & (spread >= 0.8) & tall & steep[0] & steep[1] & wide
    s = np.where(ok, hits * prec * prec, 0.0)
    return s, prec


def score(H, mask):
    """(hits * precision, precision): share of the projected court lines (inside the frame) on the mask."""
    s, prec = score_many(H[None], mask)
    return float(s[0]), float(prec[0])


def order_key(line, ref, axis):
    """Position of a line along an axis, measured where it crosses a reference line."""
    p = intersect(line, ref)
    return p[axis] if p is not None else np.inf


def search(mask, across, along):
    h, w = mask.shape
    mid_v = (0.0, 1.0, w / 2, 0.0)             # vertical line through the centre
    mid_h = (1.0, 0.0, 0.0, h * 0.75)           # horizontal line low in the frame
    across = sorted(across, key=lambda l_: -order_key(l_, mid_v, 1))   # bottom (near) to top (far)
    along = sorted(along, key=lambda l_: order_key(l_, mid_h, 0))      # left to right
    best = (0.0, 0.0, None)
    for ia, ja in itertools.combinations(range(len(across)), 2):
        for ib, jb in itertools.combinations(range(len(along)), 2):
            img = [intersect(across[i], along[j]) for i in (ia, ja) for j in (ib, jb)]
            if any(p is None for p in img):
                continue
            img = np.float32(img)
            if np.abs(img).max() > 10 * max(h, w):
                continue
            Hs = cv2.getPerspectiveTransform(UNIT, img) @ MODEL_INV
            s, prec = score_many(Hs, mask)
            k = int(np.argmax(s))
            if s[k] > best[0]:
                best = (float(s[k]), float(prec[k]), Hs[k])
    return best


def refine(H, mask, iters=3, band=5):
    """Fit each court line to the mask pixels next to its projection, then solve H on their intersections."""
    ys, xs = np.nonzero(mask)
    pix = np.c_[xs, ys].astype(np.float32)
    for _ in range(iters):
        fitted = {}
        for key, ((x1, y1), (x2, y2)) in enumerate(PAINTED):
            a = cv2.perspectiveTransform(np.float32([[[x1, y1], [x2, y2]]]), H)[0]
            d = a[1] - a[0]
            L = np.hypot(*d)
            if L < 20:
                continue
            d /= L
            rel = pix - a[0]
            along_ = rel @ d
            near = (np.abs(rel @ np.array([-d[1], d[0]])) < band) & (along_ > 0) & (along_ < L)
            if near.sum() < 15:
                continue
            fitted[key] = cv2.fitLine(pix[near], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        mdl, img = [], []
        for i, j in itertools.combinations(fitted, 2):
            (a1, a2), (b1, b2) = PAINTED[i], PAINTED[j]
            if (a1[1] == a2[1]) == (b1[1] == b2[1]):                  # need one line across, one along
                continue
            across_, along_ = (PAINTED[i], PAINTED[j]) if a1[1] == a2[1] else (PAINTED[j], PAINTED[i])
            x, y = along_[0][0], across_[0][1]
            if not (min(along_[0][1], along_[1][1]) <= y <= max(along_[0][1], along_[1][1])):
                continue                                              # e.g. centre line vs a baseline it never meets
            p = intersect(fitted[i], fitted[j])
            if p is not None:
                mdl.append([x, y])
                img.append(p)
        if len(mdl) < 4:
            break
        H2, _ = cv2.findHomography(np.float32(mdl), np.float32(img), 0)
        if H2 is None:
            break
        H = H2
    return H


def feet_share(H, dets, w, h, times):
    """Share of the frames with 2+ people where somebody's feet are in the near half and somebody's in the
    far half (court +-1 m): about the share of time the main view is on screen with a right calibration."""
    Hinv = np.linalg.inv(H)
    n = ok = 0
    for f in dets["frames"]:
        if round(f["t"], 1) not in times or len(f["p"]) < 2:
            continue
        feet = np.float32([[(p[0] + p[2]) / 2 * w, p[3] * h] for p in f["p"]])
        c = cv2.perspectiveTransform(feet[None], Hinv)[0]
        inx = (c[:, 0] > -1) & (c[:, 0] < court.WIDTH + 1)
        near = inx & (c[:, 1] > -1) & (c[:, 1] < court.NET_Y)
        far = inx & (c[:, 1] > court.NET_Y) & (c[:, 1] < court.LENGTH + 1)
        n += 1
        ok += bool(near.any() and far.any())
    return ok / n if n else 0.0


def heights(H, dets, w, h, times):
    """(median, share in 1.3-2.4 m) of the height people would have if this H were right: a standing
    person spans, in pixels, about as much as the same length of court laid across at their feet."""
    Hinv = np.linalg.inv(H)
    est = []
    for f in dets["frames"]:
        if round(f["t"], 1) not in times:
            continue
        for p in f["p"]:
            if p[3] > 0.985 or p[4] < 0.5:                      # feet out of frame, weak detection
                continue
            feet = np.float32([[[(p[0] + p[2]) / 2 * w, p[3] * h]]])
            x, y = cv2.perspectiveTransform(feet, Hinv)[0, 0]
            if not (-0.5 < x < court.WIDTH + 0.5 and -0.5 < y < court.LENGTH + 0.5):
                continue
            a = cv2.perspectiveTransform(np.float32([[[x - 0.5, y], [x + 0.5, y]]]), H)[0]
            px_per_m = float(np.hypot(*(a[1] - a[0])))
            if px_per_m > 1:
                est.append((p[3] - p[1]) * h / px_per_m)
    if len(est) < 20:
        return None, 0.0
    est = np.array(est)
    return float(np.median(est)), float(((est > 1.3) & (est < 2.4)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, help="default: end of video")
    ap.add_argument("--check", help="write the court drawn back from H on the reference frame")
    ap.add_argument("--dets", help="dets.json of stage 1: check that the players stand on the court")
    ap.add_argument("--min-lines", type=float, default=0.6)
    ap.add_argument("--min-feet", type=float, default=0.25)
    args = ap.parse_args()

    w, h, fps, dur = probe(args.video)
    t1 = dur if args.t1 is None else args.t1
    cap = cv2.VideoCapture(args.video)
    samples = sample_frames(cap, fps, args.t0, t1)
    dets = load_json(args.dets) if args.dets else None
    best = None
    for dark in (False, True):
        masks, lines = sample_masks(samples, dark)
        for mask, frame, frame_t, n_same in view_masks(samples, masks, lines):
            across, along = detect_lines(mask)
            _, _, H = search(mask, across[:6], along[:6])
            if H is None:
                continue
            H2 = refine(H, mask)
            if score(H2, mask)[0] >= score(H, mask)[0]:     # refinement can drift off: keep it only if better
                H = H2
            prec = score(H, mask)[1]
            # people are checked only where this court is on screen (not in close-ups or replays)
            times = {round(t, 1) for t in in_view(H, samples, masks)}
            feet = feet_share(H, dets, w, h, times) if dets else None
            tall, tall_ok = heights(H, dets, w, h, times) if dets else (None, 1.0)
            rank = prec * tall_ok
            print(f"  {'dark' if dark else 'light'} lines, view at {frame_t:.0f} s ({n_same}/{len(samples)} frames): "
                  f"{prec:.0%} on the lines, court on screen in {len(times)}/{len(samples)} frames" +
                  ("" if feet is None else f"; there, somebody in each half in {feet:.0%}, people {tall or 0:.2f} m "
                   f"tall ({tall_ok:.0%} between 1.3 and 2.4 m)"))
            if best is None or rank > best[0]:
                best = (rank, prec, feet, tall, tall_ok, H, frame, frame_t)
    if best is None:
        print("no court found")
        raise SystemExit(2)
    _, prec, feet, tall, tall_ok, H, frame, frame_t = best
    print(f"best: {prec:.0%} on the lines" + ("" if feet is None else
          f", {feet:.0%} with somebody in each half, people {tall or 0:.2f} m tall ({tall_ok:.0%} plausible)"))
    if feet is not None:      # people must have a human height and stand on the court; lines may be partly hidden
        human = 1.45 <= (tall or 0) <= 2.0
        good = prec >= args.min_lines and feet >= args.min_feet and tall_ok >= 0.7 and human
    else:
        good = prec >= 0.8

    points = {}
    for name, (x, y) in court.POINTS.items():
        p = H @ [x, y, 1.0]
        if p[2] > 1e-6 and 0 <= p[0] / p[2] < w and 0 <= p[1] / p[2] < h:
            points[name] = [round(float(p[0] / p[2]), 1), round(float(p[1] / p[2]), 1)]
    if args.check:
        Path(args.check).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.check, check_image(frame, H, points))
        print(f"check image -> {args.check}")
    if not good or len(points) < 4:
        print("not confident: calibrate by hand with s2_calibrate.py")
        raise SystemExit(2)
    H2, err = solve(points)
    calib = load_json(args.out) if Path(args.out).exists() else {"width": w, "height": h, "segments": []}
    seg = {"t0": args.t0, "t1": t1, "frame_t": round(frame_t, 2), "points": points, "H": H2.tolist(),
           "reproj_px": err, "auto": {"line_share": round(float(prec), 3), "feet_share": None if feet is None else round(feet, 3),
                    "people_height_m": None if tall is None else round(tall, 2), "height_ok": round(tall_ok, 3)}}
    calib["segments"] = sorted([s_ for s_ in calib["segments"] if s_["t0"] != args.t0] + [seg], key=lambda s_: s_["t0"])
    save_json(calib, args.out)
    print(f"{len(points)} points -> {args.out}")


if __name__ == "__main__":
    main()
