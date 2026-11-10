"""Shuttle flights from the per-frame track of s13_shuttle.py, and the serve flight near a given time."""
import numpy as np


def flights(p, fps, t0, w, h, min_frames=5, min_move=0.03, max_gap=4, max_jump=0.12, shots=None):
    """Group per-frame positions into moving flights.

    p: list of [x, y, how] (fractions of the frame) or None, one per frame from t0.
    Detections that come back to the same spot (within 0.5 % of the width) at three other moments more than
    1 s apart are dropped first: a white climbing hold, a logo, a shuttle lying on the floor. The rest is cut
    into runs (gaps <= max_gap frames, jumps <= max_jump x width per frame, never across a cut when `shots`, the
    shot number of each frame, is given), kept when they last at least
    min_frames frames and travel at least min_move x width.
    Returns [{"t0", "t1", "n", "a": [x, y], "b": [x, y]}, ...] in time order (a, b: first and last point).
    """
    pts = {k: (x[0] * w, x[1] * h) for k, x in enumerate(p) if x}
    seen = np.array([[k, *pts[k]] for k in pts if p[k][2] == 1]).reshape(-1, 3)
    static = set()
    near = 0.005 * w
    for k, x, y in seen:
        d = np.hypot(seen[:, 1] - x, seen[:, 2] - y)
        if ((d < near) & (np.abs(seen[:, 0] - k) > fps)).sum() >= 3:
            static.add(int(k))
    runs, cur = [], []
    for k in sorted(k for k in pts if k not in static):
        if cur:
            gap = k - cur[-1]
            jump = np.hypot(pts[k][0] - pts[cur[-1]][0], pts[k][1] - pts[cur[-1]][1])
            if gap > max_gap or jump > max_jump * w * gap or (shots is not None and shots[k] != shots[cur[-1]]):
                runs.append(cur)
                cur = []
        cur.append(k)
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        if len(r) < min_frames:
            continue
        a = pts[r[0]]
        if max(np.hypot(pts[k][0] - a[0], pts[k][1] - a[1]) for k in r) < min_move * w:
            continue
        out.append({"t0": round(t0 + r[0] / fps, 2), "t1": round(t0 + r[-1] / fps, 2), "n": len(r),
                    "a": [round(pts[r[0]][0] / w, 4), round(pts[r[0]][1] / h, 4)],
                    "b": [round(pts[r[-1]][0] / w, 4), round(pts[r[-1]][1] / h, 4)]})
    return out


def rally_groups(fl, gap=2.5, min_len=1.5):
    """Rallies as runs of flights less than `gap` s apart, kept when they last `min_len` s or more:
    [(start, end), ...]. On a BWF final (11 rallies): 11 found plus 1 rally split in two by a long lob, and ends
    within 0.5 s of the eye reading 9 times out of 10. Between rallies the shuttle is hit back to the server in
    one flight, usually a separate short group; a rally end read from it is late."""
    out = []
    for f in fl:
        if out and f["t0"] - out[-1][1] <= gap:
            out[-1][1] = f["t1"]
        else:
            out.append([f["t0"], f["t1"]])
    return [(a, b) for a, b in out if b - a >= min_len]


def serve_flight(fl, t, window=2.5, quiet=1.5):
    """Start time of the flight that most likely is the serve near time t, or None.

    A serve flight starts within `window` s of t and after at least `quiet` s without any flight (players
    getting ready); this rules out the shuttle being hit back to the server after the previous rally.
    On the floor-level test clip this put the automatic serve times within 0.7 s of the contact (median
    0.2 s), against 2.3 s (median 0.7 s) from the players' stillness alone.
    """
    best = None
    for i, f in enumerate(fl):
        prev_end = fl[i - 1]["t1"] if i else -1e9
        if abs(f["t0"] - t) <= window and f["t0"] - prev_end >= quiet:
            if best is None or abs(f["t0"] - t) < abs(best - t):
                best = f["t0"]
    return best
