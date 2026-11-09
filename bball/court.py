"""Badminton court geometry and homography helpers.

Court coordinates in metres, seen from the camera behind the near baseline:
  x across: 0 = left doubles sideline (camera view), 6.1 = right doubles sideline
  y along : 0 = near baseline, 6.7 = net, 13.4 = far baseline
"""
import numpy as np

WIDTH = 6.1
LENGTH = 13.4
NET_Y = 6.7
CENTER_X = 3.05
SINGLES_X = (0.46, 5.64)
SHORT_SERVICE_Y = (4.72, 8.68)          # near, far
DOUBLES_LONG_SERVICE_Y = (0.76, 12.64)  # near, far

# Named points that can be clicked during calibration (stage 2).
POINTS = {
    "near_left_corner": (0.0, 0.0),
    "near_right_corner": (WIDTH, 0.0),
    "near_center_baseline": (CENTER_X, 0.0),
    "near_left_singles_baseline": (SINGLES_X[0], 0.0),
    "near_right_singles_baseline": (SINGLES_X[1], 0.0),
    "near_T": (CENTER_X, SHORT_SERVICE_Y[0]),
    "near_short_left": (0.0, SHORT_SERVICE_Y[0]),
    "near_short_right": (WIDTH, SHORT_SERVICE_Y[0]),
    "near_short_left_singles": (SINGLES_X[0], SHORT_SERVICE_Y[0]),
    "near_short_right_singles": (SINGLES_X[1], SHORT_SERVICE_Y[0]),
    "net_left_post": (0.0, NET_Y),
    "net_right_post": (WIDTH, NET_Y),
    "far_T": (CENTER_X, SHORT_SERVICE_Y[1]),
    "far_short_left": (0.0, SHORT_SERVICE_Y[1]),
    "far_short_right": (WIDTH, SHORT_SERVICE_Y[1]),
    "far_short_left_singles": (SINGLES_X[0], SHORT_SERVICE_Y[1]),
    "far_short_right_singles": (SINGLES_X[1], SHORT_SERVICE_Y[1]),
    "far_long_service_left": (0.0, DOUBLES_LONG_SERVICE_Y[1]),
    "far_long_service_right": (WIDTH, DOUBLES_LONG_SERVICE_Y[1]),
    "far_long_service_center": (CENTER_X, DOUBLES_LONG_SERVICE_Y[1]),
    "far_left_corner": (0.0, LENGTH),
    "far_right_corner": (WIDTH, LENGTH),
    "far_center_baseline": (CENTER_X, LENGTH),
    "far_left_singles_baseline": (SINGLES_X[0], LENGTH),
    "far_right_singles_baseline": (SINGLES_X[1], LENGTH),
}


def court_lines():
    """All court line segments as ((x1, y1), (x2, y2)) in metres."""
    segs = []
    for x in (0.0, SINGLES_X[0], SINGLES_X[1], WIDTH):          # sidelines
        segs.append(((x, 0.0), (x, LENGTH)))
    for y in (0.0, DOUBLES_LONG_SERVICE_Y[0], SHORT_SERVICE_Y[0],
              SHORT_SERVICE_Y[1], DOUBLES_LONG_SERVICE_Y[1], LENGTH):
        segs.append(((0.0, y), (WIDTH, y)))
    segs.append(((CENTER_X, 0.0), (CENTER_X, SHORT_SERVICE_Y[0])))      # near centre line
    segs.append(((CENTER_X, SHORT_SERVICE_Y[1]), (CENTER_X, LENGTH)))   # far centre line
    segs.append(((0.0, NET_Y), (WIDTH, NET_Y)))                         # net (on the floor)
    return segs


def draw_court(img, H, color=(0, 255, 255), thickness=1, samples=60):
    """Draw every court line from H (metres -> pixels) onto img.

    Lines are sampled and only the parts in front of the camera (w > 0) are drawn,
    so a near baseline behind a low camera does not produce wild lines.
    """
    import cv2
    h, w = img.shape[:2]
    for (x1, y1), (x2, y2) in court_lines():
        s = np.linspace(0.0, 1.0, samples)
        pts = np.stack([x1 + (x2 - x1) * s, y1 + (y2 - y1) * s, np.ones_like(s)], 1) @ H.T
        ok = pts[:, 2] > 1e-6
        px = pts[:, :2] / np.where(ok, pts[:, 2], 1.0)[:, None]
        ok &= (np.abs(px[:, 0]) < 4 * w) & (np.abs(px[:, 1]) < 4 * h)
        for a, b, oa, ob in zip(px[:-1], px[1:], ok[:-1], ok[1:]):
            if oa and ob:
                cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, thickness, cv2.LINE_AA)
    return img


def minimap(points, scale=7, margin=6):
    """Top-down court image; points = [((x, y), (b, g, r)), ...] in metres."""
    import cv2
    w, h = int(WIDTH * scale) + 2 * margin, int(LENGTH * scale) + 2 * margin
    img = np.full((h, w, 3), 60, np.uint8)

    def px(x, y):  # far baseline at the top, like the camera view
        return int(margin + x * scale), int(margin + (LENGTH - y) * scale)
    for a, b in court_lines():
        cv2.line(img, px(*a), px(*b), (230, 230, 230), 1)
    for (x, y), col in points:
        if x is not None:
            cv2.circle(img, px(x, y), 3, col, -1)
    return img


def load_calib(path):
    """Return a function t -> H (metres -> pixels) or None when t is outside every calibrated range."""
    from bball.io import load_json
    segs = [(s["t0"], s["t1"], np.array(s["H"], dtype=np.float64)) for s in load_json(path)["segments"]]

    def H_at(t):
        for t0, t1, H in segs:
            if t0 <= t <= t1:
                return H
        return None
    return H_at


def to_px(H, pts_m):
    """Court metres (N, 2) -> pixels (N, 2) with H (metres -> pixels)."""
    pts = np.asarray(pts_m, dtype=np.float64).reshape(-1, 1, 2)
    import cv2
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def to_court(H, pts_px):
    """Pixels (N, 2) -> court metres (N, 2) with inv(H)."""
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
    import cv2
    return cv2.perspectiveTransform(pts, np.linalg.inv(H)).reshape(-1, 2)
