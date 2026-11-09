"""Frame access and contact sheets."""
import cv2
import numpy as np


def read_frame(cap, t, fps=30.0):
    """Frame at time t (s) from an open cv2.VideoCapture, or None."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, frame = cap.read()
    return frame if ok else None


def label(img, text, org=(6, 20), scale=0.55):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def with_zoom(frame, box, text, width=480, zoom_px=270, half=60):
    """Frame scaled to `width` + a square zoom on `box` (fractions, e.g. the far player) beside it."""
    h, w = frame.shape[:2]
    if box:
        cx, cy = (box[0] + box[2]) / 2 * w, (box[1] + box[3]) / 2 * h
    else:  # nobody tracked: zoom on the middle of the far court
        cx, cy = w * 0.4, h * 0.78
    x0, y0 = int(np.clip(cx - half, 0, w - 2 * half)), int(np.clip(cy - half, 0, h - 2 * half))
    zoom = cv2.resize(frame[y0:y0 + 2 * half, x0:x0 + 2 * half], (zoom_px, zoom_px),
                      interpolation=cv2.INTER_CUBIC)
    small = frame.copy()
    cv2.rectangle(small, (x0, y0), (x0 + 2 * half, y0 + 2 * half), (0, 140, 255), 1)
    small = cv2.resize(small, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    if small.shape[0] != zoom_px:
        zoom = cv2.resize(zoom, (int(zoom_px * small.shape[0] / zoom_px),) * 2)
    tile = np.hstack([small, zoom])
    label(tile, text, (6, 20), 0.6)
    return tile


def player_zoom(frame, box, size):
    """16:9 close-up on one player (box in fractions), resized to size = (w, h): 1.25x the box height,
    at least 135 frame rows (the far player is tiny), clamped to the frame."""
    h, w = frame.shape[:2]
    zh = min(h, max(135, 1.25 * (box[3] - box[1]) * h))
    zw = min(w, zh * 16 / 9)
    zh = zw * 9 / 16
    cx, cy = (box[0] + box[2]) / 2 * w, (box[1] + box[3]) / 2 * h
    x0, y0 = int(np.clip(cx - zw / 2, 0, w - zw)), int(np.clip(cy - zh / 2, 0, h - zh))
    return cv2.resize(frame[y0:y0 + int(zh), x0:x0 + int(zw)], size, interpolation=cv2.INTER_CUBIC)


def contact_sheet(images, cols=4, width=424):
    """Tile images (all same size) into one grid image, each scaled to `width`."""
    h0, w0 = images[0].shape[:2]
    tile_h = int(round(h0 * width / w0))
    tiles = [cv2.resize(im, (width, tile_h), interpolation=cv2.INTER_AREA) for im in images]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)
