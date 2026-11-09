"""Position smoothing and speed from players.json tracks."""
import numpy as np


def positions(frames, key):
    """(N, 2) array of court positions for 'r' or 'o', NaN where missing."""
    return np.array([f[key] if f[key] else [np.nan, np.nan] for f in frames], dtype=float)


def rolling_median(P, window, min_valid=None):
    """Median over a centred window of `window` frames, ignoring NaN; robust to single-frame jumps."""
    half = window // 2
    min_valid = min_valid or max(1, window // 2 + 1)
    out = np.full_like(P, np.nan)
    for i in range(len(P)):
        s = P[max(0, i - half):i + half + 1]
        ok = ~np.isnan(s[:, 0])
        if ok.sum() >= min_valid:
            out[i] = np.median(s[ok], axis=0)
    return out


def speed(P, fps, lag=2):
    """Central-difference speed (m/s) over +-lag frames."""
    v = np.full(len(P), np.nan)
    v[lag:-lag] = np.hypot(*(P[2 * lag:] - P[:-2 * lag]).T) / (2 * lag / fps)
    return v
