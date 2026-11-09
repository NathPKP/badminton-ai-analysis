"""Stage 7: movement per rally (distance, speed, court zones, trips out of position, heatmaps).

Usage: python s7_movement.py <players.json> <points.json> <motion.json> [--figures work/figures]

Each player is measured in HIS OWN half, so both compare directly:
  depth  = distance from his own baseline (0 = baseline, 6.7 = net)
  across = distance from his own left sideline
Window per rally: serve + 0.3 s -> rally end. Positions are de-spiked (rolling median) and
smoothed over 0.7 s before summing distance (jitter otherwise adds ~30 %). The far player's
depth is ~5x less precise (low camera), so his median window is wider; with this video it is
also biased toward the net (see "notes" in the output): use his across position only.

Output: {"points": [{"n", "winner", "rdur", "R": stats, "O": stats}], "summary": {...}}
"""
import argparse
from pathlib import Path

import numpy as np

from bball import court
from bball.io import load_json, save_json
from bball.track import positions, rolling_median

BASE = (court.CENTER_X, 3.4)      # base point in own-half coordinates (across, depth)
TRIP_M = 2.2
BACK_M = 1.6
YOU, OPP = "#2a78d6", "#eb6834"   # categorical slots 1-2 (reference palette, validated pair)
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d4"


def interp_gaps(P, max_gap):
    """Linear interpolation over NaN runs of at most max_gap frames."""
    P = P.copy()
    ok = ~np.isnan(P[:, 0])
    idx = np.where(ok)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            w = np.linspace(0, 1, b - a + 1)[1:-1, None]
            P[a + 1:b] = P[a] * (1 - w) + P[b] * w
    return P


def moving_mean(P, window):
    """NaN-aware centred moving average."""
    half = window // 2
    out = np.full_like(P, np.nan)
    for i in range(len(P)):
        s = P[max(0, i - half):i + half + 1]
        ok = ~np.isnan(s[:, 0])
        if ok.sum() >= max(1, half) and not np.isnan(P[i, 0]):
            out[i] = s[ok].mean(axis=0)
    return out


def own_half(P, key):
    """Court metres -> (across, depth) in the player's own half."""
    if key == "r":
        return P.copy()
    return np.stack([court.WIDTH - P[:, 0], court.LENGTH - P[:, 1]], axis=1)


def rally_stats(P, t, fps, t0, t1):
    m = (t >= t0) & (t <= t1)
    Q = P[m]
    tt = t[m]
    ok = ~np.isnan(Q[:, 0])
    n = len(Q)
    if ok.sum() < 3:
        return {"coverage": round(float(ok.mean()) if n else 0.0, 2)}
    steps = np.hypot(*(Q[1:] - Q[:-1]).T)
    both = ok[1:] & ok[:-1]
    dist = float(np.nansum(steps[both]))
    live = both.sum() / fps
    sp = np.full(n, np.nan)
    sp[2:-2] = np.hypot(*(Q[4:] - Q[:-4]).T) / (4 / fps)
    d_base = np.hypot(Q[:, 0] - BASE[0], Q[:, 1] - BASE[1])
    trips = []
    i = 0
    while i < n:
        if ok[i] and d_base[i] >= TRIP_M:
            j = i
            while j + 1 < n and ok[j + 1] and d_base[j + 1] >= TRIP_M:
                j += 1
            k = i + int(np.nanargmax(d_base[i:j + 1]))
            later = np.argmin(np.abs(t - (tt[k] + 1.0)))
            rec = float(np.hypot(P[later, 0] - BASE[0], P[later, 1] - BASE[1]))
            trips.append({"t": round(float(tt[k]), 2), "peak_m": round(float(d_base[k]), 2),
                          "after_1s_m": None if np.isnan(rec) else round(rec, 2)})
            i = j + 1
        else:
            i += 1
    q = Q[ok]
    return {"coverage": round(float(ok.mean()), 2), "distance_m": round(dist, 2),
            "avg_speed": round(dist / live, 2) if live > 0 else None,
            "peak_speed": round(float(np.nanmax(sp)), 2) if np.isfinite(sp).any() else None,
            "back_share": round(float((q[:, 1] <= BACK_M).mean()), 2),
            "front_share": round(float((q[:, 1] >= court.SHORT_SERVICE_Y[0]).mean()), 2),
            "median_pos": [round(float(v), 2) for v in np.median(q, axis=0)],
            "trips": trips, "live_s": round(float(live), 2)}


def aggregate(rows):
    """Sum distance / time over rallies; average the shares weighted by live time."""
    rows = [r for r in rows if r.get("distance_m") is not None]
    if not rows:
        return None
    live = sum(r["live_s"] for r in rows)
    dist = sum(r["distance_m"] for r in rows)
    trips = [tr for r in rows for tr in r["trips"]]
    rec = [tr["after_1s_m"] for tr in trips if tr["after_1s_m"] is not None]
    return {"rallies": len(rows), "live_s": round(live, 1), "distance_m": round(dist, 1),
            "avg_speed": round(dist / live, 2) if live else None,
            "peak_speed": max(r["peak_speed"] or 0 for r in rows),
            "back_share": round(sum(r["back_share"] * r["live_s"] for r in rows) / live, 2),
            "front_share": round(sum(r["front_share"] * r["live_s"] for r in rows) / live, 2),
            "trips": len(trips), "trips_per_10s": round(10 * len(trips) / live, 2) if live else None,
            "median_dist_1s_after_trip": round(float(np.median(rec)), 2) if rec else None}


def draw_half(ax, title):
    """Own half seen from above: baseline at the bottom, net at the top."""
    lines = [((0, 0), (court.WIDTH, 0)), ((0, court.DOUBLES_LONG_SERVICE_Y[0]), (court.WIDTH, court.DOUBLES_LONG_SERVICE_Y[0])),
             ((0, court.SHORT_SERVICE_Y[0]), (court.WIDTH, court.SHORT_SERVICE_Y[0])),
             ((court.CENTER_X, 0), (court.CENTER_X, court.SHORT_SERVICE_Y[0]))]
    lines += [((x, 0), (x, court.NET_Y)) for x in (0, court.SINGLES_X[0], court.SINGLES_X[1], court.WIDTH)]
    for (x1, y1), (x2, y2) in lines:
        ax.plot([x1, x2], [y1, y2], color=INK2, lw=0.8, zorder=3)
    ax.plot([0, court.WIDTH], [court.NET_Y, court.NET_Y], color=INK, lw=2, zorder=3)
    ax.plot(*BASE, marker="+", color=INK, ms=10, mew=1.5, zorder=4)
    ax.set_xlim(-0.6, court.WIDTH + 0.6)
    ax.set_ylim(-0.8, court.NET_Y + 0.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, color=INK, fontsize=11, loc="left")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("players")
    ap.add_argument("points")
    ap.add_argument("out")
    ap.add_argument("--figures", default="work/figures")
    ap.add_argument("--smooth", type=float, default=0.7, help="moving-average window (s)")
    args = ap.parse_args()

    data = load_json(args.players)
    frames, fps = data["frames"], data["fps"]
    t = np.array([f["t"] for f in frames])
    gap = int(fps)                                   # interpolate holes up to 1 s
    win = int(round(args.smooth * fps)) | 1
    P = {"r": moving_mean(rolling_median(interp_gaps(positions(frames, "r"), gap), 5), win),
         "o": moving_mean(rolling_median(interp_gaps(positions(frames, "o"), gap), 9), win)}
    H = {k: own_half(v, k) for k, v in P.items()}

    pts = load_json(args.points)["points"]
    out = []
    for p in pts:
        if not p["r1"]:
            continue
        rec = {"n": p["n"], "t": p["t"], "winner": p["winner"], "rdur": p["rdur"]}
        for key, name in (("r", "R"), ("o", "O")):
            rec[name] = rally_stats(H[key], t, fps, p["r0"] + 0.3, p["r1"])
        out.append(rec)

    def bucket(d):
        return "<=5s" if d <= 5 else "5-10s" if d <= 10 else ">10s"
    summary = {}
    for name in ("R", "O"):
        rows = [r[name] for r in out]
        summary[name] = {"all": aggregate(rows),
                         "you_won": aggregate([r[name] for r in out if r["winner"] == "R"]),
                         "you_lost": aggregate([r[name] for r in out if r["winner"] == "O"]),
                         **{b: aggregate([r[name] for r in out if bucket(r["rdur"]) == b])
                            for b in ("<=5s", "5-10s", ">10s")}}
    notes = {"O_depth": "unreliable: with only far-court points on the far short service line, the far "
                        "player's feet map ~0.5-1 m too close to the net and his depth range is squashed. "
                        "Do not use his back/front shares, trips or heatmap depth; his across position is fine."}
    save_json({"base": BASE, "trip_m": TRIP_M, "notes": notes, "points": out, "summary": summary}, args.out)

    print(f"{len(out)} rallies -> {args.out}")
    print("  #   rally  won  | YOU dist  avg  peak back front trips | OPP dist  avg  peak back front trips")
    for r in out:
        a, b = r["R"], r["O"]
        f = lambda s: (f"{s['distance_m']:5.1f} {s['avg_speed']:4.2f} {s['peak_speed']:5.2f} "
                       f"{s['back_share']:4.0%} {s['front_share']:5.0%} {len(s['trips']):5d}") if "distance_m" in s else "   (no data)"
        print(f"  {r['n']:2d} {r['rdur']:5.2f}s  {'you' if r['winner'] == 'R' else 'opp'}  | {f(a)} | {f(b)}")
    for name, label in (("R", "YOU"), ("O", "OPP")):
        for part in ("all", "you_won", "you_lost"):
            s = summary[name][part]
            if s:
                print(f"  {label} {part:8s}: {s['rallies']} rallies, {s['distance_m']} m in {s['live_s']} s "
                      f"-> {s['avg_speed']} m/s avg, peak {s['peak_speed']}, back {s['back_share']:.0%}, "
                      f"front {s['front_share']:.0%}, {s['trips']} trips out of position "
                      f"({s['trips_per_10s']}/10 s), median {s['median_dist_1s_after_trip']} m from base 1 s later")

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # heatmaps: time spent per 0.5 m cell during live play, each player in his own half
    blues = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    oranges = LinearSegmentedColormap.from_list("orange", ["#fcfcfb", "#fbe0d3", "#f4a27f", "#eb6834", "#b9471a", "#6e2809"])
    live = np.zeros(len(t), bool)
    for p in pts:
        if p["r1"]:
            live |= (t >= p["r0"] + 0.3) & (t <= p["r1"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.4))
    bins = [np.arange(-0.5, court.WIDTH + 0.51, 0.5), np.arange(-0.5, court.NET_Y + 0.01, 0.5)]
    for ax, key, cmap, title in ((axes[0], "r", blues, "You"),
                                 (axes[1], "o", oranges, "Opponent (depth NOT reliable)")):
        q = H[key][live]
        q = q[~np.isnan(q[:, 0])]
        h, xe, ye = np.histogram2d(q[:, 0], q[:, 1], bins=bins)
        h = h / fps
        ax.pcolormesh(xe, ye, np.ma.masked_equal(h.T, 0), cmap=cmap, vmin=0, zorder=1)
        draw_half(ax, f"{title}: {len(q) / fps:.0f} s of live play")
        ax.text(court.WIDTH / 2, -0.55, "own baseline", ha="center", va="center", color=INK2, fontsize=8)
        ax.text(court.WIDTH / 2, court.NET_Y + 0.15, "net", ha="center", va="bottom", color=INK2, fontsize=8)
    fig.suptitle("Where each player spends live-play time (darker = longer; + = base point)",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "s7_heatmaps.png", dpi=130)
    plt.close(fig)

    # paths: one small court per rally, both players, full court view (camera at the bottom)
    n = len(out)
    cols = 6
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(2.0 * cols, 3.9 * rows_n))
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    for ax, r in zip(np.ravel(axes), out):
        p = next(q for q in pts if q["n"] == r["n"])
        m = (t >= p["r0"] + 0.3) & (t <= p["r1"])
        for a, b in court.court_lines():
            ax.plot([a[0], b[0]], [a[1], b[1]], color=GRID, lw=0.7, zorder=1)
        ax.plot([0, court.WIDTH], [court.NET_Y] * 2, color=INK2, lw=1.5, zorder=1)
        for key, col in (("r", YOU), ("o", OPP)):
            q = P[key][m]
            ax.plot(q[:, 0], q[:, 1], color=col, lw=2, zorder=3, solid_capstyle="round")
            first = q[~np.isnan(q[:, 0])][:1]
            if len(first):
                ax.plot(*first[0], "o", color=col, ms=8, mec=SURFACE, mew=2, zorder=4)
        ax.set_xlim(-0.8, court.WIDTH + 0.8)
        ax.set_ylim(-0.8, court.LENGTH + 0.8)
        ax.set_aspect("equal")
        ax.axis("off")
        who = "you" if r["winner"] == "R" else "opp"
        ax.set_title(f"#{r['n']}  {r['t']:.0f} s  {r['rdur']:.1f} s\nwon by {who}", fontsize=9, color=INK, loc="left")
    fig.legend(handles=[plt.Line2D([], [], color=YOU, lw=2, label="You (near, bottom)"),
                        plt.Line2D([], [], color=OPP, lw=2, label="Opponent (far, top)"),
                        plt.Line2D([], [], color=INK2, marker="o", lw=0, label="position at serve")],
               loc="upper left", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_dir / "s7_paths.png", dpi=130)
    plt.close(fig)
    print(f"figures -> {fig_dir / 's7_heatmaps.png'}, {fig_dir / 's7_paths.png'}")
    print(f"NOTE opponent depth: {notes['O_depth']}")


if __name__ == "__main__":
    main()
