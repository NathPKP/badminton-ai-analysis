"""Stage 2: court calibration (homography court metres <-> pixels), one per camera position.

Interactive (a window opens on one frame):
  python s2_calibrate.py <proxy.mp4> <calib.json> [--t 0.0] [--t0 0] [--t1 end] [--check out.jpg]

  1. Press the letter of a court point listed on the right (it turns red in the list).
  2. Click on that point in the image. Right click removes the last point.
     To click precisely: use the magnifier button of the toolbar to zoom, then click the
     magnifier again to switch it off before clicking a point. The house button resets the zoom.
  3. Space toggles the contrast boost. Enter (at least 4 points) computes H and closes.

Recompute only (no window), from the points already saved in calib.json:
  python s2_calibrate.py <proxy.mp4> <calib.json> --recompute

Far-court depth: when every clicked point lies at or before the far short service line, depth
beyond it is extrapolated and badly conditioned (a low camera makes it worse): click far-court
points (far long service line, far baseline corners) whenever they are visible.

Output: {"width": W, "height": H, "segments": [{"t0", "t1", "frame_t", "points": {name: [px, py]},
         "H": 3x3 metres -> pixels, "reproj_px": {name: error}}]}
"""
import argparse
import string
from pathlib import Path

import cv2
import numpy as np

from bball import court
from bball.io import load_json, probe, save_json
from bball.video import label, read_frame

KEYS = string.ascii_lowercase
NAMES = list(court.POINTS)[:len(KEYS)]


def enhance(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(4.0, (8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def click_points(frame, initial=None):
    import matplotlib
    import matplotlib.pyplot as plt
    for k in list(matplotlib.rcParams):
        if k.startswith("keymap."):
            matplotlib.rcParams[k] = []
    matplotlib.rcParams["keymap.home"] = ["home"]

    images = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), cv2.cvtColor(enhance(frame), cv2.COLOR_BGR2RGB)]
    state = {"sel": None, "pts": dict(initial or {}), "order": list((initial or {}).keys()),
             "img": 1, "done": False}

    fig = plt.figure(figsize=(15, 7.5))
    ax = fig.add_axes([0.01, 0.03, 0.74, 0.92])
    side = fig.add_axes([0.76, 0.03, 0.23, 0.92])
    side.axis("off")
    im = ax.imshow(images[state["img"]])
    ax.set_xticks([])
    ax.set_yticks([])
    artists = []

    def redraw():
        for a in artists:
            a.remove()
        artists.clear()
        for n, (x, y) in state["pts"].items():
            artists.append(ax.plot(x, y, "+", color="red", ms=14, mew=1.5)[0])
            artists.append(ax.text(x + 4, y - 4, KEYS[NAMES.index(n)], color="yellow", fontsize=11,
                                   fontweight="bold"))
        side.clear()
        side.axis("off")
        for i, n in enumerate(NAMES):
            done = "x" if n in state["pts"] else " "
            col = "red" if n == state["sel"] else ("green" if n in state["pts"] else "black")
            side.text(0, 1 - i * 0.038, f"[{done}] {KEYS[i]}  {n}", color=col, fontsize=9,
                      family="monospace", transform=side.transAxes)
        side.text(0, 1 - len(NAMES) * 0.038 - 0.02,
                  "letter = choose point, click = place\nright click = undo, space = contrast\n"
                  "Enter = done (4+ points)", fontsize=9, transform=side.transAxes)
        sel = state["sel"] or "press a letter"
        ax.set_title(f"{len(state['pts'])} points  |  selected: {sel}", fontsize=11)
        fig.canvas.draw_idle()

    def on_key(ev):
        if ev.key in ("enter", "return"):
            if len(state["pts"]) >= 4:
                state["done"] = True
                plt.close(fig)
            else:
                ax.set_title("need at least 4 points", color="red")
                fig.canvas.draw_idle()
        elif ev.key == " ":
            state["img"] = 1 - state["img"]
            im.set_data(images[state["img"]])
            fig.canvas.draw_idle()
        elif ev.key and len(ev.key) == 1 and ev.key in KEYS[:len(NAMES)]:
            state["sel"] = NAMES[KEYS.index(ev.key)]
            redraw()

    def on_click(ev):
        if ev.inaxes is not ax or fig.canvas.toolbar is not None and fig.canvas.toolbar.mode:
            return
        if ev.button == 3 and state["order"]:
            state["pts"].pop(state["order"].pop(), None)
        elif ev.button == 1 and state["sel"]:
            n = state["sel"]
            if n in state["order"]:
                state["order"].remove(n)
            state["pts"][n] = [round(float(ev.xdata), 1), round(float(ev.ydata), 1)]
            state["order"].append(n)
            state["sel"] = None
        redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    redraw()
    plt.show()
    if not state["done"]:
        raise SystemExit("window closed without Enter: nothing saved")
    return state["pts"]


def solve(points):
    names = list(points)
    src = np.array([court.POINTS[n] for n in names], dtype=np.float64)
    dst = np.array([points[n] for n in names], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        raise SystemExit("findHomography failed: are 3 of the points on one line?")
    proj = court.to_px(H, src)
    err = {n: round(float(np.linalg.norm(p - d)), 2) for n, p, d in zip(names, proj, dst)}
    return H, err


def check_image(frame, H, points):
    img = court.draw_court(frame.copy(), H, (0, 255, 255), 1)
    for n, (x, y) in points.items():
        cv2.drawMarker(img, (int(round(x)), int(round(y))), (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    label(big, "court drawn back from H (yellow) / clicked points (red)", (10, 30), 0.8)
    return big


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--t", type=float, default=0.0, help="time (s) of the frame to click on")
    ap.add_argument("--t0", type=float, default=0.0, help="start of the range this camera position covers")
    ap.add_argument("--t1", type=float, help="end of that range (default: end of video)")
    ap.add_argument("--recompute", action="store_true", help="reuse the saved points, no window")
    ap.add_argument("--check", default="work/contact/s2_calib_check.jpg")
    args = ap.parse_args()

    w, h, fps, dur = probe(args.video)
    t1 = dur if args.t1 is None else args.t1
    calib = load_json(args.out) if Path(args.out).exists() else {"width": w, "height": h, "segments": []}
    old = next((s for s in calib["segments"] if s["t0"] == args.t0), None)

    cap = cv2.VideoCapture(args.video)
    if args.recompute:
        if old is None:
            raise SystemExit(f"no saved segment starting at t0={args.t0} in {args.out}")
        frame = read_frame(cap, old["frame_t"], fps)
        points, frame_t, t1 = old["points"], old["frame_t"], old["t1"]
    else:
        frame = read_frame(cap, args.t, fps)
        points = click_points(frame, old["points"] if old else None)
        frame_t = args.t

    H, err = solve(points)
    seg = {"t0": args.t0, "t1": t1, "frame_t": frame_t, "points": points,
           "H": H.tolist(), "reproj_px": err}
    calib["segments"] = sorted([s for s in calib["segments"] if s["t0"] != args.t0] + [seg],
                               key=lambda s: s["t0"])
    save_json(calib, args.out)

    Path(args.check).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.check, check_image(frame, H, points))
    print(f"{len(points)} points, range {args.t0:.1f}-{t1:.1f} s -> {args.out}")
    for n, e in err.items():
        print(f"  {n:28s} reprojection error {e:5.1f} px")
    if len(points) == 4:
        print("  (with exactly 4 points the errors are always 0: only the drawn court can tell if it is right)")
    print(f"check image -> {args.check}")


if __name__ == "__main__":
    main()
