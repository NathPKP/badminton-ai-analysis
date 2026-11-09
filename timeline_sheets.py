"""Contact sheets of the whole match for reading serves and rally ends by eye (stages 4 and 6).

Usage: python timeline_sheets.py <proxy.mp4> <players.json> <out_dir> [--start 12] [--end END]
           [--step 0.5] [--per-sheet 12]

Each tile = the full frame + a zoom on the far player (orange square), labelled with the time.
Sheets are named timeline_<start>-<end>.jpg (seconds).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from bball.io import load_json
from bball.video import contact_sheet, read_frame, with_zoom
from s4_serves import far_box


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("players")
    ap.add_argument("out_dir")
    ap.add_argument("--start", type=float, default=12.0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--per-sheet", type=int, default=12)
    args = ap.parse_args()

    frames = load_json(args.players)["frames"]
    cap = cv2.VideoCapture(args.video)
    end = args.end or cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) - 0.1
    times = np.round(np.arange(args.start, end, args.step), 2)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(times), args.per_sheet):
        chunk = times[i:i + args.per_sheet]
        tiles = [with_zoom(read_frame(cap, t), far_box(frames, t), f"t={t:.1f}s") for t in chunk]
        name = out / f"timeline_{chunk[0]:05.1f}-{chunk[-1]:05.1f}.jpg"
        cv2.imwrite(str(name), contact_sheet(tiles, cols=2, width=tiles[0].shape[1]))
    print(f"{len(times)} frames in {int(np.ceil(len(times) / args.per_sheet))} sheets -> {out}")


if __name__ == "__main__":
    main()
