"""Stage 0: make the 30 fps SDR working copy that every later stage reads.

Usage: python s0_proxy.py <input video> <output proxy.mp4> [--height 720] [--tonemap]
"""
import argparse
import subprocess

from bball.io import probe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--height", type=int, default=720,
                    help="target height; videos already smaller are kept at native size (no upscaling)")
    ap.add_argument("--tonemap", action="store_true", help="tone-map HDR (HLG/PQ, e.g. iPhone) to SDR")
    args = ap.parse_args()

    w, h, fps, dur = probe(args.video)
    print(f"input : {w}x{h}  {fps:.2f} fps  {dur:.1f} s")

    filters = []
    if args.tonemap:
        filters.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                       "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
    if h > args.height:
        filters.append(f"scale=-2:{args.height}")
    filters.append("fps=30")

    cmd = ["ffmpeg", "-v", "error", "-stats", "-y", "-i", args.video,
           "-vf", ",".join(filters), "-an",
           "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-g", "30",  # keyframe every second: fast seeking for contact sheets
           args.out]
    subprocess.run(cmd, check=True)

    w, h, fps, dur = probe(args.out)
    print(f"proxy : {w}x{h}  {fps:.2f} fps  {dur:.1f} s  -> {args.out}")


if __name__ == "__main__":
    main()
