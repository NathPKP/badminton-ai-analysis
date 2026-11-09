"""Download the four model files into models/ (about 420 MB; files already there are skipped).

Usage: python download_models.py [--dest models] [--skip-voice]

  yolox_s.onnx                person detection (stage 1)       Megvii YOLOX, Apache-2.0
  pose_landmarker_heavy.task  body pose (stage 8)              Google MediaPipe, Apache-2.0
  kokoro-v1.0.onnx            text-to-speech (stage 11 only)   Kokoro-82M via kokoro-onnx, Apache-2.0
  voices-v1.0.bin             voices for kokoro (stage 11 only)
"""
import argparse
import urllib.request
from pathlib import Path

MODELS = {
    "yolox_s.onnx": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                                  "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}
VOICE = ("kokoro-v1.0.onnx", "voices-v1.0.bin")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parent / "models"))
    ap.add_argument("--skip-voice", action="store_true", help="skip the text-to-speech files (no video reel)")
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        out = dest / name
        if args.skip_voice and name in VOICE:
            continue
        if out.exists() and out.stat().st_size > 0:
            print(f"{name}: already there")
            continue
        print(f"{name}: downloading...", flush=True)
        part = out.with_suffix(out.suffix + ".part")
        urllib.request.urlretrieve(url, part)
        part.replace(out)
        print(f"{name}: {out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
