"""Download the model files into models/ (about 420 MB; files already there are skipped).

Usage: python download_models.py [--dest models] [--skip-voice] [--shuttle]

  yolox_s.onnx                person detection (stage 1)       Megvii YOLOX, Apache-2.0
  pose_landmarker_heavy.task  body pose (stage 8)              Google MediaPipe, Apache-2.0
  kokoro-v1.0.onnx            text-to-speech (stage 11 only)   Kokoro-82M via kokoro-onnx, Apache-2.0
  voices-v1.0.bin             voices for kokoro (stage 11 only)

--shuttle also sets up the shuttle tracker (stage 13, optional):
  tracknet.onnx, inpaintnet.onnx   TrackNetV3 (qaz812345/TrackNetV3), MIT, weights included
The published checkpoints are PyTorch files (130 MB zip on Google Drive). They are converted once to ONNX,
which needs PyTorch and onnx in the environment, only for this step:
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install onnx
"""
import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

MODELS = {
    "yolox_s.onnx": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                                  "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}
VOICE = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
TRACKNET_ZIP = ("https://drive.usercontent.google.com/download?id=1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"
                "&export=download&confirm=t")
TRACKNET_SHA256 = "8de68d34ea2457e368e22947f1309143d60a6a7f9ddccd31cd526d4c4571bcc4"
TRACKNET_FILES = {"ckpts/TrackNet_best.pt": "TrackNet_best.pt", "ckpts/InpaintNet_best.pt": "InpaintNet_best.pt"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def convert_tracknet(ckpt_dir, dest):
    """TrackNetV3 PyTorch checkpoints -> tracknet.onnx, inpaintnet.onnx (outputs checked against PyTorch)."""
    import numpy as np
    import onnxruntime as ort
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bball.tracknet import InpaintNet, TrackNet

    t = torch.load(ckpt_dir / "TrackNet_best.pt", map_location="cpu", weights_only=True)
    seq, bg = t["param_dict"]["seq_len"], t["param_dict"]["bg_mode"]
    if (seq, bg) != (8, "concat"):
        raise SystemExit(f"unexpected TrackNet checkpoint (seq_len={seq}, bg_mode={bg}); s13_shuttle.py expects 8, concat")
    net = TrackNet((seq + 1) * 3, seq)
    net.load_state_dict(t["model"])
    i = torch.load(ckpt_dir / "InpaintNet_best.pt", map_location="cpu", weights_only=True)
    if i["param_dict"]["seq_len"] != 16:
        raise SystemExit("unexpected InpaintNet checkpoint (seq_len != 16)")
    inp = InpaintNet()
    inp.load_state_dict(i["model"])
    for name, model, example, ins, outs, axes in (
            ("tracknet.onnx", net, (torch.rand(1, (seq + 1) * 3, 288, 512),), ["frames"], ["heatmaps"],
             {"frames": {0: "n"}, "heatmaps": {0: "n"}}),
            ("inpaintnet.onnx", inp, (torch.rand(1, 16, 2), torch.rand(1, 16, 1)), ["coords", "mask"],
             ["coords_out"], {"coords": {0: "n"}, "mask": {0: "n"}, "coords_out": {0: "n"}})):
        model.eval()
        part = dest / (name + ".part")
        torch.onnx.export(model, example, str(part), input_names=ins, output_names=outs, dynamic_axes=axes,
                          opset_version=17, dynamo=False)
        with torch.no_grad():
            ref = model(*example).numpy()
        got = ort.InferenceSession(str(part), providers=["CPUExecutionProvider"]).run(
            None, {k: v.numpy() for k, v in zip(ins, example)})[0]
        if np.abs(got - ref).max() > 1e-3:
            raise SystemExit(f"{name}: ONNX output differs from PyTorch")
        part.replace(dest / name)
        print(f"{name}: {(dest / name).stat().st_size / 1e6:.0f} MB")


def shuttle(dest):
    if all((dest / n).exists() for n in ("tracknet.onnx", "inpaintnet.onnx")):
        print("tracknet.onnx, inpaintnet.onnx: already there")
        return
    try:
        import onnx  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("--shuttle converts the TrackNetV3 checkpoints to ONNX once, which needs PyTorch and onnx:\n"
                         "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                         "  pip install onnx\n"
                         "(with the project's Python, e.g. .venv\\Scripts\\python.exe -m pip ...), then run this again.")
    ckpt = dest / "tracknetv3"
    ckpt.mkdir(exist_ok=True)
    if not all((ckpt / n).exists() for n in TRACKNET_FILES.values()):
        z = dest / "TrackNetV3_ckpts.zip"
        if not z.exists() or sha256(z) != TRACKNET_SHA256:
            print("TrackNetV3 checkpoints: downloading (130 MB)...", flush=True)
            part = z.with_suffix(".zip.part")
            urllib.request.urlretrieve(TRACKNET_ZIP, part)
            if sha256(part) != TRACKNET_SHA256:
                raise SystemExit("TrackNetV3_ckpts.zip: unexpected content (checksum differs); nothing was installed. "
                                 "Download it by hand from https://github.com/qaz812345/TrackNetV3 (Inference) "
                                 f"into {dest} and run this again.")
            part.replace(z)
        with zipfile.ZipFile(z) as zf:
            for src, name in TRACKNET_FILES.items():
                (ckpt / name).write_bytes(zf.read(src))
    convert_tracknet(ckpt, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parent / "models"))
    ap.add_argument("--skip-voice", action="store_true", help="skip the text-to-speech files (no video reel)")
    ap.add_argument("--shuttle", action="store_true", help="also set up the TrackNetV3 shuttle tracker (stage 13)")
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
    if args.shuttle:
        shuttle(dest)


if __name__ == "__main__":
    main()
