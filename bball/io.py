"""Small JSON / video helpers shared by every stage."""
import json
import subprocess
from pathlib import Path


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    return path


def probe(video):
    """Return (width, height, fps, duration_s) of the first video stream using ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    s = info["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return s["width"], s["height"], float(num) / float(den), float(info["format"]["duration"])
