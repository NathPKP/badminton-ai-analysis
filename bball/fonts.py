"""Fonts for the PDF report and the video: Arial / Bahnschrift on Windows, otherwise the DejaVu fonts
that ship with matplotlib (so it also runs on Linux and macOS)."""
from pathlib import Path

WINDOWS = {"regular": "arial.ttf", "bold": "arialbd.ttf", "italic": "ariali.ttf", "display": "bahnschrift.ttf"}
DEJAVU = {"regular": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf", "italic": "DejaVuSans-Oblique.ttf",
          "display": "DejaVuSans-Bold.ttf"}


def font(style):
    """Path of a .ttf file for style = regular | bold | italic | display."""
    win = Path("C:/Windows/Fonts") / WINDOWS[style]
    if win.exists():
        return str(win)
    import matplotlib
    return str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / DEJAVU[style])
