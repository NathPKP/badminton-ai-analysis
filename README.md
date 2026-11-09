# Badminton match analysis from one phone video

[![tests](https://github.com/NathPKP/badminton-ai-analysis/actions/workflows/tests.yml/badge.svg)](https://github.com/NathPKP/badminton-ai-analysis/actions/workflows/tests.yml)

A working, fully local implementation of the badminton analysis pipeline described by
**[KiranYelisetti/badminton-ai-analysis](https://github.com/KiranYelisetti/badminton-ai-analysis)**.
That repository is an architecture and step-by-step build guide, with no code. This repository is one implementation of it, built by following the guide stage by stage with an AI assistant (Claude Code) and tested on a real singles clip.

It runs on a plain Windows laptop: no GPU, no cloud service, only free models.

**Input:** one video of a singles match, filmed from behind your baseline.
**Output**, in English or French:
- `rapport_match.pdf`, a 3-page written report to send by e-mail. It has the key figures, up to three lessons with frames and timestamps, a point-by-point table, a movement heatmap, and the method and its limits.
- `video/reel.mp4`, a vertical 9:16 highlight reel of about one minute. It shows the rallies with foot rings, trails, skeleton and a mini map. A coach voice-over (Kokoro TTS) explains each lesson on frozen frames.

## Install

You need **Python 3.12** and **[ffmpeg](https://ffmpeg.org/)** (with ffprobe) on the PATH. On Windows: `winget install Python.Python.3.12`, then `winget install Gyan.FFmpeg`, then open a new terminal.

Then, from the project folder:

```
powershell -ExecutionPolicy Bypass -File setup.ps1
```

On Linux or macOS, run `sh setup.sh` instead.

The script:
1. creates `.venv`;
2. installs the packages;
3. downloads the models into `models/` (about 420 MB; [download_models.py](download_models.py));
4. runs `run.py doctor`, which checks every piece and says how to fix what is missing.

<details><summary>Manual install</summary>

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_models.py
.venv\Scripts\python.exe run.py doctor
```

`download_models.py --skip-voice` leaves out the 350 MB of text-to-speech files if you don't need the video reel.

| File | Used by | Source | Licence |
|---|---|---|---|
| `yolox_s.onnx` | stage 1 | [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 |
| `pose_landmarker_heavy.task` | stage 8 | [Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) | Apache-2.0 |
| `kokoro-v1.0.onnx`, `voices-v1.0.bin` | stage 11 | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | Apache-2.0 |
</details>

It was developed on Windows 11; the tests also run on Linux. On Linux and macOS, use `.venv/bin/python` wherever this README says `.venv\Scripts\python.exe`.

## Analyse a match

```
.venv\Scripts\python.exe run.py new C:\videos\match.mp4 my-match --start 12
```

`--start` is the end of the warm-up, in seconds. Options:
- `--lang fr` gives a report and reel in French (English by default);
- `--camera raised` if you filmed from 2 m or higher.

The command creates `analyses/<video date>_my-match/` and runs every stage it can. When it needs you, it stops and prints what to do. Then run:

```
.venv\Scripts\python.exe run.py run my-match
```

It needs you at two points:

1. **Court calibration.** A window opens, and you click 4 or more court points: corners, T junctions, net posts… It takes about 2 minutes. For the next video filmed from the same spot, `--calib-from my-match` reuses the points.
2. **`review.json`**: who serves, and when each rally ends.
   - It is read on contact sheets (one frame every 0.5 s), starting from a draft (`review.todo.json`) pre-filled by automatic serve detection.
   - On floor-level footage, automatic server detection got 5 to 10 serves right out of 12, while reading 0.1 s close-ups got 12 out of 12.
   - After that, the score check catches reading errors on its own: any serve that breaks the service-court rule stops the run.
   - The easiest way is to let **Claude Code** do it (see [With an AI assistant](#with-an-ai-assistant)). Otherwise, fill it in by hand following [examples/review.example.json](examples/review.example.json).

The lessons (`coaching.json`) are written automatically by `s10_coach.py` from the numbers. These rules only keep a gap between points won and lost when it is large enough on a large enough sample. They cover the serve, the racket at the back of the court, the side where lost points end, recovery to the base, and long rallies. An AI assistant can write more specific lessons instead; see below.

Other commands:

```
.venv\Scripts\python.exe run.py status my-match
.venv\Scripts\python.exe run.py list
.venv\Scripts\python.exe run.py set my-match shorts_hue=[170,10]
.venv\Scripts\python.exe run.py run my-match --redo pose
```

- `status` shows the state of every stage.
- `list` lists all analyses.
- `set` changes a setting, here the colour of your shorts (red), which is how you are recognised.
- `--redo` reruns one stage and everything that depends on it.

`run.py` rebuilds a stage only when the content of one of its inputs, or one of its settings, has changed. Before replacing a file, it copies the old one to `history/<date-time>/`, so nothing is ever overwritten.

**Settings** (`run.py set <analysis> key=value`):

| Setting | Meaning |
|---|---|
| `start` | end of the warm-up (s) |
| `lang` | `en` or `fr` |
| `coach` | `auto` (rules) or `ai` (stop and wait for an AI-written coaching.json) |
| `shorts_hue` | `[min,max]` OpenCV hue (0–179) of your shorts: blue `[100,135]` (default), red `[170,10]`, green `[40,85]` |
| `camera` | `floor` (default) or `raised` |
| `thr` | stage 9 bend threshold (0.82) |
| `last_winner` | `R` (you) or `O`, if the video shows who won the last point |
| `accept_score` | serves that break the side rule but are right (fault, let), e.g. `[7]` |
| `speed` | rally playback speed in the reel (1.3) |
| `report_date` | date printed on the PDF |

## With an AI assistant

[CLAUDE.md](CLAUDE.md) tells [Claude Code](https://claude.com/claude-code) how to do the reading and writing steps, including the checks that caught mistakes during development. Open Claude Code in the project folder and ask:

- *“read the sheets of analyses/2026-11-09_my-match and write review.json”*
- *“write coaching.json for analyses/2026-11-09_my-match”*: more specific lessons than the rules can give. Every claim cites its numbers, and every frame is checked to show what its caption says. `run.py` keeps an edited coaching.json instead of regenerating it.

Any assistant that can read images and files can follow the same instructions.

## How it works

| Stage | Script | What it does |
|---|---|---|
| 0 | `s0_proxy.py` | 30 fps working copy (ffmpeg, optional HDR tone-mapping) |
| 1 | `s1_detect.py` | person detection with YOLOX-S (ONNX, CPU) at 10 fps |
| 2 | `s2_calibrate.py` | court homography from a few clicked court points |
| 3 | `s3_players.py` | identifies you (by shorts colour) and the opponent, maps feet to court metres |
| 4 | `timeline_sheets.py`, `s4_serves.py` | contact sheets for reading the match; serve detection and serving side |
| 5 | `s5_score.py` | replays the score; checks every serve against the service-court rule |
| 6 | `s6_rallies.py` | rally ends and durations |
| 7 | `s7_movement.py` | distance, speed, court zones, trips out of position, heatmaps |
| 8 | `s8_pose.py` | body posture with MediaPipe Pose (crouch, stance width, racket up) |
| 9 | `s9_pickups.py` | who bends to pick up the shuttle, which tells where it came down |
| 10 | `s10_report.py`, `s10_coach.py`, `s10_pdf.py` | numbers split by points won and lost; lessons; PDF report |
| 11 | `s11_video.py` | vertical reel with overlays and voice-over |

Details of each stage, the file formats, and what was learned along the way: **[docs/PIPELINE.md](docs/PIPELINE.md)**. Every script also runs on its own (`--help`). A user guide in French is in [docs/GUIDE_FR.md](docs/GUIDE_FR.md).

## Filming tips

The test clip was filmed with a phone on the floor. That works, with known limits. For better results:

1. Raise the camera to 2 m or more, behind your baseline, with the four court corners and the service lines in view. Then pass `--camera raised`.
2. Keep the camera still for the whole match, and use the original file rather than a re-compressed copy.
3. Wear shorts with a clear colour. Black, white or grey shorts can't be told apart by hue.

## Limitations

- **Opponent depth.** With a low camera, the far player's depth is biased toward the net. His court zones are not used; only his position across the court is.
- **Shuttle side near the net.** It is sometimes not visible. When stage 9 infers it from who picks the shuttle up, the report says so.
- **Pose.** Pose runs on the near player only, filmed from behind. 2D knee angles are unreliable from that view, so the measures are normalised by torso length instead.
- **Small samples.** A short clip gives trends, not certainties, and the report states this.
- **Scope.** Singles only, one fixed camera behind one baseline, English and French.

## Repository layout

```
run.py                 orchestrator: doctor, one folder per analysis, incremental reruns, history
s0_proxy.py … s11_video.py, timeline_sheets.py   one script per stage (s10_coach.py: automatic lessons)
bball/                 shared helpers: court geometry, I/O, smoothing, fonts, texts in both languages (i18n.py)
setup.ps1, setup.sh    one-command install
download_models.py     fetches the model files into models/
examples/              review.example.json, coaching.example.json
tests/                 unit tests (python -m unittest discover tests), run on every push
docs/                  PIPELINE.md (technical), GUIDE_FR.md (user guide in French)
CLAUDE.md              instructions for AI assistants working in the repository
analyses/              your analyses (git-ignored: personal videos and results)
```

## Privacy

`analyses/`, `models/` and video files are git-ignored. A match video shows other people, so think twice before publishing a reel or report that includes the opponent.

## Credits

- Pipeline design: [KiranYelisetti/badminton-ai-analysis](https://github.com/KiranYelisetti/badminton-ai-analysis) (MIT). This repository follows its stages, file formats and most of its thresholds. Where it departs from them (for example the stage 9 threshold or the calibration points used), the reason is given in [docs/PIPELINE.md](docs/PIPELINE.md).
- Models: YOLOX (Megvii), MediaPipe Pose Landmarker (Google), Kokoro-82M (hexgrad) via kokoro-onnx.
- Implementation written with [Claude Code](https://claude.com/claude-code).

## Licence

MIT; see [LICENSE](LICENSE). The original guide's notice is in [NOTICE](NOTICE).
