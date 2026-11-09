# Badminton match analysis from one phone video

A working, fully local implementation of the badminton analysis pipeline described by
**[KiranYelisetti/badminton-ai-analysis](https://github.com/KiranYelisetti/badminton-ai-analysis)**.
That repository is an architecture and step-by-step build guide, with no code. This repository is one implementation of it, built by following the guide stage by stage with an AI assistant (Claude Code) and tested on a real singles clip.

It runs on a plain Windows laptop: no GPU, no cloud service, only free models.

**Input:** one video of a singles match, filmed from behind your baseline.
**Output:**
- `rapport_match.pdf`, a 3-page written report to send by e-mail. It has the key figures, three lessons with frames and timestamps, a point-by-point table, a movement heatmap, and the method and its limits.
- `video/reel.mp4`, a vertical 9:16 highlight reel of about one minute. It shows the rallies with foot rings, trails, skeleton and a mini map. A coach voice-over (Kokoro TTS) explains each lesson on frozen frames.

> The report and the voice-over are in **French**. The text comes from `coaching.json` and from a few strings in `s10_pdf.py` and `s11_video.py`; see [Language](#language).

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
| 10 | `s10_report.py`, `s10_pdf.py` | all numbers split by points won and lost; PDF report |
| 11 | `s11_video.py` | vertical reel with overlays and voice-over |

`run.py` chains the stages, keeps one folder per analysis and reruns only what changed. Details of each stage, the file formats and what was learned along the way: **[docs/PIPELINE.md](docs/PIPELINE.md)**.

### What is not automatic

Three steps need a person, or an AI assistant reading images. `run.py` stops at each one and says what to do:

1. **Court calibration.** You click 4 or more court points in a window, which takes about 2 minutes. It can be reused across videos filmed from the same spot (`--calib-from`).
2. **`review.json`**: who serves, and when each rally ends. It is read on the contact sheets, with a draft pre-filled from automatic serve detection. On floor-level footage, automatic server detection was right only 5 to 10 times out of 12, while reading 0.1 s close-ups was right 12 times out of 12. The score check in stage 5 then catches any reading error automatically.
3. **`coaching.json`**: the lessons, the voice-over script and the PDF texts. They are written from `report.md`, and every claim cites its evidence. [`examples/coaching.example.json`](examples/coaching.example.json) is a complete example.

## Install

You need Python 3.12 and [ffmpeg](https://ffmpeg.org/) (with ffprobe) on the PATH. On Windows: `winget install Gyan.FFmpeg`.

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_models.py
```

`download_models.py` fetches about 420 MB into `models/`. Use `--skip-voice` to leave out the 350 MB of text-to-speech files if you don't need the video reel.

| File | Used by | Source | Licence |
|---|---|---|---|
| `yolox_s.onnx` | stage 1 | [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 |
| `pose_landmarker_heavy.task` | stage 8 | [Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) | Apache-2.0 |
| `kokoro-v1.0.onnx`, `voices-v1.0.bin` | stage 11 | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | Apache-2.0 |

It was developed and tested on Windows 11. On Linux and macOS, use `.venv/bin/python` instead. The fonts fall back to the DejaVu fonts bundled with matplotlib. The pipeline has not been tested there.

## Quick start

```
.venv\Scripts\python.exe run.py new C:\videos\match.mp4 my-match --start 12
```

`--start` is the end of the warm-up, in seconds. The command creates `analyses/<video date>_my-match/` and runs every stage it can. It stops at the next manual step and prints what to do. After each manual step, run:

```
.venv\Scripts\python.exe run.py run my-match
```

Other commands:

```
.venv\Scripts\python.exe run.py status my-match
.venv\Scripts\python.exe run.py list
.venv\Scripts\python.exe run.py set my-match shorts_hue=[170,10]
.venv\Scripts\python.exe run.py run my-match --redo pose
```

- `status` shows the state of every stage.
- `list` lists all analyses.
- `set` changes a setting; the shorts colour in the example is red, for identifying you.
- `--redo` reruns one stage and everything that depends on it.

`run.py` rebuilds a stage when the content (sha1) of one of its inputs, or one of its settings, has changed. Before replacing a result file, it copies the old one to `history/<date-time>/` in the analysis folder. Every version of the hand-written files is kept there too, so nothing is ever overwritten.

The full user guide, with all settings, is in French: **[docs/GUIDE_FR.md](docs/GUIDE_FR.md)**. Every script also runs on its own (`python s7_movement.py --help`).

## Filming tips

The test clip was filmed with a phone on the floor. That works, with known limits. For better results:

1. Raise the camera to 2 m or more, behind your baseline, with the four court corners and the service lines in view. Then pass `--camera haute`.
2. Keep the camera still for the whole match, and use the original file rather than a re-compressed copy.
3. Wear shorts with a clear colour (blue by default; `shorts_hue` changes it). Black, white or grey shorts can't be told apart by hue.

## Limitations

- **Opponent depth.** With a low camera, the far player's depth is biased toward the net. His court zones are not used; only his position across the court is.
- **Shuttle side near the net.** It is sometimes not visible. When stage 9 infers it from who picks the shuttle up, the report says so.
- **Pose.** Pose runs on the near player only, filmed from behind. 2D knee angles are unreliable from that view, so the measures are normalised by torso length instead.
- **Small samples.** A short clip gives trends, not certainties, and the report states this.
- **Scope.** Singles only, one fixed camera behind one baseline.

## Language

The pipeline was built for a French-speaking player. To produce another language:
- write `coaching.json` in that language;
- change the voice and `lang="fr-fr"` in `s11_video.py` (Kokoro has English, Spanish, Italian, Portuguese, Hindi, Japanese and Chinese voices);
- translate the fixed strings in `s10_pdf.py` and `s11_video.py`.

The code, comments and file formats are in English.

## Repository layout

```
run.py                 orchestrator: one folder per analysis, incremental reruns, history
s0_proxy.py … s11_video.py, timeline_sheets.py   one script per stage
bball/                 shared helpers (court geometry, JSON/video I/O, smoothing, fonts)
download_models.py     fetches the model files into models/
examples/              review.example.json, coaching.example.json
docs/PIPELINE.md       technical description of every stage
docs/GUIDE_FR.md       user guide (French)
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
