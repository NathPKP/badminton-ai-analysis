# Pipeline: technical description

Each stage is one script that reads and writes JSON files in one analysis folder. `run.py` passes
the paths, and every script can also run on its own (`--help`). Times are in seconds of the video.
Boxes are `[x1, y1, x2, y2]` in fractions of the frame. Court positions are in metres.

**Court frame** (`bball/court.py`):
- `x` runs across the court, from 0 to 6.1, starting at the near left sideline as the camera sees it.
- `y` runs along the court, from 0 at the near baseline to 13.4 at the far baseline; the net is at 6.7.
- Singles sidelines are at 0.46 and 5.64; the short service lines are at 4.72 and 8.68.
- Stage 7 measures each player in his own half: `depth` = distance from his own baseline, `across` = distance from his own left sideline. The two players can therefore be compared directly.

## Stage by stage

### 0. Working copy: `s0_proxy.py`
ffmpeg re-encodes the video to constant 30 fps, at most `--height` (720) rows, with no audio and one keyframe per second so that seeking for contact sheets is fast. `--tonemap` converts HDR footage (HLG or PQ, as recorded by iPhones) to SDR. Every later stage reads only `proxy.mp4`.

### 1. Detection: `s1_detect.py`
YOLOX-S runs through onnxruntime on the CPU. Each frame is letterboxed to 640, and the outputs of the strides 8, 16 and 32 are decoded. Only the person class is kept, with confidence ≥ 0.3 and NMS at 0.5. It runs on every third frame, so at 10 fps.
Output `dets.json`: `{"fps": 10, "frames": [{"t", "p": [[x1, y1, x2, y2, score], …]}]}`.

### 2. Calibration: `s2_calibrate.py`
An OpenCV window opens on one frame. For each point, press the letter of a named court point (corners, T junctions, service-line ends, net posts, far long-service line…), then click it. With at least 4 points, `cv2.findHomography` gives H (metres → pixels). The check image redraws the court from H, so it shows whether the fit is right; with exactly 4 points the reprojection error is always 0 and says nothing.
`calib.json` holds one segment per camera position (`t0`, `t1`, the clicked points, H). `--recompute` solves again from the saved points without opening the window.

### 3. Players: `s3_players.py`
The feet point (bottom centre of the box) goes through H⁻¹ to court metres.
- **You** are the near-half person whose shorts band contains at least `--min-blue` (0.35) pixels of the shorts hue. The band is 48–62 % of the box height, central half of its width; the hue is set with `--shorts-hue`, blue by default. If nobody is blue enough, the frame is left as a gap rather than guessed.
- **The opponent** is the most confident detection in the far half.
- A box cut off by the bottom of the frame is flagged `rt` (feet out of frame, position unknown).
- Short gaps are filled (`rf`, `of`) and isolated jumps removed.
- `--start` drops the warm-up.

### 4. Serves: `timeline_sheets.py`, `s4_serves.py`
`timeline_sheets.py` draws contact sheets of the whole match: one tile every 0.5 s, each with a zoom on the far player. They are used to read the match by eye.
`s4_serves.py` finds candidate serves: both players stand still (near < 0.7 m/s, far < 0.9 m/s) in their service areas for at least 0.8 s. Candidates on the same side less than 6 s apart are merged.
The side comes from the diagonal: if you stand on the camera's right and the opponent on the camera's left, both are in their right service courts, so the server's score is even. The opponent faces the camera, so his right is the camera's left.
Who serves cannot be read from positions. With `--review`, the serves listed in `review.json` are used instead, and the automatic candidates are only scored against them. `--closeups` writes 0.1 s close-ups of both players around each serve, so the server can be confirmed.

### 5. Score: `s5_score.py`
The score is replayed with "the winner of a rally serves the next one". In singles the server stands in his right court when his score is even and in his left court when it is odd. Every serve that breaks this rule is listed in `contradictions`, and `run.py` stops on any that has not been accepted with `accept_score`.
This check is free and catches most reading errors: one wrong server shifts the score, so every following serve contradicts the rule.

### 6. Rallies: `s6_rallies.py`
The rally ends come from `review.json`: the last moment of live play. `r1` is that moment plus 0.25 s, and `rdur = r1 − serve`. Racket sounds are not used, because hits from other courts make rallies look too long. `--blind` draws the end-of-rally sheets without marking the current reading, for an unbiased second reading.
Output `points.json`: one record per point (`n, t, server, side, before, after, winner, r0, r1, rdur, shuttle`).

### 7. Movement: `s7_movement.py`
The window of each rally runs from serve + 0.3 s to the rally end. Positions are de-spiked with a rolling median, then smoothed with a 0.7 s moving mean; without this, jitter adds about 30 % to the distance. Per rally and summed, the script computes:
- distance and average and peak speed;
- the share of time in the back (the last 1.6 m) and in the front (past the short service line);
- trips out of position: more than 2.2 m from the base point (3.05, 3.4), with the distance to the base 1 s later.
It also draws the heatmaps.
The far player's depth is about 5 times less precise from a low camera, and on the test clip it is biased toward the net. `notes.O_depth` records this; only his across position is used.

### 8. Pose: `s8_pose.py`
MediaPipe Pose Landmarker (heavy) runs in VIDEO mode on a square crop 1.35 times the size of your box, at 10 fps. Seen from behind, 2D knee angles look straighter than they are, so the measures are normalised by torso length:
- `crouch` = 1 − (hip-to-ankle height / torso) / (the same ratio when standing upright between points);
- `stance` = ankle distance / shoulder width when square-on;
- `racket_up` = a wrist above the shoulder line.
Contexts:
- `READY`: slower than 0.8 m/s and within 1.6 m of the base;
- `FRONT`: past the short service line;
- `REAR`: the last 1.6 m before the baseline.
Statistics are split by points won and lost; a rally whose winner is unknown counts in neither.
Validated on 24 crops rated blind: hand-up agreed 21 times out of 24, and stance width was well ordered. Crouch only separates upright from bent.

### 9. Shuttle side from pick-ups: `s9_pickups.py`
For each player, the script fits the height of an upright box against how far below the horizon the feet are (`h = a · feet_row + b`), on still, untruncated frames. A bend is a box below `--thr` of that height while moving slowly, for at least 0.3 s. The first bend after a rally end tells which side the shuttle came down on.
The guide uses 0.72, for a raised camera. From the floor, a bend only shrinks the box to about 76–80 %, so the default is 0.82. Thresholds from 0.78 to 0.85 all agreed with the eye reading.
`--review` prints the agreement with the sides read by eye. The report uses a side read by eye first, and an inferred side only when there is none, saying which it used.

### 10. Report: `s10_report.py`, `s10_pdf.py`
`s10_report.py` gathers every number into `report.json` and `report.md`:
- won and lost on your serve and on his;
- results by rally length (≤ 5 s, 5–10 s, > 10 s);
- how each point ended, combining the winner with the shuttle side;
- your position at each rally end, movement and pose by result;
- example timestamps;
- computed limits.
The coaching is then written from these numbers into `coaching.json`. [`examples/coaching.example.json`](../examples/coaching.example.json) is a complete example.
- **By rules** (`s10_coach.py`, the default `coach=auto`): five rules each look for one gap between points won and lost, and keep it only when it is large enough on a large enough sample. The rules cover the serve, the racket at the back of the court, the side where lost points end, recovery to the base, and long rallies. The strongest three become lessons, each with a freeze frame taken from the match and its numbers in `evidence`. If no rule fires, a single lesson sums up the movement.
- **By an AI assistant**, following [CLAUDE.md](../CLAUDE.md): more specific lessons, where every claim cites its numbers and every frame is checked by eye.
All fixed texts (report, reel, run.py messages) exist in English and French (`bball/i18n.py`, `MSG` in `run.py`); the `lang` setting picks one.
`s10_pdf.py` builds the A4 report with reportlab. The general parts are computed: key figures, points table, movement, method. The match-specific texts come from the `pdf` section of `coaching.json`; its format is documented at the top of the script.

### 11. Reel: `s11_video.py`
The reel is 1080×1920, with a score bar, the match video, the mini map and captions. Rallies are cut from the serve to the rally end and played at `--speed` (1.3). After the rally named in each lesson, the video freezes on the lesson frames while the voice explains. The voice is Kokoro TTS, French voice `ff_siwis`, run locally. Frames are piped raw to ffmpeg, and the finished reel only replaces an existing one at the end.

## `run.py`: orchestration

- **One folder per analysis**: `analyses/<video date>_<name>/` with `analysis.json` (source video and settings) and `.pipeline.json` (state).
- **Incremental**: each stage records the sha1 of its inputs, its command line and the settings it reads from `analysis.json`. A stage runs again when one of these changed, when an output is missing, or when it failed or was interrupted last time. Changes propagate: a corrected `review.json` reruns stages 4, 6, 7, 8, 9 and 10, but not 5 if `serves.json` comes out identical.
- **Manual gates**: calibration, `review.json` and score contradictions, plus `coaching.json` with `coach=ai`.
  - With `coach=ai`, coaching.json is asked for again whenever `report.json` changes after it was written, unless it is edited or `--accept coaching` is given.
  - With `coach=auto`, the rules rewrite it instead, as long as nobody has edited the file.
- **`doctor`** checks Python, the packages, ffmpeg, the models, the fonts and write access, and says how to fix what fails.
- **History**: a result file is copied to `history/<date-time>/` before it is replaced, and so is every version of `calib.json`, `review.json` and `coaching.json`. A copy in history is never overwritten.

## Lessons learned on floor-level footage

- **Who serves** cannot be read reliably from automatic cues. Positions, motion energy, pose and movement onset were all tried, and each gave 5 to 10 correct servers out of 12. Reading 0.1 s close-ups of both players, without knowing the expected answer, gave 12 out of 12. One-frame-every-0.5 s sheets are not enough: a serve swing was once read as the return.
- **Perspective fools the eye on depth**: twice, a shuttle looked deep when it was short. Measure it with the calibration (`mark_px`, then H⁻¹) instead of judging by eye.
- **Opponent depth**: a horizon-based correction of the calibration made the far depth worse, and it was removed. Only clicked points spread in depth across the far half, such as the far long-service line and the far corners, can fix it.
- **Coaching claims** must survive an adversarial check. The first draft overstated three claims: rally durations inflated by padding, statistics computed on too few frames, and freeze frames that did not show the fault described.
- **Stance from behind**: profile views made the raw stance ratio blow up, which is why it is normalised by torso length and by the square-on shoulder width.
