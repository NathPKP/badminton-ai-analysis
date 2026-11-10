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

**Automatic calibration: `s2_autocalib.py`.** `run.py` tries it first.
- **Line mask.** It finds the painted lines: pixels that are bright, thin and unsaturated (white top-hat). It does this on 40 frames, keeps those of the main camera view (the most long straight lines, and similar to each other), and combines them, which removes the players.
- **Line candidates.** Hough lines are split into lines across the court and lines along it.
- **Search.** Every pair across × pair along is matched to every pair of court lines in the same order. That makes about 20,000 homographies, scored in batches. Each is scored by the share of the projected court that falls on the mask.
- **Plausibility.** A homography is rejected when the projected court is squeezed into a few pixels, when the near half is too short, when the sidelines are nearly flat, or when the court is too narrow.
- **Refinement.** The best one is refined on every court line.
- **Acceptance.** It is accepted when at least 60 % of the projected court lies on painted lines, and when the players' feet from stage 1 put somebody in each half in at least 25 % of the frames that show people.
  - The feet check is what tells right from wrong. On the floor-level test clip, a wrong court reached 69 % on the lines but 0 % on the feet; the hand calibration scores 97 % on the feet.
  - On a BWF broadcast, the court was found exactly with no click.
- **Fallback.** Otherwise the window opens as before, with the points it found already placed.

### 3. Players: `s3_players.py`
The feet point (bottom centre of the box) goes through H⁻¹ to court metres.
- **You** are the near-half person whose shorts band contains at least `--min-blue` (0.35) pixels of the shorts hue. The band is 48–62 % of the box height, central half of its width; the hue is set with `--shorts-hue`, blue by default. If nobody is blue enough, the frame is left as a gap rather than guessed.
- **The opponent** is the most confident detection in the far half.
- A box cut off by the bottom of the frame is flagged `rt` (feet out of frame, position unknown).
- Short gaps are filled (`rf`, `of`) and isolated jumps removed.
- `--start` drops the warm-up.
- **TV footage** (`footage=tv` in `run.py`):
  - `--near-any` takes the most confident person in the near half instead of the shorts colour, because players change ends between games.
  - `--check-view` leaves out the frames where the calibrated court does not fall on painted lines: close-ups, replays, crowd shots. Each frame gets `"v": true/false`.

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

### 12. Benchmark: `s12_benchmark.py`
Movement per rally and per player across analyses, computed exactly as stage 7 does (same smoothing, same `rally_stats`).
- **Rallies.** They come from `points.json` when the analysis has one. On TV footage (`footage=tv`) they are found automatically:
  - The main view is on screen from the serve preparation until shortly after the rally ends.
  - So each main-view run of 4 s or more holds one rally, from the end of the still preparation to the last fast movement.
  - Checked by eye on BWF footage: the start is right, and the end comes about 1 s late.
- **Reference numbers.** [`benchmarks/pro_singles.json`](../benchmarks/pro_singles.json) holds the aggregates from two BWF singles finals: 23 rallies, both players. The videos are not part of the repository.
- **In the report.** `s10_report.py` measures the player's own rallies the same way, and the PDF adds a "compared with professional players" table.

| Per rally (median) | Pros (2 BWF finals) | Test clip (amateur, floor camera) |
|---|---|---|
| Rally length | 8.8 s | 3.8 s |
| Average speed | 1.33 m/s | 0.90 m/s |
| Peak speed | 2.7 m/s | 1.6 m/s |
| Distance per 10 s of play | 13.6 m | 10.2 m |
| Distance to the base 1 s after a trip | 1.2 m | 1.4 m |

### 13. Shuttle tracking: `s13_shuttle.py` (optional)
[TrackNetV3](https://github.com/qaz812345/TrackNetV3) (MIT, weights included) finds the shuttle in every frame.
- **Install**: `download_models.py --shuttle` downloads the published PyTorch checkpoints (130 MB, checksum verified) and converts them once to ONNX (`models/tracknet.onnx`, `models/inpaintnet.onnx`). The conversion needs PyTorch and onnx, and checks that the ONNX output matches PyTorch. Analysing a video then needs only onnxruntime. `run.py` runs the stage when these two files exist, unless `shuttle=off`.
- **Method**, as in the original `predict.py`:
  - frames resized to 512×288, 8 at a time, after a median background image;
  - the shuttle is the centre of the largest heatmap blob above 0.5;
  - InpaintNet fills short gaps inside a flight. Unlike the original, gaps longer than its 16-frame window stay empty, because over a close-up it invented a whole trajectory.
  - Windows don't overlap: the original's overlapping "weight" mode costs 8 times more for little gain.
- **Camera changes.** The original takes one background image for the whole video, which assumes a fixed camera. Phone videos posted online are often edited or filmed by hand, and broadcasts cut between cameras. So the video is scanned first:
  - A **cut** is a jump of the grey-level histogram or of the mean pixel change between two frames. The camera **moves** when the image shifts by more than 0.3 % of its width in 0.2 s (phase correlation).
  - Each still stretch of a shot gets its own background. While the camera moves, there is one per second, made from the frames within 1 s in the same shot.
  - No 8-frame window, filled-in gap or flight crosses a cut. `shuttle.json` lists the shots and the moving stretches (`camera`).
  - `--bg global` restores the original method.
  - **Broadcasts:** the real shuttle is also seen in close-ups and slow-motion replays, out of play. With `footage=tv`, `--view` keeps only the frames where stage 3 found the main court view.
- **Speed** on a 13th-gen i5 CPU: about 0.06 s per frame, so 5 min for the 157 s test clip and 15–17 min for 8 minutes of TV footage.
- **Flights** (`bball/shuttle.py`):
  - Detections that come back to the same spot at several moments are dropped first: on the test clip, 30 % of them were white holds of a climbing wall behind the court.
  - The rest is grouped into moving runs, the `flights` of `shuttle.json`.
- **Serve times.** Stage 4a moves each automatic serve to the start of the serve flight: a flight starting within 2.5 s of it, after at least 1.5 s without any flight. This rules out the shuttle being hit back to the server between rallies. Two candidates landing on the same flight are one serve, so only the closer one is kept.
  - On the test clip: all 12 serves found, with 1 false candidate (2 before).
  - The time error went from a median of 0.7 s (largest 2.3 s) to a median of 0.2 s (largest 0.7 s).
  - The 0.1 s close-ups used to read the server are cut around these times.

Seven videos were measured on windows of 60 to 150 s. "Precision" is the share of detections that are really on the shuttle, out of 16 random flight points checked by eye.

| Video | Camera | Shuttle found | Precision |
|---|---|---|---|
| BWF final, men | TV: cuts, replays, main camera moving half of the time | 94 % of rally frames (88 % with one background for the whole video) | 12–15 / 16 |
| BWF final, women | TV | 76 % of rally frames. Outside rallies: 54 %, mostly the real shuttle in replays; 11 % with `--view` | 13 / 16 |
| Junior U19 boys | fixed, low, to the side | 65 % of all frames | 14 / 16 |
| Junior U17 boys | fixed, low, in a corner, dark background | 25 % of all frames | 14 / 16 |
| Junior U15 girls | fixed, edited (score overlay, points cut out) | 60 % of all frames | 14–15 / 16 |
| Amateur men | handheld, panning (moving 28 % of the window), edited | (no reference) | 12–13 / 16: misses are on spectators' heads |
| Test clip (the author's match) | fixed, on the floor, climbing wall behind | 35 % of rally frames | 12 / 16: the misses are round white holds |

- **Per-shot backgrounds:** they help on the moving broadcast camera (88 % to 94 %) and change nothing elsewhere, so they are the default.
- **What usually fails:** white objects the size of a shuttle that move in the image, such as spectators' faces, a racket shaft, a foot, or holds of a climbing wall.
- **What is not handled:** zooms and fast pans inside a shot leave a blurred background. The tracker then finds fewer shuttles, but the precision measured on the amateur video stays the same.

- **What it does not solve yet on floor footage:**
  - **Who serves.** The first point of the serve flight is near the far player when they serve low. After a high serve the shuttle is first seen already high, where the near player's large box also covers it. Result: 7 of 12.
  - **Rally ends.** The shuttle hit back to the server right after a point looks like play, which would push some rally ends 0.5–2 s late.
  - On TV footage both should work far better. That is the next thing to measure, once the pro analyses have a `review.json` to compare against.

## Full runs on other videos (November 2026)

Five other videos were taken to the PDF and the reel: two BWF finals (read by eye with the help of the TV scoreboard), and three junior videos. U15 is a highlights edit; U19 and U17 are continuous, fixed camera. A handheld amateur video was not run, because the calibration has one homography per time range and cannot follow a camera that moves.

- **Rallies from the shuttle.** On the men's final, the groups of flights gave all 11 rallies, plus one split in two by a long lob (the 1.3 s let on the women's final is dropped by the 1.5 s minimum). Rally ends fell within 0.5 s of the eye reading 9 times out of 10. `review.todo.json` is now pre-filled with them.
- **Who serves is still not automatic.** With the stage 4 guess, 15 of 29 (U19) and 20 of 36 (U17) serves agree with the service-court rule, which is chance level. The reports made from these unchecked readings have a wrong score.
- **Never accept contradictions in bulk.** On U15 the score built by the rule "the winner serves" came out 10-3, while the score drawn on the video was 14-11: in a highlights edit, the next rally shown is not the next one played. `accept_score` is for one known fault or let, not for making a run finish. For a clip that starts during a game, `start_score` gives the score at the first serve.
- **Broadcast transitions.** Serve 9 of the women's final broke the side rule: the second before it was the BWF logo transition, on which the server's position was measured.
- **Speeds above 6 m/s** (8 to 10 m/s on the juniors) mean the player tracking jumped onto someone else, such as a player on the next court. Nothing filters these jumps yet.
- **Rule-based lessons on small samples.** The serve rule fires on 5 vs 6 points (women's final) and turns "7 of 10 won on my serve" into "vary the serve" (U15). Its sample threshold is too low.

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
- **The service-court rule does not identify the server on its own.** With the 12 sides measured correctly, 128 of the 4,096 possible server sequences still obey the rule.
  - With the rule, the "who moves first" guess and the side where the shuttle was picked up, the most likely sequence had 10 servers right out of 12, but the true one ranked only 29th.
  - When the draft is written, the shuttle side isn't known yet. With only the rule and the guess, the result drops to 7 out of 12, worse than the guess alone (8 out of 12).
  - So stage 5 uses the rule as a check, not as a solver. The shuttle tracker (stage 13) times the serves precisely, but on this footage it does not tell the server either (7 out of 12).
- **Player speed does not find amateur rallies.** On the test clip, the faster of the two players moves at a median 1.0 m/s during rallies and 0.7 m/s between them, and the two ranges overlap widely: players walk to pick up the shuttle about as fast as they play. Rally ends are therefore read by eye. The shuttle tracker sees play well, but it also sees the shuttle being hit back to the server after a point.
- **Perspective fools the eye on depth**: twice, a shuttle looked deep when it was short. Measure it with the calibration (`mark_px`, then H⁻¹) instead of judging by eye.
- **Opponent depth**: a horizon-based correction of the calibration made the far depth worse, and it was removed. Only clicked points spread in depth across the far half, such as the far long-service line and the far corners, can fix it.
- **Coaching claims** must survive an adversarial check. The first draft overstated three claims: rally durations inflated by padding, statistics computed on too few frames, and freeze frames that did not show the fault described.
- **Stance from behind**: profile views made the raw stance ratio blow up, which is why it is normalised by torso length and by the square-on shoulder width.
