# Working in this repository

Local badminton match analysis: one phone video in, a PDF report and a vertical reel out. `run.py` drives
the stages and stops where reading by eye or writing is needed; those steps are what an assistant is usually
asked to do here. Read `README.md` for the overview and `docs/PIPELINE.md` for each stage and file format.

- Run everything with the project's Python: `.venv\Scripts\python.exe` on Windows, `.venv/bin/python` elsewhere.
  `run.py doctor` checks the installation.
- `analyses/` holds the users' match videos and results: never commit it, never copy frames from it anywhere public.
- After any change to an analysis file, run `run.py run <analysis>`: it reruns what depends on it and keeps
  the old version in `history/`. Do not delete files from an analysis to force a rerun; use `--redo <stage>`.
- The analysis' language is `params.lang` in `analysis.json` (`en` or `fr`); write user-facing text in it.

## “Read the sheets and write review.json”

Goal: `analyses/<id>/review.json` (format: `examples/review.example.json`), the truth for serves and rally ends.

1. Start from `review.todo.json` (serves detected automatically: times within about 2 s, or 0.7 s when the note
   says they come from the shuttle's flight; server guessed: the shuttle does not tell it on floor footage) and
   the sheets `contact/timeline/*.jpg`: one tile every 0.5 s, each labelled with its time, with a zoom on the
   far player. `R` = the near player (back to the camera, the user), `O` = the far player.
2. For every serve: `t` = racket contact, `server`. For every rally: `t` = last moment of live play (last hit,
   lunge or shuttle landing), `shuttle` = side where it came down (`R` / `O` / `null` when not visible).
   The first seconds are often warm-up or players taking their places: start at `params.start`.
3. Save as `review.json`, run `run.py run <id>`. Stage 4 then writes 0.1 s close-ups around every serve in
   `contact/serve_closeups/`: confirm each server there, reading both players without assuming the answer.
   Tiles every 0.5 s are NOT enough to tell who serves (a serve swing was once read as the return).
4. Stage 5 checks every serve against the service-court rule (server's score even → right court). If `run.py`
   reports contradictions, one wrong server shifts every later serve: fix the first contradiction, rerun.
   Only if a contradiction is real (fault, let) set `accept_score=[n]`.
5. `contact/rally_ends/end_NN.jpg` (0.25 s tiles) shows each rally end you wrote; re-check them.

## “Write coaching.json”

Goal: `analyses/<id>/coaching.json` (format and a full example: `examples/coaching.example.json`; the `pdf`
section is documented at the top of `s10_pdf.py`). With `coach=auto` the rules of `s10_coach.py` already
wrote one; a file you edit is kept by `run.py` instead of being regenerated.

- Read `report.md` and `report.json`. Pick at most three lessons where points won and lost really differ;
  state the sample size, and say “trend” when it is small.
- Every number must come from `report.json` / `motion.json` / `pose_stats.json`; list where in `evidence`.
- Every freeze frame must show what its caption claims. Extract the frame and look at it
  (`cv2.VideoCapture(proxy.mp4)`, frame = round(t × 30)). The floor camera lies about depth: measure a
  position with the calibration (`bball.court.to_court`) rather than judging it by eye.
- `voice` and the video texts talk to the player (“you”); the `pdf` section is the player's own report
  (“I”). A lesson card shows at most two images (a zoom counts as one). `after_point` must be a point number.
- Have the claims checked by someone who tries to refute them before the video is rendered.

## Code conventions

- One script per stage, JSON in and out, paths as arguments; `run.py` knows each stage's inputs and outputs.
- User-facing fixed text lives in `bball/i18n.py` (PDF, reel, report) and `MSG` in `run.py`, in both languages.
- `python -m unittest discover tests` must pass; keep `README.md`, `docs/` and the docstrings in step with the code.
