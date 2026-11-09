"""Full pipeline, one folder per analysis (analyses/<date>_<name>/): nothing is ever overwritten.

  python run.py doctor                      check the installation
  python run.py new <video> <name> [--start 12] [--lang en|fr] [--calib-from ID] [--shorts-hue 100 135]
  python run.py run <ID> [--redo STAGE] [--accept coaching] [--no-window]
  python run.py set <ID> start=12 thr=0.8 shorts_hue=[170,10] ...
  python run.py status <ID>
  python run.py list
(python = the project's .venv\\Scripts\\python.exe on Windows, .venv/bin/python elsewhere)

`run` does every stage that is missing or out of date (the content of one of its input files or
one of its settings changed since it last ran), in order, and stops where a person has to act:
  - the court calibration (stage 2, a window opens; --calib-from reuses another analysis' points),
  - review.json: serves and rally ends read by eye on the contact sheets (an AI assistant can do it),
  - a serve that breaks the service-court rule (score check),
  - coaching.json, only with coach=ai: the lessons written from report.md by an AI assistant. With
    coach=auto (default) s10_coach.py writes them by rules, and rewrites them when the numbers change
    as long as nobody has edited the file.
Then run it again: it carries on from there. Before a stage replaces a result file, the old version
is copied to <analysis>/history/<date-time>/, and so is every version of the hand-written files
(calib, review, coaching). Not kept: proxy.mp4 (rebuilt from the video), contact sheets and figures
(rebuilt from the results). Messages follow the analysis' language (lang=en|fr).
ID = the folder name under analyses/, or any unique part of it ("match1").
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from bball.i18n import Text

ROOT = Path(__file__).resolve().parent
ANALYSES = ROOT / "analyses"
MODELS = ROOT / "models"
PY = sys.executable
ME = r".venv\Scripts\python.exe run.py" if os.name == "nt" else ".venv/bin/python run.py"   # as printed in messages
STATE = ".pipeline.json"
HAND_WRITTEN = ("calib.json", "review.json", "coaching.json")
CAMERA = {"floor": "sol", "raised": "haute", "sol": "sol", "haute": "haute"}

MSG = {
    "en": {
        "stages": {"proxy": "0. working copy at 30 fps", "detect": "1. player detection", "calib": "2. court calibration",
                   "players": "3. player positions", "sheets": "4a. reading sheets + detected serves",
                   "review": "reading serves and rally ends", "serves": "4. serves",
                   "score": "5. score + service-side check", "rallies": "6. rally ends", "movement": "7. movement",
                   "pose": "8. posture", "pickups": "9. side where the shuttle lands", "report": "10. report numbers",
                   "coaching": "lessons (coaching.json)", "pdf": "10. PDF report", "video": "11. video reel"},
        "why": {"todo": "to do", "forced": "redo requested", "settings": "settings changed", "changed": "{f} changed",
                "reread": "{f} changed: to be re-read", "start": "start setting missing", "redo": "redo"},
        "kept": "   (version kept: {p}, {why})",
        "kept_why": {"used": "version used", "redo": "stage redone", "recalib": "before recalibration",
                     "regen": "regenerated"},
        "failed": "   FAILED: stage “{s}”, see the message above.",
        "all_done": "All up to date. Report: {pdf}\n                 Video:  {video}",
        "calib_reused": "   calibration points taken from {src}",
        "calib_check": "   CHECK the image {img}: the yellow court must fall on the lines.",
        "calib_moved": "   If the camera moved: {me} run {id} --redo calib",
        "calib_failed": "   FAILED to reuse the calibration: see the message above.",
        "calib_todo": "   TO DO: calibrate the court (a window opens): {me} run {id}",
        "calib_window": "   Calibration window: press the letter of a court point, then click it; Enter when done "
                        "(see s2_calibrate.py).",
        "calib_stopped": "   calibration stopped: nothing changed",
        "start_todo": "   TO DO: give the end of the warm-up (in seconds), for example:\n     {me} set {id} start=12\n"
                      "   then: {me} run {id}",
        "review_todo": "   TO DO: read who serves and when each rally ends.\n"
                       "     Sheets: {rel}/contact/timeline (one frame every 0.5 s)\n"
                       "     Pre-filled draft: {todo} (format: examples/review.example.json)\n"
                       "     With Claude Code open in this folder: “read the sheets of {rel} and write review.json”\n"
                       "   then: {me} run {id}",
        "score_bad": "   TO CHECK: serve(s) {bad} break the service-court rule.\n"
                     "     A server or a serve is probably misread in {rel}/review.json\n"
                     "     (close-ups: {rel}/contact/serve_closeups). Fix it, then: {me} run {id}\n"
                     "     If that is really what happened (fault, let...): {me} set {id} accept_score={list}",
        "coach_reread": "   TO RE-READ: the numbers of {rel}/report.md changed since coaching.json was written.\n"
                        "     Ask your AI assistant: “update coaching.json for {rel}”\n"
                        "     If it is still right as it is: {me} run {id} --accept coaching\n"
                        "     Or let the rules rewrite it: {me} set {id} coach=auto, then delete coaching.json",
        "coach_todo": "   TO DO: write the lessons in {rel}/coaching.json from {rel}/report.md, with the “pdf” section\n"
                      "     (texts of the written report, first person). Ask your AI assistant: “write coaching.json for {rel}”\n"
                      "     (example: examples/coaching.example.json; format at the top of s10_pdf.py).\n"
                      "     Or let the rules write it: {me} set {id} coach=auto",
        "then": "   then: {me} run {id}",
        "coach_auto": "   lessons written by rules (s10_coach.py); for more specific ones, have an AI assistant\n"
                      "   rewrite coaching.json: run.py keeps an edited file.",
        "accepted": "   {f} kept as it is",
        "status_video": "{id}   video: {video}",
        "status_params": "   settings: {p}",
        "list_none": "no analysis yet; to start: {me} new <video> <name>",
        "list_done": "finished",
        "list_next": "next step: {s} ({why})",
        "new": "new analysis: {dir}",
        "exists": "{name} already exists: choose another name, or carry on with: {me} run {name}",
        "no_video": "video not found: {v}",
        "unknown_stage": "unknown stage “{s}”; stages: {list}",
        "not_found": "analysis “{id}” not found or ambiguous (matches: {names}). See: {me} list",
        "none": "none",
        "not_analysis": "{dir} is not an analysis (no analysis.json)",
        "no_calib": "no calib.json in the analysis {src}",
        "unknown_setting": "unknown setting “{k}”; settings:\n{list}",
        "removed": "   {k} removed",
        "params": {"start": "end of the warm-up (s): everything before is ignored",
                   "lang": "language of the report, the reel and the messages: en or fr",
                   "coach": "auto (lessons written by rules) or ai (written by an AI assistant)",
                   "shorts_hue": "hue of your shorts [min,max] (OpenCV 0-179): blue [100,135], red [170,10], green [40,85]",
                   "min_blue": "minimum share of pixels of the shorts colour (0.35)",
                   "thr": "bend threshold of stage 9 (0.82)",
                   "last_winner": "winner of the last point if the video shows it: R (you) or O",
                   "speed": "playback speed of the rallies in the reel (1.3)",
                   "report_date": "date printed on the PDF report (default: the day the analysis was created)",
                   "camera": "floor (phone on the floor) or raised (camera 2 m or higher)",
                   "calib_from": "analysis whose calibration points are reused",
                   "accept_score": "serves that break the side rule but are right (fault, let...), e.g. [7]"},
        "bad_value": {"shorts_hue": "shorts_hue: two numbers, e.g. shorts_hue=[170,10]",
                      "accept_score": "accept_score: the list of accepted serve numbers, e.g. accept_score=[7]",
                      "number": "{k}: a number, e.g. {k}=12", "last_winner": "last_winner: R (you) or O (opponent)",
                      "camera": "camera: floor or raised", "lang": "lang: en or fr", "coach": "coach: auto or ai"},
    },
    "fr": {
        "stages": {"proxy": "0. copie de travail 30 i/s", "detect": "1. détection des joueurs", "calib": "2. calibrage du terrain",
                   "players": "3. positions des joueurs", "sheets": "4a. planches de lecture + services détectés",
                   "review": "lecture des services et fins d'échange", "serves": "4. services",
                   "score": "5. score + contrôle du côté de service", "rallies": "6. fins d'échange",
                   "movement": "7. déplacements", "pose": "8. posture", "pickups": "9. côté où tombe le volant",
                   "report": "10. chiffres du rapport", "coaching": "écriture des conseils", "pdf": "10. rapport PDF",
                   "video": "11. vidéo commentée"},
        "why": {"todo": "à faire", "forced": "refait demandé", "settings": "réglages modifiés", "changed": "{f} modifié",
                "reread": "{f} a changé : à relire", "start": "réglage start manquant", "redo": "refait"},
        "kept": "   (version gardée : {p}, {why})",
        "kept_why": {"used": "version utilisée", "redo": "étape refaite", "recalib": "avant recalibrage",
                     "regen": "regénéré"},
        "failed": "   ÉCHEC de l'étape « {s} » : voir le message ci-dessus.",
        "all_done": "Tout est à jour. Rapport : {pdf}\n               Vidéo : {video}",
        "calib_reused": "   points de calibrage repris de {src}",
        "calib_check": "   VÉRIFIE l'image {img} : le terrain jaune doit tomber sur les lignes.",
        "calib_moved": "   Si la caméra a bougé : {me} run {id} --redo calib",
        "calib_failed": "   ÉCHEC du calibrage repris : voir le message ci-dessus.",
        "calib_todo": "   À FAIRE : calibrer le terrain (une fenêtre s'ouvre) : {me} run {id}",
        "calib_window": "   Fenêtre de calibrage : lettre du point, puis clic dessus ; Entrée pour valider (voir s2_calibrate.py).",
        "calib_stopped": "   calibrage interrompu : rien n'a changé",
        "start_todo": "   À FAIRE : indiquer la fin de l'échauffement (en secondes), par exemple :\n     {me} set {id} start=12\n"
                      "   puis : {me} run {id}",
        "review_todo": "   À FAIRE : lire qui sert et quand chaque échange se termine.\n"
                       "     Planches : {rel}/contact/timeline (une image toutes les 0,5 s)\n"
                       "     Brouillon pré-rempli : {todo} (format : examples/review.example.json)\n"
                       "     Avec Claude Code ouvert dans ce dossier : « lis les planches de {rel} et écris review.json »\n"
                       "   puis : {me} run {id}",
        "score_bad": "   À VÉRIFIER : le(s) service(s) {bad} ne respecte(nt) pas la règle du côté de service.\n"
                     "     Le serveur ou un service est sans doute mal lu dans {rel}/review.json\n"
                     "     (gros plans : {rel}/contact/serve_closeups). Corrige, puis : {me} run {id}\n"
                     "     Si c'est vraiment ce qui s'est passé (faute, let...) : {me} set {id} accept_score={list}",
        "coach_reread": "   À RELIRE : les chiffres de {rel}/report.md ont changé depuis l'écriture de coaching.json.\n"
                        "     Demande à ton assistant IA : « mets à jour coaching.json pour {rel} »\n"
                        "     S'il est encore juste tel quel : {me} run {id} --accept coaching\n"
                        "     Ou le faire réécrire par les règles : {me} set {id} coach=auto, puis supprime coaching.json",
        "coach_todo": "   À FAIRE : écrire les conseils dans {rel}/coaching.json à partir de {rel}/report.md, avec la partie\n"
                      "     « pdf » (textes du rapport écrit, à la première personne). Demande à ton assistant IA :\n"
                      "     « écris coaching.json pour {rel} » (exemple : examples/coaching.example.json ; format en tête\n"
                      "     de s10_pdf.py). Ou le faire écrire par les règles : {me} set {id} coach=auto",
        "then": "   puis : {me} run {id}",
        "coach_auto": "   conseils écrits par les règles (s10_coach.py) ; pour des conseils plus précis, fais réécrire\n"
                      "   coaching.json par un assistant IA : run.py garde un fichier modifié.",
        "accepted": "   {f} gardé tel quel",
        "status_video": "{id}   vidéo : {video}",
        "status_params": "   réglages : {p}",
        "list_none": "aucune analyse ; pour commencer : {me} new <vidéo> <nom>",
        "list_done": "terminée",
        "list_next": "prochaine étape : {s} ({why})",
        "new": "nouvelle analyse : {dir}",
        "exists": "{name} existe déjà : choisis un autre nom, ou continue-la avec : {me} run {name}",
        "no_video": "vidéo introuvable : {v}",
        "unknown_stage": "étape inconnue « {s} » ; étapes : {list}",
        "not_found": "analyse « {id} » introuvable ou ambiguë (correspond à : {names}). Voir : {me} list",
        "none": "aucune",
        "not_analysis": "{dir} n'est pas une analyse (pas de analysis.json)",
        "no_calib": "pas de calib.json dans l'analyse {src}",
        "unknown_setting": "réglage inconnu « {k} » ; réglages :\n{list}",
        "removed": "   {k} supprimé",
        "params": {"start": "fin de l'échauffement (s) : tout ce qui est avant est ignoré",
                   "lang": "langue du rapport, de la vidéo et des messages : en ou fr",
                   "coach": "auto (conseils écrits par des règles) ou ai (écrits par un assistant IA)",
                   "shorts_hue": "teinte de ton short [min,max] (OpenCV 0-179) : bleu [100,135], rouge [170,10], vert [40,85]",
                   "min_blue": "part minimale de pixels de la couleur du short (0.35)",
                   "thr": "seuil de flexion de l'étape 9 (0.82)",
                   "last_winner": "gagnant du dernier point si la vidéo le montre : R (toi) ou O",
                   "speed": "vitesse de lecture des échanges dans la vidéo (1.3)",
                   "report_date": "date écrite sur le rapport PDF (par défaut : le jour de création de l'analyse)",
                   "camera": "floor (téléphone posé au sol) ou raised (caméra à 2 m ou plus)",
                   "calib_from": "analyse dont on reprend les points de calibrage",
                   "accept_score": "services qui contredisent la règle du côté mais sont justes (faute, let...), ex. [7]"},
        "bad_value": {"shorts_hue": "shorts_hue : deux nombres, ex. shorts_hue=[170,10]",
                      "accept_score": "accept_score : la liste des numéros de service acceptés, ex. accept_score=[7]",
                      "number": "{k} : un nombre, ex. {k}=12", "last_winner": "last_winner : R (toi) ou O (adversaire)",
                      "camera": "camera : floor ou raised", "lang": "lang : en ou fr", "coach": "coach : auto ou ai"},
    },
}


def msg(lang, key, **kw):
    s = MSG[lang][key]
    return s.format(**kw) if kw and isinstance(s, str) else s


sys.stdout.reconfigure(errors="replace")


# ---------------------------------------------------------------- files
def load(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def dump(obj, path):
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


_SHA = {}


def sha(path):
    """Content fingerprint (cached per size + mtime), None if the file does not exist."""
    if not path.exists():
        return None
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _SHA:
        _SHA[key] = hashlib.sha1(path.read_bytes()).hexdigest()
    return _SHA[key]


def num(x):
    """Settings as written on a command line: 12, 12.0 and "12" are the same value."""
    return f"{float(x):g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def resolve(ident, lang="en"):
    if (ANALYSES / ident / "analysis.json").exists():
        return ANALYSES / ident
    hits = [p for p in ANALYSES.iterdir() if (p / "analysis.json").exists() and ident in p.name] if ANALYSES.exists() else []
    if len(hits) != 1:
        raise SystemExit(msg(lang, "not_found", id=ident, names=", ".join(p.name for p in hits) or msg(lang, "none"), me=ME))
    return hits[0]


class Analysis:
    def __init__(self, folder):
        self.dir = folder
        self.meta = load(folder / "analysis.json")
        if self.meta is None:
            raise SystemExit(msg("en", "not_analysis", dir=folder))
        self.state = load(folder / STATE, {})
        self.state.setdefault("stages", {})
        self.state.setdefault("hand_written", {})
        self.stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    @property
    def params(self):
        return self.meta["params"]

    @property
    def lang(self):
        return self.params.get("lang", "fr")       # analyses made before lang existed were in French

    def m(self, key, **kw):
        return msg(self.lang, key, **kw)

    def set_params(self, **kv):
        """Write settings, re-reading analysis.json first so a concurrent `set` is not lost."""
        self.meta = load(self.dir / "analysis.json")
        for k, v in kv.items():
            if v is None:
                self.meta["params"].pop(k, None)
            else:
                self.meta["params"][k] = v
        dump(self.meta, self.dir / "analysis.json")

    def save(self):
        dump(self.state, self.dir / STATE)

    def keep(self, rel, why):
        """Copy a file into history/<run time>/ before it is replaced; never overwrites a kept copy."""
        src = self.dir / rel
        if not src.is_file():
            return
        dst = self.dir / "history" / self.stamp / rel
        n = 1
        while dst.exists():
            if sha(dst) == sha(src):
                return
            n += 1
            dst = dst.with_name(f"{Path(rel).stem}.{n}{Path(rel).suffix}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(self.m("kept", p=dst.relative_to(self.dir), why=self.m("kept_why")[why]))

    def snapshot_hand_written(self):
        for name in HAND_WRITTEN:
            h = sha(self.dir / name)
            if h and self.state["hand_written"].get(name) != h:
                self.keep(name, "used")
                self.state["hand_written"][name] = h


# ---------------------------------------------------------------- stages
class Stage:
    def __init__(self, key, outputs, inputs, cmds=None, manual=False, reads=()):
        """reads = settings the script takes from analysis.json itself (compared like its command line)."""
        self.key, self.outputs, self.inputs, self.cmds, self.manual = key, outputs, inputs, cmds, manual
        self.reads = reads


def stages(a):
    A, p = a.dir, a.params

    def f(rel):
        return str(A / rel)

    def script(name, *args):
        return [PY, str(ROOT / name), *map(num, args)]

    start = ["--start", p["start"]] if "start" in p else None
    hue = ["--shorts-hue", *p["shorts_hue"]] if "shorts_hue" in p else []
    min_blue = ["--min-blue", p["min_blue"]] if "min_blue" in p else []
    thr = ["--thr", p["thr"]] if "thr" in p else []
    last = ["--last-winner", p["last_winner"]] if p.get("last_winner") else []
    speed = ["--speed", p["speed"]] if "speed" in p else []
    lang = ["--lang", a.lang]
    # printed date: the setting, else the day the analysis was created (fixed, so the PDF is not rebuilt every day)
    date = p.get("report_date") or Text(a.lang).date(datetime.date.fromisoformat(a.meta["created"][:10]))
    return [
        Stage("proxy", ["proxy.mp4"], [], [script("s0_proxy.py", a.meta["video"], f("proxy.mp4"))]),
        Stage("detect", ["dets.json"], ["proxy.mp4"],
              [script("s1_detect.py", f("proxy.mp4"), MODELS / "yolox_s.onnx", f("dets.json"),
                      "--sheet", f("contact/s1_dets.jpg"))]),
        Stage("calib", ["calib.json"], [], manual=True),
        Stage("players", ["players.json"], ["dets.json", "calib.json"],
              start and [script("s3_players.py", f("proxy.mp4"), f("dets.json"), f("calib.json"), f("players.json"),
                                *start, *hue, *min_blue, "--sheet", f("contact/s3_players.jpg"),
                                "--plot", f("figures/s3_tracks.png"))]),
        Stage("sheets", ["serves_auto.json"], ["players.json"],
              start and [script("timeline_sheets.py", f("proxy.mp4"), f("players.json"), f("contact/timeline"), *start),
                         script("s4_serves.py", f("proxy.mp4"), f("players.json"), f("serves_auto.json"),
                                "--sheets", f("contact/serves_auto"))]),
        Stage("review", ["review.json"], [], manual=True),
        Stage("serves", ["serves.json"], ["players.json", "review.json"],
              [script("s4_serves.py", f("proxy.mp4"), f("players.json"), f("serves.json"), "--review", f("review.json"),
                      "--sheets", f("contact/serves"), "--closeups", f("contact/serve_closeups"))]),
        Stage("score", ["score.json"], ["serves.json"], [script("s5_score.py", f("serves.json"), f("score.json"), *last)]),
        Stage("rallies", ["points.json"], ["players.json", "score.json", "review.json"],
              [script("s6_rallies.py", f("proxy.mp4"), f("players.json"), f("score.json"), f("review.json"),
                      f("points.json"), "--sheets", f("contact/rally_ends"))]),
        Stage("movement", ["motion.json"], ["players.json", "points.json"],
              [script("s7_movement.py", f("players.json"), f("points.json"), f("motion.json"), "--figures", f("figures"))]),
        Stage("pose", ["pose_stats.json", "pose_frames.json"], ["players.json", "points.json"],
              [script("s8_pose.py", f("proxy.mp4"), f("players.json"), f("points.json"),
                      MODELS / "pose_landmarker_heavy.task", f("pose_stats.json"),
                      "--frames", f("pose_frames.json"), "--sheet", f("contact/s8_pose.jpg"))]),
        Stage("pickups", ["pickups.json"], ["players.json", "points.json", "review.json"],
              [script("s9_pickups.py", f("proxy.mp4"), f("players.json"), f("points.json"), f("pickups.json"),
                      "--review", f("review.json"), *thr, "--sheet", f("contact/s9_pickups.jpg"))]),
        Stage("report", ["report.json", "report.md"],
              ["players.json", "points.json", "review.json", "motion.json", "pose_stats.json", "pose_frames.json",
               "pickups.json"],
              [script("s10_report.py", A, *lang)], reads=("camera",)),
        Stage("coaching", ["coaching.json"], ["report.json"], [script("s10_coach.py", A, *lang)], manual=True),
        Stage("pdf", ["rapport_match.pdf"], ["report.json", "score.json", "coaching.json"],
              [script("s10_pdf.py", A, f("rapport_match.pdf"), "--date", date, *lang)],
              reads=("camera", "accept_score")),
        Stage("video", ["video/reel.mp4"], ["players.json", "calib.json", "points.json", "pose_frames.json", "coaching.json"],
              [script("s11_video.py", A, f("video/reel.mp4"), *speed, *lang)]),
    ]


STAGE_KEYS = ["proxy", "detect", "calib", "players", "sheets", "review", "serves", "score", "rallies", "movement",
              "pose", "pickups", "report", "coaching", "pdf", "video"]
REDO_KEYS = [k for k in STAGE_KEYS if k not in ("review", "coaching")]


def portable(s, a):
    """Commands (+ settings read from analysis.json) as compared between runs: independent of where the
    folders live and of the source video path (the analysis keeps its proxy.mp4 if the video is moved)."""
    if not s.cmds or s.manual:
        return None
    out = [[c.replace(str(a.meta["video"]), "{VIDEO}").replace(str(a.dir), "{A}").replace(str(ROOT), "{ROOT}")
            for c in cmd[1:]] for cmd in s.cmds]
    return out + [[f"{k}={json.dumps(a.params.get(k))}" for k in s.reads]] if s.reads else out


def record(a, s, inputs=None):
    a.state["stages"][s.key] = {"cmds": portable(s, a), "done": time.time(),
                                "inputs": inputs if inputs is not None else {i: sha(a.dir / i) for i in s.inputs},
                                "outputs": {o: sha(a.dir / o) for o in s.outputs}}


def up_to_date(a, s):
    """None if done and current, else why it has to run: (code, changed files)."""
    if not all((a.dir / o).exists() for o in s.outputs):
        return "todo", []
    rec = a.state["stages"].get(s.key)
    if rec and rec.get("force"):
        return "forced", []
    if rec is None or "inputs" not in rec:   # made before run.py existed (or by an older run.py): adopt
        record(a, s)
        return None
    changed = [i for i in s.inputs if rec["inputs"].get(i) != sha(a.dir / i)]
    if s.manual:
        # a hand-written file is out of date only if what it was written from changed and it was not edited since
        if changed and all(rec["outputs"].get(o) == sha(a.dir / o) for o in s.outputs):
            return "reread", changed
        if changed:
            record(a, s)
        return None
    if rec["cmds"] != portable(s, a):
        return "settings", []
    return ("changed", changed) if changed else None


def why_text(a, why):
    code, files = why
    return a.m("why")[code].format(f=", ".join(files))


def title(a, s):
    return a.m("stages")[s.key]


def next_step(a):
    for s in stages(a):
        why = up_to_date(a, s)
        if why:
            return s, why_text(a, why)
        if s.cmds is None and not s.manual and not (a.dir / s.outputs[0]).exists():
            return s, a.m("why")["start"]
    return None, None


# ---------------------------------------------------------------- manual steps
def calibrate(a, window, forced):
    calib, check = a.dir / "calib.json", a.dir / "contact/s2_calib_check.jpg"
    src = a.params.get("calib_from")
    ident = a.dir.name
    if not calib.exists() and src and not forced:
        c = load(resolve(src, a.lang) / "calib.json")
        if c is None:
            raise SystemExit(a.m("no_calib", src=src))
        if len(c["segments"]) == 1:              # same camera spot, whole new video
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                        str(a.dir / "proxy.mp4")], capture_output=True, text=True, check=True).stdout)
            c["segments"][0].update(t0=0.0, t1=dur, frame_t=0.0)
        tmp = a.dir / "calib.from.json"
        dump(c, tmp)
        cmd = [PY, str(ROOT / "s2_calibrate.py"), str(a.dir / "proxy.mp4"), str(tmp), "--recompute", "--check", str(check)]
        if subprocess.run(cmd, cwd=ROOT).returncode:
            print(a.m("calib_failed"))
            return False
        tmp.replace(calib)
        print(a.m("calib_reused", src=resolve(src, a.lang).name))
        print(a.m("calib_check", img=check))
        print(a.m("calib_moved", me=ME, id=ident))
        return True
    if not window:
        print(a.m("calib_todo", me=ME, id=ident))
        return False
    a.keep("calib.json", "recalib")
    print(a.m("calib_window"))
    cmd = [PY, str(ROOT / "s2_calibrate.py"), str(a.dir / "proxy.mp4"), str(calib), "--check", str(check)]
    if subprocess.run(cmd, cwd=ROOT).returncode or not calib.exists():
        print(a.m("calib_stopped"))
        return False
    print(a.m("calib_check", img=check))
    return True


def review_todo(a):
    """Draft review.json from the automatic serves; rewritten while nobody has edited it."""
    todo = a.dir / "review.todo.json"
    if todo.exists() and sha(todo) != a.state.get("todo_sha"):
        return todo                      # being filled in: never overwrite it
    auto = load(a.dir / "serves_auto.json")["serves"]
    dump({"about": "DRAFT: correct it, then rename it review.json (format: examples/review.example.json). serves: "
                   "contact time + server (R = you, O = opponent), detected automatically (time within about 2 s, "
                   "server guessed). rally_ends: last moment of live play of each rally + side where the shuttle came "
                   "down (R / O / null). Read on contact/timeline; confirm the server on contact/serve_closeups "
                   "(made at stage 4).",
          "serves": [{"t": s["t"], "server": s["server_guess"], "note": "auto, to check"} for s in auto],
          "rally_ends": []}, todo)
    a.state["todo_sha"] = sha(todo)
    return todo


def stop_message(a, key, why=None):
    rel, ident = a.dir.relative_to(ROOT).as_posix(), a.dir.name
    if key == "start":
        print(a.m("start_todo", me=ME, id=ident))
    elif key == "review":
        print(a.m("review_todo", rel=rel, todo=review_todo(a).relative_to(ROOT).as_posix(), me=ME, id=ident))
    elif key == "score":
        bad = load(a.dir / "score.json")["contradictions"]
        print(a.m("score_bad", bad=bad, rel=rel, me=ME, id=ident, list=json.dumps(bad, separators=(",", ":"))))
    elif key == "coaching":
        print(a.m("coach_reread" if why and why[0] == "reread" else "coach_todo", rel=rel, me=ME, id=ident))
        print(a.m("then", me=ME, id=ident))


def check_score(a):
    """False (and say why) when a serve breaks the service-court rule and was not accepted."""
    bad = set(load(a.dir / "score.json")["contradictions"]) - set(a.params.get("accept_score") or [])
    if bad:
        stop_message(a, "score")
        return False
    return True


def auto_coaching(a):
    """True when the rules may (re)write coaching.json: coach=auto and the file is missing or still exactly
    what they wrote last time."""
    if a.params.get("coach", "ai") != "auto":
        return False
    c = a.dir / "coaching.json"
    return not c.exists() or sha(c) == a.state.get("coach_auto_sha")


def run_cmds(a, s):
    """Run a stage's commands; True if all succeeded, None if one failed."""
    for cmd in s.cmds:
        print("   " + " ".join(Path(c).name if i < 2 else c for i, c in enumerate(cmd)), flush=True)
        if subprocess.run(cmd, cwd=ROOT).returncode:
            print(a.m("failed", s=title(a, s)))
            return None
    return True


# ---------------------------------------------------------------- commands
def run(a, redo=None, accept=None, window=True):
    a.snapshot_hand_written()
    if redo:
        if redo not in REDO_KEYS:
            raise SystemExit(a.m("unknown_stage", s=redo, list=", ".join(REDO_KEYS)))
        a.state["stages"].setdefault(redo, {})["force"] = True
    if accept:
        record(a, next(s for s in stages(a) if s.key == accept))
        print(a.m("accepted", f=f"{accept}.json"))
    a.save()
    for s in stages(a):
        why = up_to_date(a, s)
        if why is None:
            if s.key == "score" and not check_score(a):
                return
            continue
        print(f"== {title(a, s)} ({why_text(a, why)})")
        inputs = {i: sha(a.dir / i) for i in s.inputs}     # what this run reads
        if s.key == "calib":
            ok = calibrate(a, window, forced=why[0] == "forced")
        elif s.key == "coaching" and auto_coaching(a):
            a.keep("coaching.json", "regen")
            ok = run_cmds(a, s)
            if ok:
                a.state["coach_auto_sha"] = sha(a.dir / "coaching.json")
                print(a.m("coach_auto"))
        elif s.manual:
            stop_message(a, s.key, why)
            ok = False
        elif s.cmds is None:
            stop_message(a, "start")
            ok = False
        else:
            for o in s.outputs:
                if o != "proxy.mp4":
                    a.keep(o, "redo")
            ok = run_cmds(a, s)
        if ok:
            record(a, s, inputs)
            a.snapshot_hand_written()
        elif ok is None:                     # failed: its files may be half written, redo it next time
            a.state["stages"][s.key] = {"force": True}
        a.save()
        if not ok or s.key == "score" and not check_score(a):
            return
    print(a.m("all_done", pdf=a.dir / "rapport_match.pdf", video=a.dir / "video" / "reel.mp4"))


def status(a):
    print(a.m("status_video", id=a.dir.name, video=a.meta["video"]))
    print(a.m("status_params", p=", ".join(f"{k}={v}" for k, v in a.params.items())))
    for s in stages(a):
        why = up_to_date(a, s)
        text = why_text(a, why) if why else None
        if why is None and s.cmds is None and not s.manual and not (a.dir / s.outputs[0]).exists():
            text = a.m("why")["start"]
        print(f"   {'[ok]' if text is None else '[  ]'} {title(a, s)}{'' if text is None else '  -> ' + text}")
    a.save()


def parse_value(k, v, lang):
    bad = MSG[lang]["bad_value"]
    try:
        x = json.loads(v)
    except json.JSONDecodeError:
        x = v
    if k == "shorts_hue":
        parts = re.findall(r"-?\d+", v)
        if len(parts) != 2:
            raise SystemExit(bad["shorts_hue"])
        return [int(n) for n in parts]
    if k == "accept_score" and not (isinstance(x, list) and all(isinstance(n, int) for n in x)):
        raise SystemExit(bad["accept_score"])
    if k in ("start", "min_blue", "thr", "speed") and not isinstance(x, (int, float)):
        raise SystemExit(bad["number"].format(k=k))
    if k == "last_winner" and x not in ("R", "O"):
        raise SystemExit(bad["last_winner"])
    if k == "camera":
        if x not in CAMERA:
            raise SystemExit(bad["camera"])
        return CAMERA[x]
    if k == "lang" and x not in ("en", "fr"):
        raise SystemExit(bad["lang"])
    if k == "coach" and x not in ("auto", "ai"):
        raise SystemExit(bad["coach"])
    if k == "calib_from":
        return resolve(str(x), lang).name
    return x


def cmd_new(args):
    lang = args.lang
    video = Path(args.video).resolve()
    if not video.exists():
        raise SystemExit(msg(lang, "no_video", v=video))
    day = args.date or datetime.date.fromtimestamp(video.stat().st_mtime).isoformat()
    slug = re.sub(r"[^a-z0-9]+", "-", args.name.lower()).strip("-")
    folder = ANALYSES / f"{day}_{slug}"
    if folder.exists():
        raise SystemExit(msg(lang, "exists", name=folder.name, me=ME))
    params = {"lang": lang, "coach": args.coach}
    if args.start is not None:
        params["start"] = args.start
    if args.shorts_hue:
        params["shorts_hue"] = args.shorts_hue
    if args.calib_from:
        params["calib_from"] = resolve(args.calib_from, lang).name
    if args.camera:
        params["camera"] = CAMERA[args.camera]
    folder.mkdir(parents=True)
    dump({"id": folder.name, "video": str(video), "created": datetime.datetime.now().isoformat(timespec="seconds"),
          "params": params}, folder / "analysis.json")
    print(msg(lang, "new", dir=folder))
    run(Analysis(folder), window=not args.no_window)


def cmd_set(args):
    a = Analysis(resolve(args.id))
    new = {}
    for kv in args.pairs:
        k, _, v = kv.partition("=")
        if k not in MSG["en"]["params"]:
            raise SystemExit(a.m("unknown_setting", k=k, list="\n".join(f"  {n}: {d}" for n, d in a.m("params").items())))
        new[k] = parse_value(k, v, a.lang) if v != "" else None
        print(f"   {k} = {new[k]}" if new[k] is not None else a.m("removed", k=k))
    a.set_params(**new)


def doctor():
    """Check Python, packages, ffmpeg, models, fonts and write access; say how to fix what fails."""
    ok = True

    def line(good, what, fix=""):
        nonlocal ok
        ok &= bool(good) or fix.startswith("(optional")
        print(f"  [{'ok' if good else '!!'}] {what}" + ("" if good or not fix else f"\n        -> {fix}"))

    v = sys.version_info
    line(v[:2] >= (3, 10), f"Python {v.major}.{v.minor} ({PY})", "use Python 3.12 (the tested version)")
    if v[:2] != (3, 12):
        print("  [..] tested with Python 3.12; if a package fails to install, use 3.12")
    if ".venv" not in PY:
        print("  [..] not running from the project's .venv: run " + ME.replace(" run.py", " run.py doctor"))
    for mod, pip, optional in (("numpy", "numpy", False), ("cv2", "opencv-contrib-python", False),
                               ("onnxruntime", "onnxruntime", False), ("mediapipe", "mediapipe", False),
                               ("matplotlib", "matplotlib", False), ("PIL", "pillow", False),
                               ("reportlab", "reportlab", False), ("kokoro_onnx", "kokoro-onnx", True),
                               ("soundfile", "soundfile", True)):
        try:
            m = __import__(mod)
            line(True, f"{pip} {getattr(m, '__version__', '')}".strip())
        except Exception as e:                       # noqa: BLE001  (any import failure is worth reporting)
            fix = f"pip install -r requirements.txt  ({type(e).__name__}: {e})"
            line(False, pip, ("(optional, for the video reel) " if optional else "") + fix)
    for exe in ("ffmpeg", "ffprobe"):
        path = shutil.which(exe)
        line(path, f"{exe}: {path or 'not found'}",
             "install ffmpeg and put it on the PATH (Windows: winget install Gyan.FFmpeg, then open a new terminal)")
    for name, size, optional in (("yolox_s.onnx", 30e6, False), ("pose_landmarker_heavy.task", 25e6, False),
                                 ("kokoro-v1.0.onnx", 300e6, True), ("voices-v1.0.bin", 25e6, True)):
        f = MODELS / name
        good = f.exists() and f.stat().st_size >= size
        line(good, f"models/{name}" + (f" ({f.stat().st_size / 1e6:.0f} MB)" if f.exists() else ""),
             ("(optional, for the video reel) " if optional else "") + "python download_models.py")
    try:
        from bball.fonts import font
        fonts = [font(s) for s in ("regular", "bold", "italic", "display")]
        line(all(Path(f).exists() for f in fonts), "fonts: " + ", ".join(Path(f).name for f in fonts),
             "fonts not found: install matplotlib")
    except Exception as e:                           # noqa: BLE001
        line(False, f"fonts ({e})", "pip install -r requirements.txt")
    try:
        ANALYSES.mkdir(exist_ok=True)
        probe = ANALYSES / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        line(True, f"write access to {ANALYSES}")
    except OSError as e:
        line(False, f"write access to {ANALYSES} ({e})",
             "on Windows with Controlled folder access on, allow python.exe and ffmpeg.exe "
             "(Windows Security > Ransomware protection), or move the project out of Documents")
    print("Ready." if ok else "Fix the items marked !! and run doctor again.")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check the installation")
    p = sub.add_parser("new", help="new analysis of a video")
    p.add_argument("video")
    p.add_argument("name", help="short name, e.g. match-paul")
    p.add_argument("--start", type=float, help="end of the warm-up (s)")
    p.add_argument("--lang", choices=("en", "fr"), default="en", help="language of the report, reel and messages")
    p.add_argument("--coach", choices=("auto", "ai"), default="auto",
                   help="lessons written by rules (auto, default) or by an AI assistant (ai)")
    p.add_argument("--shorts-hue", type=int, nargs=2, metavar=("LO", "HI"), help="hue of your shorts (default blue 100 135)")
    p.add_argument("--calib-from", help="reuse the calibration of another analysis (same camera spot)")
    p.add_argument("--camera", choices=("floor", "raised", "sol", "haute"), help="camera on the floor (default) or raised")
    p.add_argument("--date", help="match date YYYY-MM-DD (default: date of the video file)")
    p.add_argument("--no-window", action="store_true", help="do not open the calibration window")
    p = sub.add_parser("run", help="do the missing or out-of-date stages")
    p.add_argument("id")
    p.add_argument("--redo", help="redo this stage (and what depends on it): " + ", ".join(REDO_KEYS))
    p.add_argument("--accept", choices=("coaching",), help="keep coaching.json as it is although the numbers changed")
    p.add_argument("--no-window", action="store_true", help="do not open the calibration window")
    p = sub.add_parser("set", help="change a setting, e.g. start=12")
    p.add_argument("id")
    p.add_argument("pairs", nargs="+", metavar="key=value")
    p = sub.add_parser("status", help="state of every stage of an analysis")
    p.add_argument("id")
    sub.add_parser("list", help="all analyses")
    args = ap.parse_args()

    if args.cmd == "doctor":
        sys.exit(0 if doctor() else 1)
    try:
        import cv2  # noqa: F401  (only to check that this is the project's Python)
    except ImportError:
        raise SystemExit(f"This Python does not have the project's packages. Run: {ME} doctor")
    if args.cmd == "new":
        cmd_new(args)
    elif args.cmd == "run":
        run(Analysis(resolve(args.id)), redo=args.redo, accept=args.accept, window=not args.no_window)
    elif args.cmd == "set":
        cmd_set(args)
    elif args.cmd == "status":
        status(Analysis(resolve(args.id)))
    else:
        folders = sorted(p for p in ANALYSES.iterdir() if (p / "analysis.json").exists()) if ANALYSES.exists() else []
        if not folders:
            print(msg("en", "list_none", me=ME))
        for folder in folders:
            a = Analysis(folder)
            s, why = next_step(a)
            a.save()
            print(f"{folder.name:32s} {a.m('list_done') if s is None else a.m('list_next', s=title(a, s), why=why)}")


if __name__ == "__main__":
    main()
