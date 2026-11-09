"""Full pipeline, one folder per analysis (analyses/<date>_<name>/): nothing is ever overwritten.

  .venv\\Scripts\\python.exe run.py new <video> <name> [--start 12] [--calib-from ID] [--shorts-hue 100 135]
  .venv\\Scripts\\python.exe run.py run <ID> [--redo STAGE] [--accept coaching] [--no-window]
  .venv\\Scripts\\python.exe run.py set <ID> start=12 thr=0.8 shorts_hue=[170,10] ...
  .venv\\Scripts\\python.exe run.py status <ID>
  .venv\\Scripts\\python.exe run.py list

`run` does every stage that is missing or out of date (the content of one of its input files or
one of its settings changed since it last ran), in order, and stops where a person has to act:
  - the court calibration (stage 2, a window opens; --calib-from reuses another analysis' points),
  - review.json: serves and rally ends read by eye on the contact sheets (Claude does it),
  - a serve that breaks the service-court rule (score check),
  - coaching.json: the lessons, written from report.md (Claude does it); asked again whenever the
    numbers of report.json change afterwards.
Then run it again: it carries on from there. Before a stage replaces a result file, the old version
is copied to <analysis>/history/<date-time>/, and so is every version of the hand-written files
(calib, review, coaching). Not kept: proxy.mp4 (rebuilt from the video), contact sheets and figures
(rebuilt from the results).
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

ROOT = Path(__file__).resolve().parent
ANALYSES = ROOT / "analyses"
MODELS = ROOT / "models"
PY = sys.executable
ME = r".venv\Scripts\python.exe run.py" if os.name == "nt" else ".venv/bin/python run.py"   # as printed in messages
STATE = ".pipeline.json"
HAND_WRITTEN = ("calib.json", "review.json", "coaching.json")
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
          "novembre", "décembre"]
PARAMS = {"start": "fin de l'échauffement (s) : tout ce qui est avant est ignoré",
          "shorts_hue": "teinte de ton short [min,max] (OpenCV 0-179) : bleu [100,135], rouge [170,10], vert [40,85]",
          "min_blue": "part minimale de pixels de la couleur du short (0.35)",
          "thr": "seuil de flexion de l'étape 9 (0.82)",
          "last_winner": "gagnant du dernier point si la vidéo le montre : R (toi) ou O",
          "speed": "vitesse de lecture des échanges dans la vidéo (1.3)",
          "report_date": "date écrite sur le rapport PDF",
          "camera": "sol (téléphone posé au sol) ou haute (caméra surélevée)",
          "calib_from": "analyse dont on reprend les points de calibrage",
          "accept_score": "services qui contredisent la règle du côté mais sont justes (faute, let...), ex. [7]"}

sys.stdout.reconfigure(errors="replace")
try:
    import cv2  # noqa: F401  (only to check that this is the project's Python)
except ImportError:
    raise SystemExit(f"Ce Python ne contient pas les outils du projet. Lance : {ME} ...")


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


def resolve(ident):
    if (ANALYSES / ident / "analysis.json").exists():
        return ANALYSES / ident
    hits = [p for p in ANALYSES.iterdir() if (p / "analysis.json").exists() and ident in p.name] if ANALYSES.exists() else []
    if len(hits) != 1:
        names = ", ".join(p.name for p in hits) or "aucune"
        raise SystemExit(f"analyse « {ident} » introuvable ou ambiguë (correspond à : {names}). Voir : {ME} list")
    return hits[0]


class Analysis:
    def __init__(self, folder):
        self.dir = folder
        self.meta = load(folder / "analysis.json")
        if self.meta is None:
            raise SystemExit(f"{folder} n'est pas une analyse (pas de analysis.json)")
        self.state = load(folder / STATE, {})
        self.state.setdefault("stages", {})
        self.state.setdefault("hand_written", {})
        self.stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        if "report_date" not in self.params:      # fixed once, so the PDF is not rebuilt every new day
            today = datetime.date.today()
            self.set_params(report_date=f"{today.day} {MONTHS[today.month - 1]} {today.year}")

    @property
    def params(self):
        return self.meta["params"]

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
        print(f"   (version gardée : {dst.relative_to(self.dir)}, {why})")

    def snapshot_hand_written(self):
        for name in HAND_WRITTEN:
            h = sha(self.dir / name)
            if h and self.state["hand_written"].get(name) != h:
                self.keep(name, "version utilisée")
                self.state["hand_written"][name] = h


# ---------------------------------------------------------------- stages
class Stage:
    def __init__(self, key, title, outputs, inputs, cmds=None, manual=False, reads=()):
        """reads = settings the script takes from analysis.json itself (compared like its command line)."""
        self.key, self.title, self.outputs, self.inputs, self.cmds, self.manual = key, title, outputs, inputs, cmds, manual
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
    return [
        Stage("proxy", "0. copie de travail 30 i/s", ["proxy.mp4"], [],
              [script("s0_proxy.py", a.meta["video"], f("proxy.mp4"))]),
        Stage("detect", "1. détection des joueurs", ["dets.json"], ["proxy.mp4"],
              [script("s1_detect.py", f("proxy.mp4"), MODELS / "yolox_s.onnx", f("dets.json"),
                      "--sheet", f("contact/s1_dets.jpg"))]),
        Stage("calib", "2. calibrage du terrain", ["calib.json"], [], manual=True),
        Stage("players", "3. positions des joueurs", ["players.json"], ["dets.json", "calib.json"],
              start and [script("s3_players.py", f("proxy.mp4"), f("dets.json"), f("calib.json"), f("players.json"),
                                *start, *hue, *min_blue, "--sheet", f("contact/s3_players.jpg"),
                                "--plot", f("figures/s3_tracks.png"))]),
        Stage("sheets", "4a. planches de lecture + services détectés", ["serves_auto.json"], ["players.json"],
              start and [script("timeline_sheets.py", f("proxy.mp4"), f("players.json"), f("contact/timeline"), *start),
                         script("s4_serves.py", f("proxy.mp4"), f("players.json"), f("serves_auto.json"),
                                "--sheets", f("contact/serves_auto"))]),
        Stage("review", "lecture des services et fins d'échange", ["review.json"], [], manual=True),
        Stage("serves", "4. services", ["serves.json"], ["players.json", "review.json"],
              [script("s4_serves.py", f("proxy.mp4"), f("players.json"), f("serves.json"), "--review", f("review.json"),
                      "--sheets", f("contact/serves"), "--closeups", f("contact/serve_closeups"))]),
        Stage("score", "5. score + contrôle du côté de service", ["score.json"], ["serves.json"],
              [script("s5_score.py", f("serves.json"), f("score.json"), *last)]),
        Stage("rallies", "6. fins d'échange", ["points.json"], ["players.json", "score.json", "review.json"],
              [script("s6_rallies.py", f("proxy.mp4"), f("players.json"), f("score.json"), f("review.json"),
                      f("points.json"), "--sheets", f("contact/rally_ends"))]),
        Stage("movement", "7. déplacements", ["motion.json"], ["players.json", "points.json"],
              [script("s7_movement.py", f("players.json"), f("points.json"), f("motion.json"), "--figures", f("figures"))]),
        Stage("pose", "8. posture", ["pose_stats.json", "pose_frames.json"], ["players.json", "points.json"],
              [script("s8_pose.py", f("proxy.mp4"), f("players.json"), f("points.json"),
                      MODELS / "pose_landmarker_heavy.task", f("pose_stats.json"),
                      "--frames", f("pose_frames.json"), "--sheet", f("contact/s8_pose.jpg"))]),
        Stage("pickups", "9. côté où tombe le volant", ["pickups.json"], ["players.json", "points.json", "review.json"],
              [script("s9_pickups.py", f("proxy.mp4"), f("players.json"), f("points.json"), f("pickups.json"),
                      "--review", f("review.json"), *thr, "--sheet", f("contact/s9_pickups.jpg"))]),
        Stage("report", "10. chiffres du rapport", ["report.json", "report.md"],
              ["players.json", "points.json", "review.json", "motion.json", "pose_stats.json", "pose_frames.json",
               "pickups.json"],
              [script("s10_report.py", A)], reads=("camera",)),
        Stage("coaching", "écriture des conseils", ["coaching.json"], ["report.json"], manual=True),
        Stage("pdf", "10. rapport PDF", ["rapport_match.pdf"], ["report.json", "score.json", "coaching.json"],
              [script("s10_pdf.py", A, f("rapport_match.pdf"), "--date", p["report_date"])], reads=("camera", "accept_score")),
        Stage("video", "11. vidéo commentée", ["video/reel.mp4"],
              ["players.json", "calib.json", "points.json", "pose_frames.json", "coaching.json"],
              [script("s11_video.py", A, f("video/reel.mp4"), *speed)]),
    ]


STAGE_KEYS = ["proxy", "detect", "calib", "players", "sheets", "review", "serves", "score", "rallies", "movement",
              "pose", "pickups", "report", "coaching", "pdf", "video"]


def portable(s, a):
    """Commands (+ settings read from analysis.json) as compared between runs: independent of where the
    folders live and of the source video path (the analysis keeps its proxy.mp4 if the video is moved)."""
    if not s.cmds:
        return None
    out = [[c.replace(str(a.meta["video"]), "{VIDEO}").replace(str(a.dir), "{A}").replace(str(ROOT), "{ROOT}")
            for c in cmd[1:]] for cmd in s.cmds]
    return out + [[f"{k}={json.dumps(a.params.get(k))}" for k in s.reads]] if s.reads else out


def record(a, s, inputs=None):
    a.state["stages"][s.key] = {"cmds": portable(s, a), "done": time.time(),
                                "inputs": inputs if inputs is not None else {i: sha(a.dir / i) for i in s.inputs},
                                "outputs": {o: sha(a.dir / o) for o in s.outputs}}


def up_to_date(a, s):
    """None if done and current, else the reason it has to run."""
    if not all((a.dir / o).exists() for o in s.outputs):
        return "à faire"
    rec = a.state["stages"].get(s.key)
    if rec and rec.get("force"):
        return "refait demandé"
    if rec is None or "inputs" not in rec:   # made before run.py existed (or by an older run.py): adopt
        record(a, s)
        return None
    changed = [i for i in s.inputs if rec["inputs"].get(i) != sha(a.dir / i)]
    if s.manual:
        # a hand-written file is out of date only if what it was written from changed and it was not edited since
        if changed and all(rec["outputs"].get(o) == sha(a.dir / o) for o in s.outputs):
            return f"{', '.join(changed)} a changé : à relire"
        if changed:
            record(a, s)
        return None
    if rec["cmds"] != portable(s, a):
        return "réglages modifiés"
    return f"{', '.join(changed)} modifié" if changed else None


def next_step(a):
    for s in stages(a):
        why = up_to_date(a, s)
        if why:
            return s, why
        if s.cmds is None and not s.manual and not (a.dir / s.outputs[0]).exists():
            return s, "réglage start manquant"
    return None, None


# ---------------------------------------------------------------- manual steps
def calibrate(a, window, forced):
    calib, check = a.dir / "calib.json", a.dir / "contact/s2_calib_check.jpg"
    src = a.params.get("calib_from")
    if not calib.exists() and src and not forced:
        c = load(resolve(src) / "calib.json")
        if c is None:
            raise SystemExit(f"pas de calib.json dans l'analyse {src}")
        if len(c["segments"]) == 1:              # same camera spot, whole new video
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                        str(a.dir / "proxy.mp4")], capture_output=True, text=True, check=True).stdout)
            c["segments"][0].update(t0=0.0, t1=dur, frame_t=0.0)
        tmp = a.dir / "calib.from.json"
        dump(c, tmp)
        cmd = [PY, str(ROOT / "s2_calibrate.py"), str(a.dir / "proxy.mp4"), str(tmp), "--recompute", "--check", str(check)]
        if subprocess.run(cmd, cwd=ROOT).returncode:
            print("   ÉCHEC du calibrage repris : voir le message ci-dessus.")
            return False
        tmp.replace(calib)
        print(f"   points de calibrage repris de {resolve(src).name}")
        print(f"   VÉRIFIE l'image {check} : le terrain jaune doit tomber sur les lignes.\n"
              f"   Si la caméra a bougé : {ME} run {a.dir.name} --redo calib")
        return True
    if not window:
        print(f"   À FAIRE : calibrer le terrain (une fenêtre s'ouvre) : {ME} run {a.dir.name}")
        return False
    a.keep("calib.json", "avant recalibrage")
    print("   Fenêtre de calibrage : lettre du point, puis clic dessus ; Entrée pour valider (voir s2_calibrate.py).")
    cmd = [PY, str(ROOT / "s2_calibrate.py"), str(a.dir / "proxy.mp4"), str(calib), "--check", str(check)]
    if subprocess.run(cmd, cwd=ROOT).returncode or not calib.exists():
        print("   calibrage interrompu : rien n'a changé")
        return False
    print(f"   VÉRIFIE l'image {check} : le terrain jaune doit tomber sur les lignes.")
    return True


def review_todo(a):
    """Draft review.json from the automatic serves; rewritten while nobody has edited it."""
    todo = a.dir / "review.todo.json"
    if todo.exists() and sha(todo) != a.state.get("todo_sha"):
        return todo                      # being filled in: never overwrite it
    auto = load(a.dir / "serves_auto.json")["serves"]
    dump({"about": "BROUILLON à corriger puis renommer en review.json. serves: heure de frappe + serveur (R = toi, O = adversaire), "
                   "détectés automatiquement (heure à ±2 s près, serveur deviné). rally_ends: dernier moment de jeu de chaque "
                   "échange + côté où tombe le volant (R / O / null). Lire sur contact/timeline, confirmer le serveur sur "
                   "contact/serve_closeups (créé à l'étape 4).",
          "serves": [{"t": s["t"], "server": s["server_guess"], "note": "auto, à vérifier"} for s in auto],
          "rally_ends": []}, todo)
    a.state["todo_sha"] = sha(todo)
    return todo


def stop_message(a, key, why=None):
    rel, ident = a.dir.relative_to(ROOT), a.dir.name
    if key == "start":
        print(f"   À FAIRE : indiquer la fin de l'échauffement (en secondes), par exemple :\n"
              f"     {ME} set {ident} start=12\n   puis : {ME} run {ident}")
    elif key == "review":
        todo = review_todo(a)
        print(f"   À FAIRE : lire qui sert et quand chaque échange se termine.\n"
              f"     Planches : {rel}\\contact\\timeline (une image toutes les 0,5 s)\n"
              f"     Brouillon pré-rempli : {todo.relative_to(ROOT)}\n"
              f"     Demande à Claude : « lis les planches de {rel} et écris review.json »\n"
              f"   puis : {ME} run {ident}")
    elif key == "score":
        bad = load(a.dir / "score.json")["contradictions"]
        print(f"   À VÉRIFIER : le(s) service(s) {bad} ne respecte(nt) pas la règle du côté de service.\n"
              f"     Le serveur ou un service est sans doute mal lu dans {rel}\\review.json\n"
              f"     (gros plans : {rel}\\contact\\serve_closeups). Corrige, puis : {ME} run {ident}\n"
              f"     Si c'est vraiment ce qui s'est passé (faute, let...) : {ME} set {ident} accept_score={json.dumps(bad, separators=(',', ':'))}")
    elif key == "coaching":
        if why and "changé" in why:
            print(f"   À RELIRE : les chiffres de {rel}\\report.md ont changé depuis l'écriture de coaching.json.\n"
                  f"     Demande à Claude : « mets à jour coaching.json pour {rel} »\n"
                  f"     S'il est encore juste tel quel : {ME} run {ident} --accept coaching")
        else:
            print(f"   À FAIRE : écrire les conseils dans {rel}\\coaching.json à partir de {rel}\\report.md,\n"
                  f"     avec la partie « pdf » (textes du rapport écrit, à la première personne).\n"
                  f"     Demande à Claude : « écris coaching.json pour {rel} »\n"
                  f"     (modèle : examples\\coaching.example.json ; format en tête de s10_pdf.py)")
        print(f"   puis : {ME} run {ident}")


def check_score(a):
    """False (and say why) when a serve breaks the service-court rule and was not accepted."""
    bad = set(load(a.dir / "score.json")["contradictions"]) - set(a.params.get("accept_score") or [])
    if bad:
        stop_message(a, "score")
        return False
    return True


# ---------------------------------------------------------------- commands
def run(a, redo=None, accept=None, window=True):
    a.snapshot_hand_written()
    if redo:
        if redo not in STAGE_KEYS or redo in ("review", "coaching"):
            raise SystemExit(f"étape inconnue « {redo} » ; étapes : {', '.join(k for k in STAGE_KEYS if k not in ('review', 'coaching'))}")
        a.state["stages"].setdefault(redo, {})["force"] = True
    if accept:
        s = next(s for s in stages(a) if s.key == accept)
        record(a, s)
        print(f"   {accept}.json gardé tel quel")
    a.save()
    for s in stages(a):
        why = up_to_date(a, s)
        if why is None:
            if s.key == "score" and not check_score(a):
                return
            continue
        print(f"== {s.title} ({why})")
        inputs = {i: sha(a.dir / i) for i in s.inputs}     # what this run reads
        if s.key == "calib":
            ok = calibrate(a, window, forced=why == "refait demandé")
        elif s.manual:
            stop_message(a, s.key, why)
            ok = False
        elif s.cmds is None:
            stop_message(a, "start")
            ok = False
        else:
            for o in s.outputs:
                if o != "proxy.mp4":
                    a.keep(o, "étape refaite")
            ok = True
            for cmd in s.cmds:
                print("   " + " ".join(Path(c).name if i < 2 else c for i, c in enumerate(cmd)), flush=True)
                if subprocess.run(cmd, cwd=ROOT).returncode:
                    print(f"   ÉCHEC de l'étape « {s.title} » : voir le message ci-dessus.")
                    ok = None
                    break
        if ok:
            record(a, s, inputs)
            a.snapshot_hand_written()
        elif ok is None:                     # failed: its files may be half written, redo it next time
            a.state["stages"][s.key] = {"force": True}
        a.save()
        if not ok or s.key == "score" and not check_score(a):
            return
    print(f"Tout est à jour. Rapport : {a.dir / 'rapport_match.pdf'}\n               Vidéo : {a.dir / 'video/reel.mp4'}")


def status(a):
    print(f"{a.dir.name}   vidéo : {a.meta['video']}")
    print("   réglages : " + ", ".join(f"{k}={v}" for k, v in a.params.items()))
    for s in stages(a):
        why = up_to_date(a, s)
        if why is None and s.cmds is None and not s.manual and not (a.dir / s.outputs[0]).exists():
            why = "réglage start manquant"
        print(f"   {'[ok]' if why is None else '[  ]'} {s.title}{'' if why is None else '  -> ' + why}")
    a.save()


def parse_value(k, v):
    try:
        x = json.loads(v)
    except json.JSONDecodeError:
        x = v
    if k == "shorts_hue":
        parts = re.findall(r"-?\d+", v)
        if len(parts) != 2:
            raise SystemExit("shorts_hue : deux nombres, ex. shorts_hue=[170,10]")
        return [int(n) for n in parts]
    if k == "accept_score":
        if not (isinstance(x, list) and all(isinstance(n, int) for n in x)):
            raise SystemExit("accept_score : la liste des numéros de service acceptés, ex. accept_score=[7]")
    if k in ("start", "min_blue", "thr", "speed") and not isinstance(x, (int, float)):
        raise SystemExit(f"{k} : un nombre, ex. {k}=12")
    if k == "last_winner" and x not in ("R", "O"):
        raise SystemExit("last_winner : R (toi) ou O (adversaire)")
    if k == "camera" and x not in ("sol", "haute"):
        raise SystemExit("camera : sol ou haute")
    if k == "calib_from":
        return resolve(str(x)).name
    return x


def cmd_new(args):
    video = Path(args.video).resolve()
    if not video.exists():
        raise SystemExit(f"vidéo introuvable : {video}")
    day = args.date or datetime.date.fromtimestamp(video.stat().st_mtime).isoformat()
    slug = re.sub(r"[^a-z0-9]+", "-", args.name.lower()).strip("-")
    folder = ANALYSES / f"{day}_{slug}"
    if folder.exists():
        raise SystemExit(f"{folder.name} existe déjà : choisis un autre nom, ou continue-la avec : {ME} run {folder.name}")
    params = {}
    if args.start is not None:
        params["start"] = args.start
    if args.shorts_hue:
        params["shorts_hue"] = args.shorts_hue
    if args.calib_from:
        params["calib_from"] = resolve(args.calib_from).name
    if args.camera:
        params["camera"] = args.camera
    folder.mkdir(parents=True)
    dump({"id": folder.name, "video": str(video), "created": datetime.datetime.now().isoformat(timespec="seconds"),
          "params": params}, folder / "analysis.json")
    print(f"nouvelle analyse : {folder}")
    run(Analysis(folder), window=not args.no_window)


def cmd_set(args):
    a = Analysis(resolve(args.id))
    new = {}
    for kv in args.pairs:
        k, _, v = kv.partition("=")
        if k not in PARAMS:
            raise SystemExit(f"réglage inconnu « {k} » ; réglages :\n" + "\n".join(f"  {n} : {d}" for n, d in PARAMS.items()))
        new[k] = parse_value(k, v) if v != "" else None
        print(f"   {k} = {new[k]}" if new[k] is not None else f"   {k} supprimé")
    a.set_params(**new)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new", help="nouvelle analyse d'une vidéo")
    p.add_argument("video")
    p.add_argument("name", help="nom court, ex. « match-paul »")
    p.add_argument("--start", type=float, help="fin de l'échauffement (s)")
    p.add_argument("--shorts-hue", type=int, nargs=2, metavar=("LO", "HI"), help="teinte du short (défaut bleu 100 135)")
    p.add_argument("--calib-from", help="reprendre le calibrage d'une autre analyse (même emplacement de caméra)")
    p.add_argument("--camera", choices=("sol", "haute"), help="caméra posée au sol (défaut) ou surélevée")
    p.add_argument("--date", help="date du match AAAA-MM-JJ (défaut : date du fichier vidéo)")
    p.add_argument("--no-window", action="store_true", help="ne pas ouvrir la fenêtre de calibrage")
    p = sub.add_parser("run", help="faire les étapes manquantes ou à refaire")
    p.add_argument("id")
    p.add_argument("--redo", help="refaire cette étape (et ce qui en dépend) : " +
                   ", ".join(k for k in STAGE_KEYS if k not in ("review", "coaching")))
    p.add_argument("--accept", choices=("coaching",), help="garder coaching.json tel quel malgré des chiffres changés")
    p.add_argument("--no-window", action="store_true", help="ne pas ouvrir la fenêtre de calibrage")
    p = sub.add_parser("set", help="changer un réglage, ex. start=12")
    p.add_argument("id")
    p.add_argument("pairs", nargs="+", metavar="clé=valeur")
    p = sub.add_parser("status", help="état des étapes d'une analyse")
    p.add_argument("id")
    sub.add_parser("list", help="toutes les analyses")
    args = ap.parse_args()

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
            print(f"aucune analyse ; pour commencer : {ME} new <vidéo> <nom>")
        for folder in folders:
            a = Analysis(folder)
            s, why = next_step(a)
            a.save()
            print(f"{folder.name:32s} {'terminée' if s is None else 'prochaine étape : ' + s.title + ' (' + why + ')'}")


if __name__ == "__main__":
    main()
