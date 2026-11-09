"""Stage 10 (automatic coaching): coaching.json written by rules from the numbers, without an AI assistant.

Usage: python s10_coach.py <analysis dir> [--lang en|fr] [--max 3]

Each rule looks for one clear gap between points won and points lost in report.json / motion.json and
scores how strong it is; the strongest (at most --max) become the lessons, each with one freeze frame
taken from the match and the numbers behind it in "evidence". The texts are generic on purpose: an AI
assistant reading report.md and the frames writes better ones (replace coaching.json; run.py then
keeps it). The output has the same format: intro / lessons / outro for the reel, "pdf" for the report.

Rules (a lesson needs at least the stated sample):
  serve     your win rate on your serve is 20+ points below the one on his serve (3+ serves each)
  racket    racket up at the back of the court 15+ points more often on points won (10+ frames each)
  side      half or more of the lost points (2+) end on the same side of you, more than on won points
  recovery  1 s after a trip out of position you are 0.3+ m further from your base on lost points
            (3+ trips on each side)
  long      you lose two thirds or more of the rallies longer than 5 s (3+ such rallies)
If none applies, one lesson sums up the movement numbers.
"""
import argparse
from pathlib import Path

from bball.i18n import Text, analysis_lang
from bball.io import load_json, save_json

L = {
    "en": {
        "intro": {"title": "AI analysis of my match", "subtitle": "{n} rallies · positions 10 times per second · posture · score",
                  "voice": "Here are your {n} rallies, analysed by AI, and what cost you points."},
        "outro": {"title": "To work on", "note_few": "Only {n} rallies: trends, not certainties",
                  "voice": "{a} to {b}. To work on: {items}."},
        "and": " and ",
        "side": {"right": "right", "left": "left"},
        "pct_voice": "{p} percent",
        "serve": {"title": "{i}. Your serve", "pdf_title": "{i}. My serve", "item": "Vary your serve",
                  "summary": "<b>My serve</b>: I win {w} of {n} points when I serve.",
                  "stat": "On your serve: {w} of {n} points won ({w2} of {n2} on his serve)",
                  "fix": "Vary it: short and low, or long to the back line",
                  "voice": "{N}: when you serve, you win {w} of {n} points. When he serves, {w2} of {n2}. "
                           "Vary your serve: short and low, or long to the back.",
                  "constat": "When I serve, I win {w} of {n} points; when my opponent serves, I win {w2} of {n2}.",
                  "travail": "Vary the serve: short and low over the net, or long to the back of the court, so that it "
                             "cannot be attacked.",
                  "caption": "Point {k}: lost on your serve", "pdf_caption": "Point {k}, {mmss}: a point lost on my serve."},
        "racket": {"title": "{i}. Racket up at the back", "pdf_title": "{i}. Racket up at the back of the court",
                   "item": "Prepare the racket early at the back",
                   "summary": "<b>The racket at the back of the court</b>: it is up more often on the points I win.",
                   "stat": "At the back, racket up: {pw} of the time on points you win, {pl} on points you lose",
                   "fix": "When you move back: prepare the racket early",
                   "voice": "{N}: at the back of the court, your racket is up {pwv} of the time when you win the point, "
                            "{plv} when you lose it. Prepare it early, like here.",
                   "constat": "At the back of the court, my racket is up (hand above the shoulders) {pw} of the time on "
                              "points I win, against {pl} on points I lose ({fw} and {fl} frames measured).",
                   "travail": "When I move back, prepare the racket early, before the shuttle arrives.",
                   "caption": "Point {k}: racket ready early", "pdf_caption": "Point {k}, {mmss}: racket up before the hit."},
        "side": {"title": "{i}. Your {side} side", "pdf_title": "{i}. My {side} side", "item": "Move earlier to your {side}",
                 "summary": "<b>My {side} side</b>: {k} of my {nl} lost points end on my {side}.",
                 "stat": "{k} of your {nl} lost points end on your {side}",
                 "fix": "Towards your {side}: move earlier, finish balanced",
                 "voice": "{N}: {k} of your {nl} lost points end on your {side}. Move earlier, to hit in balance.",
                 "constat": "{k} of my {nl} lost points end on my {side} side (points {list}), against {kw} of my {nw} "
                            "points won.",
                 "travail": "As soon as the shuttle goes to my {side}, start earlier so as to hit in balance.",
                 "caption": "Point {k2}: lost on your {side}", "pdf_caption": "Point {k2}, {mmss}: end of a lost point, on my {side}."},
        "recovery": {"title": "{i}. Back to the base", "pdf_title": "{i}. Back to the base", "item": "Come back to the centre",
                     "summary": "<b>Recovery</b>: after being pulled out, I come back to my base more slowly on the points I lose.",
                     "stat": "1 s after being pulled out: {l} m from your base on lost points, {w} m on won points",
                     "fix": "After each shot: come back to the centre",
                     "voice": "{N}: one second after being pulled out of position, you are still {l} metres from your base "
                              "on points you lose, {w} on points you win. Come back to the centre after each shot.",
                     "constat": "One second after being pulled away from my base, I am still {l} m from it on points I "
                                "lose, against {w} m on points I win.",
                     "travail": "After each shot, push back to the centre of the court before the next one.",
                     "caption": "Point {k}: one second after a trip", "pdf_caption": "Point {k}, {mmss}: one second after being pulled out of position."},
        "long": {"title": "{i}. Long rallies", "pdf_title": "{i}. Long rallies", "item": "Stay patient in long rallies",
                 "summary": "<b>Long rallies</b>: I lose most of the rallies longer than 5 s.",
                 "stat": "Rallies over 5 s: {w} won, {l} lost",
                 "fix": "In long rallies: stay patient, play safe",
                 "voice": "{N}: when the rally lasts more than five seconds, you win {w} and lose {l}. Stay patient, "
                          "play safe, and wait for the right shuttle to attack.",
                 "constat": "In rallies longer than 5 s, I win {wp} and lose {l}.",
                 "travail": "In long rallies, stay patient: keep the shuttle deep and wait for a short one before attacking.",
                 "caption": "Point {k}: your longest lost rally", "pdf_caption": "Point {k}, {mmss}: end of my longest lost rally ({d} s)."},
        "move": {"title": "{i}. Your movement", "pdf_title": "{i}. My movement", "item": "Film a longer match",
                 "summary": "<b>My movement</b>: no clear gap between points won and lost on this clip.",
                 "stat": "Average speed {v} m/s, peak {p} m/s, {d} m covered",
                 "fix": "Film a longer match to find what costs you points",
                 "voice": "{N}: you move at {v} metres per second on average. No clear weakness on this clip: film a "
                          "longer match to find what costs you points.",
                 "constat": "I move at {v} m/s on average, with a peak at {p} m/s; no clear gap between the points I "
                            "win and the points I lose on this clip.",
                 "travail": "Film a longer match, so that the trends become clear.",
                 "caption": "Point {k}", "pdf_caption": "Point {k}, {mmss}."},
    },
    "fr": {
        "intro": {"title": "Analyse IA de mon match", "subtitle": "{n} échanges · positions 10 fois par seconde · posture · score",
                  "voice": "Voici tes {n} échanges, analysés par l'IA. Et ce qui t'a coûté des points."},
        "outro": {"title": "À travailler", "note_few": "{n} échanges seulement : des tendances, pas des certitudes",
                  "voice": "{a} à {b}. À travailler : {items}."},
        "and": " et ",
        "side": {"right": "droite", "left": "gauche"},
        "pct_voice": "{p} pour cent",
        "serve": {"title": "{i}. Ton service", "pdf_title": "{i}. Mon service", "item": "Varier ton service",
                  "summary": "<b>Mon service</b> : je gagne {wp} sur {n} quand je sers.",
                  "stat": "Quand tu sers : {wpg} sur {n} ({w2} sur {n2} quand il sert)",
                  "fix": "Varie : service court et bas, ou long jusqu'au fond",
                  "voice": "{N} : quand tu sers, tu gagnes {wp} sur {n}. Quand il sert, {w2} sur {n2}. "
                           "Varie ton service : court et bas, ou long jusqu'au fond.",
                  "constat": "Quand je sers, je gagne {wp} sur {n} ; quand l'adversaire sert, j'en gagne {w2} sur {n2}.",
                  "travail": "Varier le service : court et bas, au ras du filet, ou long jusqu'au fond du court, pour ne "
                             "pas offrir un volant à attaquer.",
                  "caption": "Point {k} : perdu sur ton service", "pdf_caption": "Point {k}, {mmss} : un point perdu sur mon service."},
        "racket": {"title": "{i}. Raquette armée au fond", "pdf_title": "{i}. Armer la raquette au fond du court",
                   "item": "Armer la raquette tôt au fond",
                   "summary": "<b>La raquette au fond du court</b> : elle est armée plus souvent sur les points que je gagne.",
                   "stat": "Au fond du court, raquette armée : {pw} du temps quand tu gagnes le point, {pl} quand tu le perds",
                   "fix": "Quand tu recules : arme la raquette tôt",
                   "voice": "{N} : au fond du court, ta raquette est armée {pwv} du temps quand tu gagnes le point, "
                            "{plv} quand tu le perds. Arme-la tôt, comme ici.",
                   "constat": "Quand je suis au fond du court, ma raquette est armée (main au-dessus des épaules) {pw} "
                              "du temps sur les points que je gagne, contre {pl} sur les points que je perds ({fw} et {fl} "
                              "images mesurées).",
                   "travail": "Quand je recule, armer la raquette tôt, avant que le volant arrive.",
                   "caption": "Point {k} : raquette armée tôt", "pdf_caption": "Point {k}, {mmss} : raquette armée avant la frappe."},
        "side": {"title": "{i}. Ton côté {side}", "pdf_title": "{i}. Mon côté {side}", "item": "Partir plus tôt vers ta {side}",
                 "summary": "<b>Mon côté {side}</b> : {k} de mes {nl} points perdus finissent sur ma {side}.",
                 "stat": "{k} de tes {nl} points perdus finissent sur ta {side}",
                 "fix": "Vers la {side} : pars plus tôt, finis équilibré",
                 "voice": "{N} : {k} de tes {nl} points perdus finissent sur ta {side}. Démarre plus tôt pour frapper équilibré.",
                 "constat": "{k} de mes {nl} points perdus finissent sur ma {side} (points {list}), contre {kw} de mes "
                            "{nw} points gagnés.",
                 "travail": "Dès que le volant part vers ma {side}, démarrer plus tôt pour frapper équilibré.",
                 "caption": "Point {k2} : perdu sur ta {side}", "pdf_caption": "Point {k2}, {mmss} : fin d'un point perdu, sur ma {side}."},
        "recovery": {"title": "{i}. Retour à la base", "pdf_title": "{i}. Le retour à la base", "item": "Revenir au centre",
                     "summary": "<b>Le replacement</b> : après avoir été déplacé, je reviens moins vite à ma base sur les points perdus.",
                     "stat": "1 s après un déplacement : {l} m de ta base sur les points perdus, {w} m sur les points gagnés",
                     "fix": "Après chaque frappe : reviens au centre",
                     "voice": "{N} : une seconde après avoir été déplacé, tu es encore à {l} mètres de ta base sur les "
                              "points perdus, {w} sur les points gagnés. Reviens au centre après chaque frappe.",
                     "constat": "Une seconde après avoir été déplacé loin de ma base, je suis encore à {l} m de celle-ci "
                                "sur les points perdus, contre {w} m sur les points gagnés.",
                     "travail": "Après chaque frappe, revenir au centre du terrain avant le coup suivant.",
                     "caption": "Point {k} : une seconde après un déplacement", "pdf_caption": "Point {k}, {mmss} : une seconde après un déplacement."},
        "long": {"title": "{i}. Les échanges longs", "pdf_title": "{i}. Les échanges longs", "item": "Rester patient dans les longs échanges",
                 "summary": "<b>Les échanges longs</b> : je perds la plupart des échanges de plus de 5 s.",
                 "stat": "Échanges de plus de 5 s : {w} gagné(s), {l} perdu(s)",
                 "fix": "Dans les longs échanges : patience, jeu sûr",
                 "voice": "{N} : quand l'échange dure plus de cinq secondes, tu en gagnes {w} et tu en perds {l}. "
                          "Reste patient, joue sûr, et attends le bon volant pour attaquer.",
                 "constat": "Sur les échanges de plus de 5 s, je gagne {wp} et j'en perds {l}.",
                 "travail": "Dans les échanges longs, rester patient : garder le volant au fond et attendre un volant "
                            "court avant d'attaquer.",
                 "caption": "Point {k} : ton plus long échange perdu", "pdf_caption": "Point {k}, {mmss} : fin de mon plus long échange perdu ({d} s)."},
        "move": {"title": "{i}. Tes déplacements", "pdf_title": "{i}. Mes déplacements", "item": "Filmer un match plus long",
                 "summary": "<b>Mes déplacements</b> : pas d'écart net entre points gagnés et perdus sur cette vidéo.",
                 "stat": "Vitesse moyenne {v} m/s, pointe à {p} m/s, {d} m parcourus",
                 "fix": "Filme un match plus long pour voir ce qui coûte des points",
                 "voice": "{N} : tu te déplaces à {v} mètres par seconde en moyenne. Pas de faiblesse nette sur cette "
                          "vidéo : filme un match plus long pour voir ce qui te coûte des points.",
                 "constat": "Je me déplace à {v} m/s en moyenne, avec une pointe à {p} m/s ; pas d'écart net entre les "
                            "points gagnés et perdus sur cette vidéo.",
                 "travail": "Filmer un match plus long, pour que les tendances ressortent.",
                 "caption": "Point {k}", "pdf_caption": "Point {k}, {mmss}."},
    },
}
SIDE_M = 0.6          # metres from the centre line to call an end position "right" / "left"


def mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def rules(rep, motion, pts):
    """Candidate lessons: (strength, id, values, freeze time, point, evidence)."""
    sm, rows = rep["summary"], rep["points"]
    done = [r for r in rows if r.get("rdur") and r["winner"]]
    by_n = {p["n"]: p for p in pts}
    out = []

    ys, hs = sm["on_your_serve"], sm["on_his_serve"]
    n1, n2 = ys["won"] + ys["lost"], hs["won"] + hs["lost"]
    if n1 >= 3 and n2 >= 3 and hs["won"] / n2 - ys["won"] / n1 >= 0.2:
        lost = [r for r in done if r["server"] == "R" and r["winner"] == "O"]
        if lost:
            k = lost[0]["n"]
            out.append((hs["won"] / n2 - ys["won"] / n1, "serve", dict(w=ys["won"], n=n1, w2=hs["won"], n2=n2),
                        by_n[k]["r0"] + 1.0, k, "o",
                        [f"report.json on_your_serve {ys}, on_his_serve {hs}", f"freeze: point {k}, lost on your serve"]))

    rear = sm.get("pose_you", {}).get("REAR", {})
    w_, l_ = rear.get("won") or {}, rear.get("lost") or {}
    if (w_.get("racket_up") is not None and l_.get("racket_up") is not None and w_.get("n_racket", 0) >= 10
            and l_.get("n_racket", 0) >= 10 and w_["racket_up"] - l_["racket_up"] >= 0.15):
        ex = sm.get("examples", {}).get("rear_racket_up_won") or []
        if ex:
            k, t = ex[len(ex) // 2]
            out.append((1.5 * (w_["racket_up"] - l_["racket_up"]), "racket",
                        dict(pw=w_["racket_up"], pl=l_["racket_up"], fw=w_["n_racket"], fl=l_["n_racket"]), t, k, None,
                        [f"pose_stats REAR won {w_}, lost {l_}", f"freeze: point {k} at {t} s, racket up on a won point"]))

    def side(r):
        x = (r.get("you_at_end") or {}).get("x")
        return None if x is None else "right" if x > 3.05 + SIDE_M else "left" if x < 3.05 - SIDE_M else None
    lost = [r for r in done if r["winner"] == "O"]
    won = [r for r in done if r["winner"] == "R"]
    for sd in ("right", "left"):
        ls, ws = [r for r in lost if side(r) == sd], [r for r in won if side(r) == sd]
        if lost and len(ls) >= 2 and len(ls) / len(lost) >= 0.5 and len(ls) / len(lost) > len(ws) / max(len(won), 1):
            k = ls[0]["n"]
            out.append((len(ls) / len(lost) - len(ws) / max(len(won), 1), "side",
                        dict(sd=sd, k=len(ls), nl=len(lost), kw=len(ws), nw=len(won), list=", ".join(str(r["n"]) for r in ls),
                             k2=k), by_n[k]["r1"] - 0.3, k, None,
                        [f"you_at_end x on lost points {[r['you_at_end']['x'] for r in ls]} (> {SIDE_M} m from the centre)",
                         f"freeze: end of point {k}"]))

    mv = sm.get("movement_you", {})
    mw, ml = mv.get("you_won") or {}, mv.get("you_lost") or {}
    dw, dl = mw.get("median_dist_1s_after_trip"), ml.get("median_dist_1s_after_trip")
    if dw is not None and dl is not None and mw.get("trips", 0) >= 3 and ml.get("trips", 0) >= 3 and dl - dw >= 0.3:
        trip = next(((p["n"], tr["t"]) for p in motion["points"] if p["winner"] == "O"
                     for tr in p["R"].get("trips", []) if tr.get("after_1s_m")), None)
        if trip:
            k, t = trip
            out.append((dl - dw, "recovery", dict(w=dw, l=dl), t + 1.0, k, None,
                        [f"motion summary you_won {mw}", f"you_lost {ml}", f"freeze: point {k}, 1 s after the trip at {t} s"]))

    longr = [r for r in done if r["rdur"] > 5]
    ll = [r for r in longr if r["winner"] == "O"]
    if len(longr) >= 3 and len(ll) / len(longr) >= 2 / 3:
        r = max(ll, key=lambda r: r["rdur"])
        out.append((len(ll) / len(longr) - 0.5, "long", dict(w=len(longr) - len(ll), l=len(ll), d=r["rdur"]),
                    by_n[r["n"]]["r1"] - 0.3, r["n"], None,
                    [f"rallies > 5 s: {[(x['n'], x['rdur'], x['winner']) for x in longr]}"]))
    return sorted(out, key=lambda c: -c[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work")
    ap.add_argument("--lang", choices=("en", "fr"), help="default: the analysis' language")
    ap.add_argument("--max", type=int, default=3, help="most lessons")
    ap.add_argument("--out", help="default: <analysis dir>/coaching.json")
    args = ap.parse_args()
    w = Path(args.work)
    T = Text(args.lang or analysis_lang(w))
    S = L[T.lang]
    rep, motion, pts = load_json(w / "report.json"), load_json(w / "motion.json"), load_json(w / "points.json")["points"]
    done = [r for r in rep["points"] if r.get("rdur") and r["winner"]]

    def pct(x):
        return f"{round(100 * x)}%" if T.lang == "en" else f"{round(100 * x)} %"

    picked = rules(rep, motion, pts)[:args.max]
    if not picked:                                       # nothing clear: one lesson on the movement numbers
        a = rep["summary"]["movement_you"]["all"]
        r = next((r for r in done if r["winner"] == "R"), done[0])
        picked = [(0, "move", dict(v=T.num(a["avg_speed"], 2), p=T.num(a["peak_speed"], 1), d=T.num(a["distance_m"], 0)),
                   r["r1"] - 0.3, r["n"], None, ["no rule reached its threshold"])]

    picked.sort(key=lambda c: c[3])                     # numbered in the order they appear in the reel
    lessons, pdf_lessons, items, summary = [], [], [], []
    for i, (strength, rid, v, t, k, zoom, evidence) in enumerate(picked, 1):
        tpl = S[rid]
        v = dict(v)
        if rid == "racket":
            v.update(pw=pct(v["pw"]), pl=pct(v["pl"]), pwv=S["pct_voice"].format(p=round(100 * v["pw"])),
                     plv=S["pct_voice"].format(p=round(100 * v["pl"])))
        if rid == "recovery":
            v.update(w=T.num(v["w"], 1), l=T.num(v["l"], 1))
        if rid == "long":
            v.update(d=T.num(v["d"]))
        if rid == "side":
            v["side"] = S["side"][v.pop("sd")]
        if "w" in v and isinstance(v["w"], int):
            v.update(wp=T.plural(v["w"], "point"), wpg=T.plural(v["w"], "point_won"))
        v.update(i=i, N=T.count(i), k=v.get("k", k), mmss=mmss(t))
        freeze = {"t": round(t, 1), "caption": tpl["caption"].format(**v)}
        if zoom:
            freeze["zoom"] = zoom
        lessons.append({"id": rid, "after_point": k, "title": tpl["title"].format(**v), "stat": tpl["stat"].format(**v),
                        "fix": tpl["fix"].format(**v), "voice": tpl["voice"].format(**v), "freeze": [freeze],
                        "strength": round(strength, 2), "evidence": evidence})
        image = {"t": round(t, 1)}
        if zoom:
            image["zoom"] = zoom
        pdf_lessons.append({"title": tpl["pdf_title"].format(**v), "constat": tpl["constat"].format(**v),
                            "travail": tpl["travail"].format(**v), "images": [image],
                            "caption": tpl["pdf_caption"].format(**v)})
        items.append(tpl["item"].format(**v))
        summary.append(tpl["summary"].format(**v))

    last = done[-1]["after"].split("-")
    outro_items = S["and"].join([", ".join(x.lower() for x in items[:-1]), items[-1].lower()]) if len(items) > 1 else items[0].lower()
    coach = {"about": "Generated by s10_coach.py from report.json (rules, see its docstring). Replace it with an "
                      "AI-written coaching.json for more specific lessons.",
             "generated": "auto",
             "intro": {k: s.format(n=len(done)) for k, s in S["intro"].items()},
             "lessons": lessons,
             "outro": {"title": S["outro"]["title"], "items": items,
                       "note": S["outro"]["note_few"].format(n=len(done)) if len(done) < 40 else "",
                       "voice": S["outro"]["voice"].format(a=last[0], b=last[1], items=outro_items)},
             "pdf": {"summary": summary, "lessons": pdf_lessons}}
    out = Path(args.out or w / "coaching.json")
    save_json(coach, out)
    print(f"{len(lessons)} lesson(s) -> {out}")
    for l_ in lessons:
        print(f"  [{l_['strength']:.2f}] {l_['title']}: {l_['stat']}  (freeze {l_['freeze'][0]['t']} s, after point {l_['after_point']})")


if __name__ == "__main__":
    main()
