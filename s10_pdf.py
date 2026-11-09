"""Stage 10 (written version): a PDF report to send by e-mail, in French, first person (the player's report).

Usage: python s10_pdf.py <analysis dir> <out.pdf> [--date "27 octobre 2026"]

Reads report.json, score.json, coaching.json and the tracks; draws the illustrations with the same
overlays as the video (s11_video.draw_video). Everything specific to one match (lesson texts, image
times, how some points ended) comes from the optional "pdf" section of coaching.json:
  "pdf": {"summary": ["<b>Mon service</b> : ...", ...],
          "lessons": [{"title", "constat", "travail", "caption", "width_cm" (opt.),
                       "images": [{"t", "zoom": "o"|"r" (opt.), "mark_px": [x, y] (opt.), "mark_label"}]}],
          "endings": {"1": "sa faute de service", ...}, "uncertain": [6],
          "camera": "téléphone posé au sol derrière ma ligne de fond", "limits": "texte"}
Without it, the lessons of the video (title / stat / fix / freeze) are used as they are.
"""
import argparse
import datetime
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from bball.fonts import font
from bball.io import load_json
from bball.video import player_zoom
from s11_video import Tracks, draw_video, text, FONT_B

pdfmetrics.registerFont(TTFont("Arial", font("regular")))
pdfmetrics.registerFont(TTFont("Arial-Bold", font("bold")))
pdfmetrics.registerFont(TTFont("Arial-Italic", font("italic")))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")

INK, INK2, RULE, PANEL = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#d9d8d4"), colors.HexColor("#f3f2ef")
ME, OPP = colors.HexColor("#2a78d6"), colors.HexColor("#eb6834")
S = {
    "title": ParagraphStyle("t", fontName="Arial-Bold", fontSize=20, leading=24, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("s", fontName="Arial", fontSize=10, leading=14, textColor=INK2, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=13.5, leading=17, textColor=INK, spaceBefore=12, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=11.5, leading=15, textColor=ME, spaceBefore=8, spaceAfter=3),
    "p": ParagraphStyle("p", fontName="Arial", fontSize=10, leading=14.2, textColor=INK, spaceAfter=5),
    "small": ParagraphStyle("sm", fontName="Arial", fontSize=8.5, leading=11.5, textColor=INK2, spaceAfter=4),
    "cap": ParagraphStyle("c", fontName="Arial-Italic", fontSize=8.5, leading=11, textColor=INK2, alignment=TA_CENTER, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Arial", fontSize=8.5, leading=10.5, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Arial-Bold", fontSize=8.5, leading=10.5, textColor=INK),
}
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
          "novembre", "décembre"]
NUMBERS = ["Aucun", "Un", "Deux", "Trois", "Quatre", "Cinq", "Six"]

# report.json speaks to the player ("ta faute"); this report is in the first person
FIRST_PERSON = {"sa faute (volant dehors ou service)": "sa faute (volant dehors)",
                "son coup gagnant (ou ta faute au filet)": "son coup gagnant (ou ma faute au filet)",
                "ton coup gagnant (ou sa faute au filet)": "mon coup gagnant (ou sa faute au filet)",
                "ta frappe sortie": "ma frappe sortie"}


def fr(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def plural(n, words):
    return f"{n} {words}" if n <= 1 else f"{n} " + " ".join(w + "s" for w in words.split())


def save_img(img, path):
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def pic(path, width_cm):
    im = cv2.imread(str(path))
    h, w = im.shape[:2]
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def frame_with_overlays(cap, tr, t, scale, mark=None, label=None):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * 30)))
    raw = cap.read()[1]
    im = draw_video(raw, tr, t, scale, trail_s=0.0)
    if mark:
        mx, my = (np.array(mark) * scale).astype(int)
        cv2.circle(im, (mx, my), 30, (0, 230, 255), 4, cv2.LINE_AA)
        im = text(im, [(label or "volant", (mx, my - 42), 32, (0, 230, 255), FONT_B, "ms", 5)])
    return im, raw


def lesson_image(cap, tr, scale, panels, path):
    """One panel without zoom = the full frame; otherwise the panels (and zooms) side by side at 960x541."""
    if len(panels) == 1 and not panels[0].get("zoom"):
        p = panels[0]
        return save_img(frame_with_overlays(cap, tr, p["t"], scale, p.get("mark_px"), p.get("mark_label"))[0], path)
    ims = []
    for p in panels:
        a, raw = frame_with_overlays(cap, tr, p["t"], scale, p.get("mark_px"), p.get("mark_label"))
        ims.append(cv2.resize(a, (960, 541)))
        box = tr.box_near(p["zoom"], p["t"]) if p.get("zoom") else None
        if box:
            ims.append(player_zoom(raw, box, (962, 541)))
        elif p.get("zoom"):
            print(f"WARNING: no {p['zoom']} box near {p['t']} s: zoom left out")
    gap = np.full((541, 20, 3), 255, np.uint8)
    return save_img(np.hstack([x for im in ims for x in (gap, im)][1:]), path)


def heatmap(tr, pts, path):
    """Heatmap of the near player's live positions; returns (path, positions)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from s7_movement import BASE
    live = np.zeros(len(tr.t), bool)
    for p in pts:
        if p["r1"]:
            live |= (tr.t >= p["r0"] + 0.3) & (tr.t <= p["r1"])
    q = tr.P["r"][live]
    q = q[~np.isnan(q[:, 0])]
    h, xe, ye = np.histogram2d(q[:, 0], q[:, 1], bins=[np.arange(-0.5, 6.61, 0.5), np.arange(-0.5, 6.71, 0.5)])
    cmap = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    fig.patch.set_facecolor("white")
    ax.pcolormesh(xe, ye, np.ma.masked_equal(h.T / 10.0, 0), cmap=cmap, vmin=0)
    for (x1, y1), (x2, y2) in [((0, 0), (6.1, 0)), ((0, 0.76), (6.1, 0.76)), ((0, 4.72), (6.1, 4.72)), ((3.05, 0), (3.05, 4.72))] + \
            [((x, 0), (x, 6.7)) for x in (0, 0.46, 5.64, 6.1)]:
        ax.plot([x1, x2], [y1, y2], color="#52514e", lw=0.8)
    ax.plot([0, 6.1], [6.7, 6.7], color="#0b0b0b", lw=2.2)
    ax.plot(*BASE, "+", color="#0b0b0b", ms=11, mew=1.6)
    ax.text(3.05, 6.85, "filet", ha="center", va="bottom", fontsize=8, color="#52514e")
    ax.text(3.05, -0.75, "ma ligne de fond (côté caméra)", ha="center", va="center", fontsize=8, color="#52514e")
    ax.set_xlim(-0.6, 6.7)
    ax.set_ylim(-1.0, 7.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, q


def pdf_section(coach):
    """coaching.json "pdf" section, completed with the video's lessons for whatever it leaves out."""
    def sentence(s):
        return s if s.rstrip()[-1:] in ".!?…" else s + "."
    pdf = dict(coach.get("pdf", {}))
    if "pdf" not in coach:
        print("WARNING: coaching.json has no 'pdf' section: the PDF reuses the video's lessons (2nd person)")
    pdf.setdefault("summary", coach["outro"]["items"])
    pdf.setdefault("lessons", [{"title": l_["title"], "constat": sentence(l_["stat"]), "travail": sentence(l_["fix"]),
                                "images": [{k: f[k] for k in ("t", "zoom", "mark_px", "mark_label") if k in f}
                                           for f in l_["freeze"]],
                                "caption": " / ".join(f"{f['caption']} ({mmss(f['t'])})" for f in l_["freeze"])}
                               for l_ in coach["lessons"]])
    return pdf


def count_word(n):
    return NUMBERS[n] if n < len(NUMBERS) else str(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work")
    ap.add_argument("out")
    today = datetime.date.today()
    ap.add_argument("--date", default=f"{today.day} {MONTHS[today.month - 1]} {today.year}")
    args = ap.parse_args()
    w = Path(args.work)
    img_dir = w / "report_img"
    img_dir.mkdir(exist_ok=True)

    rep = load_json(w / "report.json")
    rows, sm = rep["points"], rep["summary"]
    mv = sm["movement_you"]
    pts = load_json(w / "points.json")["points"]
    score = load_json(w / "score.json")
    pdf = pdf_section(load_json(w / "coaching.json"))
    tr = Tracks(w)
    cap = cv2.VideoCapture(str(w / "proxy.mp4"))
    scale = 1080 / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_s = round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    done = [r for r in rows if r.get("rdur") and r["winner"]]
    last = done[-1]["after"]
    unfinished = rows[-1]["n"] if rows[-1].get("rdur") is None else None
    unknown = rows[-1]["winner"] is None and not unfinished
    meta = load_json(w / "analysis.json") if (w / "analysis.json").exists() else {}
    accepted = meta.get("params", {}).get("accept_score") or []

    # ---- document
    story = [Paragraph("Analyse vidéo de mon match de badminton", S["title"]),
             Paragraph(f"Simple · {len(done)} échanges filmés ({int(video_s // 60)} min {int(video_s % 60):02d} de vidéo) · "
                       f"rapport du {args.date} · analyse automatique par IA, faite en local", S["sub"])]

    you, opp = last.split("-")
    durs = [r["rdur"] for r in done]
    srv_you, srv_opp = sm["on_your_serve"], sm["on_his_serve"]
    stop = (f" (la vidéo s'arrête pendant le {unfinished}<super>e</super> échange)" if unfinished else
            " (résultat du dernier échange inconnu)" if unknown else "")
    key = [[Paragraph("<b>Score sur la vidéo</b>", S["cell"]), Paragraph(f"<b>{you} – {opp}</b>{stop}", S["cell"])],
           [Paragraph("<b>Durée des échanges</b>", S["cell"]), Paragraph(f"médiane {fr(sm['rally_s']['median'])} s (de {fr(min(durs))} à {fr(max(durs))} s)", S["cell"])],
           [Paragraph("<b>Ma vitesse en jeu</b>", S["cell"]), Paragraph(f"{fr(mv['all']['avg_speed'], 2)} m/s en moyenne, pointe à {fr(mv['all']['peak_speed'], 1)} m/s ; "
                                                                        f"{fr(mv['all']['distance_m'], 0)} m parcourus en {fr(mv['all']['live_s'], 0)} s de jeu effectif", S["cell"])],
           [Paragraph("<b>Au service</b>", S["cell"]), Paragraph(f"quand je sers : {plural(srv_you['won'], 'point gagné')} sur "
                                                                 f"{srv_you['won'] + srv_you['lost']} ; quand il sert : "
                                                                 f"{srv_opp['won']} sur {srv_opp['won'] + srv_opp['lost']}", S["cell"])]]
    t = Table(key, colWidths=[4.2 * cm, 12.8 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.white), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    n_sum = len(pdf["summary"])
    story += [t, Spacer(1, 8),
              Paragraph("En bref", S["h1"]),
              Paragraph(f"{count_word(n_sum)} point{'s' if n_sum > 1 else ''} à travailler "
                        f"ressort{'ent' if n_sum > 1 else ''} de l'analyse :", S["p"])]
    story += [Paragraph(f"{i}. {x}", S["p"]) for i, x in enumerate(pdf["summary"], 1)]
    few = f"Sur {len(done)} échanges seulement, ce sont des tendances, pas des certitudes. " if len(done) < 40 else ""
    story.append(Paragraph(few + "Chaque constat ci-dessous renvoie à un moment précis de la vidéo (minute:seconde) "
                                 "pour pouvoir le vérifier.", S["small"]))

    # lessons
    n_les = len(pdf["lessons"])
    story.append(Paragraph(f"Les {count_word(n_les).lower()} enseignements" if n_les > 1 else "L'enseignement", S["h1"]))
    for i, ls in enumerate(pdf["lessons"], 1):
        path = lesson_image(cap, tr, scale, ls["images"], img_dir / f"lesson{i}.jpg")
        single = len(ls["images"]) == 1 and not ls["images"][0].get("zoom")
        story.append(KeepTogether([
            Paragraph(ls["title"], S["h2"]),
            Paragraph(f"<b>Constat.</b> {ls['constat']}", S["p"]),
            Paragraph(f"<b>À travailler.</b> {ls['travail']}", S["p"]),
            pic(path, ls.get("width_cm", 11 if single else 17)), Paragraph(ls["caption"], S["cap"])]))

    # points table
    who = {"R": "moi", "O": "adv.", None: "?"}
    head = ["#", "Vidéo", "Service", "Score", "Durée", "Gagné par", "Comment le point se termine"]
    data = [[Paragraph(f"<b>{h}</b>", S["cell"]) for h in head]]
    endings = {int(k): v for k, v in pdf.get("endings", {}).items()}
    for r in rows:
        ending = endings.get(r["n"]) or FIRST_PERSON.get(r.get("ending"), r.get("ending", r.get("note", "?")))
        if r["n"] in pdf.get("uncertain", []) and r.get("ending"):
            ending += " (incertain)"
        data.append([Paragraph(str(r["n"]), S["cell"]), Paragraph(mmss(r["t"]), S["cell"]),
                     Paragraph(who[r["server"]], S["cell"]),
                     Paragraph(f"{r['before']} → {r['after'] or '?'}", S["cell"]),
                     Paragraph(f"{fr(r['rdur'])} s" if r.get("rdur") else "—", S["cell"]),
                     Paragraph(f"<b>{who[r['winner']]}</b>" if r["winner"] else "—", S["cell"]),
                     Paragraph(ending, S["cell"])])
    tb = Table(data, colWidths=[0.8 * cm, 1.3 * cm, 1.5 * cm, 1.9 * cm, 1.4 * cm, 1.8 * cm, 8.3 * cm], repeatRows=1)
    style = [("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for i, r in enumerate(rows, 1):
        if r["winner"]:
            style.append(("TEXTCOLOR", (5, i), (5, i), ME if r["winner"] == "R" else OPP))
    tb.setStyle(TableStyle(style))
    story += [Paragraph("Le match point par point", S["h1"]), tb,
              Paragraph("Score : moi – adversaire. « Comment le point se termine » combine le gagnant (celui qui sert ensuite) et le côté "
                        "où le volant est tombé : par exemple, un volant de mon côté sur un point perdu = son coup gagnant ou ma faute au filet.",
                        S["small"])]

    # movement
    heat, q = heatmap(tr, pts, img_dir / "heat.png")
    a, won, lost = mv["all"], mv.get("you_won") or {}, mv.get("you_lost") or {}
    speed = f"Vitesse moyenne en jeu : <b>{fr(a['avg_speed'], 2)} m/s</b>"
    if 0.8 <= a["avg_speed"] <= 1.2:
        speed += ", la valeur habituelle en simple (environ 1 m/s)"
    if won.get("avg_speed") is None or lost.get("avg_speed") is None:
        speed += "."
    elif abs(won["avg_speed"] - lost["avg_speed"]) < 0.05:
        speed += ", identique sur les points gagnés et perdus."
    else:
        speed += f" ({fr(won['avg_speed'], 2)} m/s sur les points gagnés, {fr(lost['avg_speed'], 2)} sur les points perdus)."
    zones = f"Je passe {fr(100 * a['back_share'], 0)} % du temps de jeu au fond du court et "
    zones += (f"presque jamais devant la ligne de service court ({fr(100 * a['front_share'], 0)} %)." if a["front_share"] < 0.05 else
              f"{fr(100 * a['front_share'], 0)} % devant la ligne de service court.")
    if len(q):
        from s7_movement import BASE
        dx, dy = np.median(q[:, 0]) - BASE[0], np.median(q[:, 1]) - BASE[1]
        across = "le centre" if abs(dx) < 0.5 else "le côté gauche" if dx < 0 else "le côté droit"
        depth = "un peu en arrière de ma base" if dy < -0.2 else "un peu en avant de ma base" if dy > 0.2 else "autour de ma base"
        zones += f" Ma zone principale est {across}, {depth} (+)."
    move = [Paragraph(speed, S["p"]), Paragraph(zones, S["p"])]
    if won.get("median_dist_1s_after_trip") is not None and lost.get("median_dist_1s_after_trip") is not None:
        trips = (f"{plural(won['trips'], 'déplacement')} de chaque côté" if won["trips"] == lost["trips"] else
                 f"{plural(lost['trips'], 'déplacement')} sur les points perdus, {won['trips']} sur les points gagnés")
        sure = " : à confirmer sur plus de points" if min(won["trips"], lost["trips"]) < 10 else ""
        move.append(Paragraph(f"Une seconde après avoir été déplacé loin de ma base, je suis encore à {fr(lost['median_dist_1s_after_trip'], 1)} m "
                              f"de celle-ci sur les points perdus, contre {fr(won['median_dist_1s_after_trip'], 1)} m sur les points gagnés "
                              f"({trips}{sure}).", S["p"]))
    story.append(KeepTogether([
        Paragraph("Déplacements", S["h1"]),
        Table([[pic(heat, 6.3), move]], colWidths=[6.8 * cm, 10.2 * cm], style=[("VALIGN", (0, 0), (-1, -1), "TOP")]),
        Paragraph("Temps passé à chaque endroit de ma moitié de terrain pendant le jeu (plus foncé = plus longtemps).", S["cap"])]))

    # method
    n_srv, bad = len(score["points"]), len(score["contradictions"])
    contra = score["contradictions"]
    if not bad:
        check = f"{n_srv} services sur {n_srv} cohérents"
    elif set(contra) <= set(accepted):
        check = (f"{n_srv - bad} services sur {n_srv} cohérents ; "
                 + (f"le service {contra[0]} est une exception vérifiée" if bad == 1 else
                    f"les services {', '.join(map(str, contra))} sont des exceptions vérifiées") + " : faute ou let")
    else:
        check = (f"{n_srv - bad} services sur {n_srv} cohérents ; "
                 + (f"le service {contra[0]} est à revérifier" if bad == 1 else
                    f"les services {', '.join(map(str, contra))} sont à revérifier"))
    limits = pdf.get("limits") or (" ; ".join(rep["limits"]) + ".")
    story += [Paragraph("Comment c'est mesuré, et les limites", S["h1"]),
              Paragraph(f"La vidéo ({pdf.get('camera', 'téléphone posé au sol derrière ma ligne de fond' if meta.get('params', {}).get('camera', 'sol') == 'sol' else 'caméra surélevée derrière ma ligne de fond')}) est analysée image par image : "
                        "détection des joueurs 10 fois par seconde, calibrage du terrain pour passer en mètres, posture avec un "
                        "modèle de squelette, services et fins d'échange relus image par image, score reconstitué avec la règle "
                        f"« le gagnant sert » et contrôlé par la règle du côté de service ({check}). Tout tourne sur un PC, "
                        "avec des outils gratuits.", S["p"]),
              Paragraph(f"<b>Limites.</b> {limits[0].upper() + limits[1:]}", S["p"])]

    def footer(c, doc):
        c.saveState()
        c.setFont("Arial", 8)
        c.setFillColor(INK2)
        c.drawString(2 * cm, 1.2 * cm, "Analyse vidéo de mon match de badminton")
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
        c.restoreState()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(args.out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=2 * cm, title="Analyse vidéo de mon match de badminton", author="")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
