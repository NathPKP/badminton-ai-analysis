"""Stage 10 (written version): a PDF report to send by e-mail, in the first person (the player's report).

Usage: python s10_pdf.py <analysis dir> <out.pdf> [--lang en|fr] [--date "9 November 2026"]

Reads report.json, score.json, coaching.json and the tracks; draws the illustrations with the same
overlays as the video (s11_video.draw_video). The fixed texts are in bball/i18n.py (--lang; default:
the analysis' language). Everything specific to one match (lesson texts, image times, how some points
ended) comes from the optional "pdf" section of coaching.json:
  "pdf": {"summary": ["<b>My serve</b>: ...", ...],
          "lessons": [{"title", "constat", "travail", "caption", "width_cm" (opt.),
                       "images": [{"t", "zoom": "o"|"r" (opt.), "mark_px": [x, y] (opt.), "mark_label"}]}],
          "endings": {"1": "his serve fault", ...}, "uncertain": [6],
          "camera": "phone on the floor behind my baseline", "limits": "text"}
("constat" = the finding, "travail" = what to work on.) Without it, the lessons of the video
(title / stat / fix / freeze) are used as they are.
"""
import argparse
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

import s11_video
from bball.fonts import font
from bball.i18n import Text, analysis_lang
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
T = Text("en")


def mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


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
        im = text(im, [(label or T("mark"), (mx, my - 42), 32, (0, 230, 255), FONT_B, "ms", 5)])
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
    ax.text(3.05, 6.85, T("heat_net"), ha="center", va="bottom", fontsize=8, color="#52514e")
    ax.text(3.05, -0.75, T("heat_base"), ha="center", va="center", fontsize=8, color="#52514e")
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


def main():
    global T
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work")
    ap.add_argument("out")
    ap.add_argument("--lang", choices=("en", "fr"), help="default: the analysis' language")
    ap.add_argument("--date", help="report date as printed (default: today)")
    args = ap.parse_args()
    w = Path(args.work)
    T = Text(args.lang or analysis_lang(w))
    s11_video.set_lang(T.lang)
    date = args.date or T.date()
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
    params = meta.get("params", {})
    accepted = params.get("accept_score") or []

    # ---- document
    story = [Paragraph(T("title"), S["title"]),
             Paragraph(T("subtitle", n=len(done), min=int(video_s // 60), sec=int(video_s % 60), date=date), S["sub"])]

    you, opp = last.split("-")
    durs = [r["rdur"] for r in done]
    srv_you, srv_opp = sm["on_your_serve"], sm["on_his_serve"]
    stop = T("stop_unfinished", n=unfinished) if unfinished else T("stop_unknown") if unknown else ""
    a = mv["all"]
    key = [[Paragraph(f"<b>{T('k_score')}</b>", S["cell"]), Paragraph(f"<b>{you} – {opp}</b>{stop}", S["cell"])],
           [Paragraph(f"<b>{T('k_rally')}</b>", S["cell"]),
            Paragraph(T("v_rally", med=T.num(sm["rally_s"]["median"]), lo=T.num(min(durs)), hi=T.num(max(durs))), S["cell"])],
           [Paragraph(f"<b>{T('k_speed')}</b>", S["cell"]),
            Paragraph(T("v_speed", avg=T.num(a["avg_speed"], 2), peak=T.num(a["peak_speed"], 1),
                        dist=T.num(a["distance_m"], 0), live=T.num(a["live_s"], 0)), S["cell"])],
           [Paragraph(f"<b>{T('k_serve')}</b>", S["cell"]),
            Paragraph(T("v_serve", won=T.plural(srv_you["won"], "point_won"), n1=srv_you["won"] + srv_you["lost"],
                        w2=srv_opp["won"], n2=srv_opp["won"] + srv_opp["lost"]), S["cell"])]]
    t = Table(key, colWidths=[4.2 * cm, 12.8 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.white), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    n_sum = len(pdf["summary"])
    story += [t, Spacer(1, 8), Paragraph(T("h_summary"), S["h1"]),
              Paragraph(T.pick(n_sum, "summary_intro", N=T.count(n_sum)), S["p"])]
    story += [Paragraph(f"{i}. {x}", S["p"]) for i, x in enumerate(pdf["summary"], 1)]
    few = T("few", n=len(done)) if len(done) < 40 else ""
    story.append(Paragraph(few + T("each_ref"), S["small"]))

    # lessons
    n_les = len(pdf["lessons"])
    story.append(Paragraph(T.pick(n_les, "h_lessons", n=T.count(n_les).lower()), S["h1"]))
    for i, ls in enumerate(pdf["lessons"], 1):
        path = lesson_image(cap, tr, scale, ls["images"], img_dir / f"lesson{i}.jpg")
        single = len(ls["images"]) == 1 and not ls["images"][0].get("zoom")
        story.append(KeepTogether([
            Paragraph(ls["title"], S["h2"]),
            Paragraph(f"<b>{T('finding')}</b> {ls.get('constat') or ls['finding']}", S["p"]),
            Paragraph(f"<b>{T('work')}</b> {ls.get('travail') or ls['work']}", S["p"]),
            pic(path, ls.get("width_cm", 11 if single else 17)), Paragraph(ls["caption"], S["cap"])]))

    # points table
    who = T("who")
    data = [[Paragraph(f"<b>{h}</b>", S["cell"]) for h in T("table_head")]]
    endings = {int(k): v for k, v in pdf.get("endings", {}).items()}
    for r in rows:
        ending = (endings.get(r["n"]) or T("end_first").get(r.get("ending_code"))
                  or r.get("ending") or r.get("note") or "?")
        if r["n"] in pdf.get("uncertain", []) and r.get("ending_code"):
            ending += T("uncertain")
        data.append([Paragraph(str(r["n"]), S["cell"]), Paragraph(mmss(r["t"]), S["cell"]),
                     Paragraph(who[r["server"]], S["cell"]),
                     Paragraph(f"{r['before']} → {r['after'] or '?'}", S["cell"]),
                     Paragraph(f"{T.num(r['rdur'])} s" if r.get("rdur") else "—", S["cell"]),
                     Paragraph(f"<b>{who[r['winner']]}</b>" if r["winner"] else "—", S["cell"]),
                     Paragraph(ending, S["cell"])])
    tb = Table(data, colWidths=[0.8 * cm, 1.3 * cm, 1.5 * cm, 1.9 * cm, 1.6 * cm, 1.8 * cm, 8.1 * cm], repeatRows=1)
    style = [("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for i, r in enumerate(rows, 1):
        if r["winner"]:
            style.append(("TEXTCOLOR", (5, i), (5, i), ME if r["winner"] == "R" else OPP))
    tb.setStyle(TableStyle(style))
    story += [Paragraph(T("h_table"), S["h1"]), tb, Paragraph(T("table_note"), S["small"])]

    # movement
    heat, q = heatmap(tr, pts, img_dir / "heat.png")
    won, lost = mv.get("you_won") or {}, mv.get("you_lost") or {}
    speed = T("speed", v=T.num(a["avg_speed"], 2))
    if 0.8 <= a["avg_speed"] <= 1.2:
        speed += T("speed_usual")
    if won.get("avg_speed") is None or lost.get("avg_speed") is None:
        speed += "."
    elif abs(won["avg_speed"] - lost["avg_speed"]) < 0.05:
        speed += T("speed_same")
    else:
        speed += T("speed_diff", w=T.num(won["avg_speed"], 2), l=T.num(lost["avg_speed"], 2))
    zones = T("zones", b=T.num(100 * a["back_share"], 0))
    zones += (T("front_rare", f=T.num(100 * a["front_share"], 0)) if a["front_share"] < 0.05 else
              T("front", f=T.num(100 * a["front_share"], 0)))
    if len(q):
        from s7_movement import BASE
        dx, dy = np.median(q[:, 0]) - BASE[0], np.median(q[:, 1]) - BASE[1]
        across = T("across")["center" if abs(dx) < 0.5 else "left" if dx < 0 else "right"]
        depth = T("depth")["behind" if dy < -0.2 else "ahead" if dy > 0.2 else "on"]
        zones += T("zone_main", across=across, depth=depth)
    move = [Paragraph(speed, S["p"]), Paragraph(zones, S["p"])]
    if won.get("median_dist_1s_after_trip") is not None and lost.get("median_dist_1s_after_trip") is not None:
        trips = (T("trips_same", n=T.plural(won["trips"], "trip")) if won["trips"] == lost["trips"] else
                 T("trips_diff", l=T.plural(lost["trips"], "trip"), w=won["trips"]))
        sure = T("trips_sure") if min(won["trips"], lost["trips"]) < 10 else ""
        move.append(Paragraph(T("recovery", l=T.num(lost["median_dist_1s_after_trip"], 1),
                                w=T.num(won["median_dist_1s_after_trip"], 1), trips=trips, sure=sure), S["p"]))
    story.append(KeepTogether([
        Paragraph(T("h_move"), S["h1"]),
        Table([[pic(heat, 6.3), move]], colWidths=[6.8 * cm, 10.2 * cm], style=[("VALIGN", (0, 0), (-1, -1), "TOP")]),
        Paragraph(T("cap_heat"), S["cap"])]))

    # method
    n_srv, contra = len(score["points"]), score["contradictions"]
    ok, listed = n_srv - len(contra), ", ".join(map(str, contra))
    check = (T("check_ok", n=n_srv) if not contra else
             T.pick(len(contra), "check_accepted" if set(contra) <= set(accepted) else "check_bad", ok=ok, n=n_srv, list=listed))
    limits = pdf.get("limits") or (T("limit_sep").join(rep["limits"]) + ".")
    camera = pdf.get("camera") or T("camera_haute" if params.get("camera") == "haute" else "camera_sol")
    story += [Paragraph(T("h_method"), S["h1"]),
              Paragraph(T("method", camera=camera, check=check), S["p"]),
              Paragraph(f"<b>{T('limits')}</b> {limits[0].upper() + limits[1:]}", S["p"])]

    def footer(c, doc):
        c.saveState()
        c.setFont("Arial", 8)
        c.setFillColor(INK2)
        c.drawString(2 * cm, 1.2 * cm, T("title"))
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, T("page", n=doc.page))
        c.restoreState()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(args.out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=2 * cm, title=T("title"), author="")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
