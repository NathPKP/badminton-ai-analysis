"""Stage 11 (optional): vertical highlight reel with overlays, mini map and a coach voice-over.

Usage: python s11_video.py <work dir> <out.mp4> [--speed 1.2] [--voice ff_siwis] [--models models]

Layout (1080x1920): score bar / match video with foot rings, trails, skeleton and name tags /
top-down mini map with speeds / captions. Rallies are cut from serve to rally end (dead time
removed) and played at --speed; after the rallies named in coaching.json the video freezes on
the lesson frames while the coach (Kokoro TTS, local) explains. Reads from the work dir:
proxy.mp4, players.json, calib.json, points.json, pose_frames.json, coaching.json.
"""
import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bball import court
from bball.fonts import font
from bball.io import load_json
from bball.track import positions, rolling_median
from bball.video import player_zoom
from s7_movement import BASE, interp_gaps, moving_mean
from s8_pose import BONES

W, H, FPS = 1080, 1920, 30
BG, INK, INK2, LINE = (20, 20, 19), (236, 239, 240), (183, 194, 195), (90, 90, 88)   # BGR
YOU, OPP = (229, 135, 57), (38, 89, 217)        # dark-mode categorical slots 1-2 (#3987e5, #d95926), BGR
VID_Y, VID_H = 230, 609                          # match video band
MAP_Y = VID_Y + VID_H + 40
CAP_Y = 1700
FONT_R, FONT_B, FONT_D = font("regular"), font("bold"), font("display")
_fonts = {}


def font(path, size):
    if (path, size) not in _fonts:
        _fonts[(path, size)] = ImageFont.truetype(path, size)
    return _fonts[(path, size)]


def bgr2rgb(c):
    return (c[2], c[1], c[0])


def text(img, items):
    """Draw [(str, (x, y), size, colour_bgr, path, anchor[, outline])] with Pillow (accents) onto a BGR image."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for it in items:
        s, xy, size, col, path, anchor = it[:6]
        outline = it[6] if len(it) > 6 else 0
        d.text(xy, s, font=font(path, size), fill=bgr2rgb(col), anchor=anchor,
               stroke_width=outline, stroke_fill=(10, 10, 10))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def wrap(s, path, size, width):
    words, lines, cur = s.split(), [], ""
    for w_ in words:
        trial = (cur + " " + w_).strip()
        if font(path, size).getlength(trial) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w_
    return lines + ([cur] if cur else [])


class Tracks:
    """Smoothed positions (court m), speeds and pose, interpolated at any source time."""

    def __init__(self, w):
        data = load_json(w / "players.json")
        self.frames = data["frames"]
        self.t = np.array([f["t"] for f in self.frames])
        self.H_at = court.load_calib(w / "calib.json")
        self.segs = load_json(w / "calib.json")["segments"]
        self.P = {"r": moving_mean(rolling_median(interp_gaps(positions(self.frames, "r"), 10), 5), 7),
                  "o": moving_mean(rolling_median(interp_gaps(positions(self.frames, "o"), 10), 9), 7)}
        self.V = {}
        for k, P in self.P.items():
            v = np.full(len(P), np.nan)
            v[2:-2] = np.hypot(*(P[4:] - P[:-4]).T) / 0.4
            self.V[k] = v
        self.pose = {round(f["t"], 2): f for f in load_json(w / "pose_frames.json")["frames"]}

    def H(self, t):
        """Homography at time t; between or outside the calibrated ranges, the nearest one."""
        H = self.H_at(t)
        if H is None:
            s = min(self.segs, key=lambda s: min(abs(t - s["t0"]), abs(t - s["t1"])))
            H = np.array(s["H"], dtype=np.float64)
        return H

    def at(self, key, t):
        i = np.searchsorted(self.t, t)
        if i <= 0 or i >= len(self.t):
            return None, None
        a, b = i - 1, i
        u = (t - self.t[a]) / max(self.t[b] - self.t[a], 1e-6)
        pa, pb = self.P[key][a], self.P[key][b]
        if np.isnan(pa[0]) or np.isnan(pb[0]):
            return None, None
        va, vb = self.V[key][a], self.V[key][b]
        v = vb if np.isnan(va) else va if np.isnan(vb) else va * (1 - u) + vb * u
        return pa * (1 - u) + pb * u, None if np.isnan(v) else v

    def box_near(self, key, t, within=0.3):
        """Detection box of player key ("r" / "o") at the frame nearest t that has one (<= within s)."""
        near = [k for k in np.argsort(np.abs(self.t - t)) if abs(self.t[k] - t) <= within]
        return next((self.frames[k][key + "b"] for k in near if self.frames[k][key + "b"]), None)

    def skeleton(self, t, size):
        """Pose landmarks of the near player in frame pixels, from the nearest pose frame (<= 0.15 s)."""
        k = int(np.argmin(np.abs(self.t - t)))
        if abs(self.t[k] - t) > 0.15:
            return None
        f = self.pose.get(round(float(self.t[k]), 2))
        b = self.frames[k]["rb"]
        if f is None or b is None:
            return None
        w_, h_ = size
        x1, y1, x2, y2 = b[0] * w_, b[1] * h_, b[2] * w_, b[3] * h_
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.35
        lm = np.array(f["lm"])
        return np.stack([cx + (lm[:, 0] - 0.5) * side, cy + (lm[:, 1] - 0.5) * side, lm[:, 2]], 1)


def draw_video(frame, tr, src_t, scale, trail_s=1.5):
    """Foot rings, trails, skeleton and name tags on the scaled match frame."""
    h0, w0 = frame.shape[:2]
    img = cv2.resize(frame, (W, int(round(h0 * scale))), interpolation=cv2.INTER_CUBIC)
    sk = tr.skeleton(src_t, (w0, h0))
    if sk is not None:
        for a, b in BONES:
            if sk[a, 2] > 0.5 and sk[b, 2] > 0.5:
                cv2.line(img, tuple((sk[a, :2] * scale).astype(int)), tuple((sk[b, :2] * scale).astype(int)),
                         (255, 255, 255), 3, cv2.LINE_AA)
    tags = []
    for key, col, name in (("r", YOU, "TOI"), ("o", OPP, "ADV")):
        pts = []
        for dt in np.arange(-trail_s, 0.001, 0.1):
            p, _ = tr.at(key, src_t + dt)
            if p is not None:
                pts.append(court.to_px(tr.H(src_t + dt), [p])[0] * scale)
        for j in range(1, len(pts)):
            a = j / len(pts)
            cv2.line(img, tuple(pts[j - 1].astype(int)), tuple(pts[j].astype(int)),
                     tuple(int(c * a + 40 * (1 - a)) for c in col), max(2, int(6 * a)), cv2.LINE_AA)
        p, v = tr.at(key, src_t)
        if p is None:
            continue
        fx, fy = court.to_px(tr.H(src_t), [p])[0] * scale
        rx = 34 if key == "r" else 16
        if fy < img.shape[0] - 4:
            cv2.ellipse(img, (int(fx), int(fy)), (rx, rx // 3), 0, 0, 360, col, 3, cv2.LINE_AA)
        k = int(np.argmin(np.abs(tr.t - src_t)))
        b = tr.frames[k][key + "b"]
        top = int(b[1] * h0 * scale) - 12 if b else int(fy) - 120
        label = f"{name} {v:.1f} m/s" if v is not None and np.isfinite(v) else name
        tags.append((label, (int(fx), max(top, 30)), 32, col, FONT_B, "ms", 5))
    return text(img, tags)


def draw_map(canvas, tr, src_t, x0, y0, ppm=56, trail_s=2.0):
    """Top-down court, camera side at the bottom (you), with trails and the base point."""
    cw, ch = court.WIDTH * ppm, court.LENGTH * ppm

    def px(p):
        return int(x0 + p[0] * ppm), int(y0 + ch - p[1] * ppm)
    cv2.rectangle(canvas, (int(x0) - 14, int(y0) - 14), (int(x0 + cw) + 14, int(y0 + ch) + 14), (34, 34, 33), -1)
    for a, b in court.court_lines():
        cv2.line(canvas, px(a), px(b), LINE, 2, cv2.LINE_AA)
    cv2.line(canvas, px((0, court.NET_Y)), px((court.WIDTH, court.NET_Y)), INK, 4)
    bx, by = px(BASE)
    cv2.drawMarker(canvas, (bx, by), INK2, cv2.MARKER_CROSS, 18, 2)
    for key, col in (("r", YOU), ("o", OPP)):
        pts = [tr.at(key, src_t + dt)[0] for dt in np.arange(-trail_s, 0.001, 0.1)]
        pts = [px(p) for p in pts if p is not None]
        for j in range(1, len(pts)):
            cv2.line(canvas, pts[j - 1], pts[j], col, 4, cv2.LINE_AA)
        p, _ = tr.at(key, src_t)
        if p is not None:
            cv2.circle(canvas, px(p), 14, col, -1, cv2.LINE_AA)
            cv2.circle(canvas, px(p), 14, BG, 3, cv2.LINE_AA)


def score_bar(canvas, items, score, sub, banner=None):
    you, opp = score.split("-")
    items += [("TOI", (300, 70), 40, YOU, FONT_B, "mm"), ("ADV", (780, 70), 40, OPP, FONT_B, "mm"),
              (f"{you}  –  {opp}", (540, 78), 92, INK, FONT_D, "mm"), (sub, (540, 165), 32, INK2, FONT_R, "mm")]
    if banner:
        txt, col = banner
        cv2.rectangle(canvas, (0, VID_Y + VID_H - 90), (W, VID_Y + VID_H), col, -1)
        items.append((txt, (540, VID_Y + VID_H - 45), 52, (255, 255, 255), FONT_B, "mm"))


def captions(items, s, y=CAP_Y + 40, size=40):
    for i, line in enumerate(wrap(s, FONT_B, size, W - 120)[:3]):
        items.append((line, (540, y + i * (size + 14)), size, INK, FONT_B, "mm"))


def sentence_at(voice, t):
    """Caption chunk of a voice line at time t (s) into it, timed by character count."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?:])\s+", voice) if p.strip()]
    total, acc = sum(len(p) for p in parts), 0
    for p in parts:
        acc += len(p)
        if t <= acc / total * voice_len(voice):
            return p[:1].upper() + p[1:]
    return parts[-1][:1].upper() + parts[-1][1:]


_vlen = {}


def voice_len(v):
    return _vlen.get(v, 1.0)


def card(title, lines, sub=None, img=None, img2=None, caps=(None, None), stat=None, fix=None):
    canvas = np.full((H, W, 3), BG, np.uint8)
    items = [(title, (540, 150), 64, INK, FONT_B, "mm")]
    y = 260
    ims = [(im, cap) for im, cap in ((img, caps[0]), (img2, caps[1])) if im is not None]
    if ims:
        # everything must end above the caption band: shrink the images if needed
        below = (60 + 58 * len(wrap(stat, FONT_B, 46, W - 140)) if stat else 0) + (160 if fix else 0)
        room = CAP_Y - 40 - y - below - 30 * len(ims)
        iw = int(min(W, room / len(ims) * ims[0][0].shape[1] / ims[0][0].shape[0]))
        for im, cap in ims:
            h_ = int(im.shape[0] * iw / im.shape[1])
            x0 = (W - iw) // 2
            canvas[y:y + h_, x0:x0 + iw] = cv2.resize(im, (iw, h_), interpolation=cv2.INTER_CUBIC)
            if cap:
                cv2.rectangle(canvas, (x0, y + h_ - 64), (x0 + iw, y + h_), (0, 0, 0), -1)
                items.append((cap, (540, y + h_ - 32), 36, INK, FONT_B, "mm"))
            y += h_ + 30
    if stat:
        for i, line in enumerate(wrap(stat, FONT_B, 46, W - 140)):
            items.append((line, (540, y + 40 + i * 58), 46, YOU, FONT_B, "mm"))
        y += 60 + 58 * len(wrap(stat, FONT_B, 46, W - 140))
    if fix:
        cv2.rectangle(canvas, (60, y), (W - 60, y + 130), (40, 40, 38), -1)
        items.append(("À travailler", (540, y + 35), 30, INK2, FONT_R, "mm"))
        items.append((fix, (540, y + 88), 36, INK, FONT_B, "mm"))
    for i, line in enumerate(lines):
        items.append((line, (540, 420 + i * 70), 44, INK, FONT_R, "mm"))
    if sub:
        items.append((sub, (540, 230), 34, INK2, FONT_R, "mm"))
    return canvas, items


def tts(lines, models, voice, speed):
    from kokoro_onnx import Kokoro
    k = Kokoro(str(models / "kokoro-v1.0.onnx"), str(models / "voices-v1.0.bin"))
    out = {}
    for s in lines:
        a, sr = k.create(s, voice=voice, speed=speed, lang="fr-fr")
        out[s] = (np.asarray(a, np.float32), sr)
        _vlen[s] = len(a) / sr
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work")
    ap.add_argument("out")
    ap.add_argument("--speed", type=float, default=1.3, help="rally playback speed")
    ap.add_argument("--pre", type=float, default=0.4, help="seconds kept before the serve")
    ap.add_argument("--post", type=float, default=0.5, help="seconds kept after the rally end")
    ap.add_argument("--voice", default="ff_siwis")
    ap.add_argument("--voice-speed", type=float, default=1.1)
    ap.add_argument("--models", default="models")
    args = ap.parse_args()
    w = Path(args.work)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    tr = Tracks(w)
    pts = load_json(w / "points.json")["points"]
    coach = load_json(w / "coaching.json")
    cap = cv2.VideoCapture(str(w / "proxy.mp4"))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_end = cap.get(cv2.CAP_PROP_FRAME_COUNT) / src_fps
    scale = W / cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    voices = [coach["intro"]["voice"]] + [l_["voice"] for l_ in coach["lessons"]] + [coach["outro"]["voice"]]
    print("synthesising the voice-over...")
    audio = tts(voices, Path(args.models), args.voice, args.voice_speed)
    sr = next(iter(audio.values()))[1]

    def src_frame(t):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * src_fps)))
        return cap.read()[1]

    # ---- timeline: (kind, payload, duration)
    segs = [("intro", None, voice_len(coach["intro"]["voice"]) + 0.8)]
    lessons = {}
    for l_ in coach["lessons"]:
        if l_["after_point"] not in {p["n"] for p in pts}:
            raise SystemExit(f"coaching.json: lesson {l_['title']!r} after_point {l_['after_point']} is not a point number")
        if len(l_["freeze"]) + sum(bool(f.get("zoom")) for f in l_["freeze"]) > 2:
            print(f"WARNING: lesson {l_['title']!r}: a card shows 2 images at most (zooms count), the others are left out")
        lessons.setdefault(l_["after_point"], []).append(l_)
    for p in pts:
        t0 = p["r0"] - args.pre
        t1 = min(p["r1"] + args.post, src_end - 0.05) if p["r1"] else src_end - 0.05
        segs.append(("rally", (p, t0, t1), (t1 - t0) / args.speed))
        for l_ in lessons.get(p["n"], []):
            segs.append(("lesson", l_, voice_len(l_["voice"]) + 1.0))
    segs.append(("outro", None, voice_len(coach["outro"]["voice"]) + 1.5))
    total = sum(d for _, _, d in segs)
    print(f"{len(segs)} segments, {total:.1f} s")

    track = np.zeros(int((total + 1) * sr), np.float32)
    tmp = Path(tempfile.mkdtemp(dir=w))
    ff = None
    try:
        wav = tmp / "voice.wav"
        ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                               "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                               "-pix_fmt", "yuv420p", str(tmp / "video.mp4")], stdin=subprocess.PIPE)
        t_out = 0.0
        for kind, data, dur in segs:
            n = int(round(dur * FPS))
            voice = None
            if kind == "intro":
                voice = coach["intro"]["voice"]
                bg = cv2.GaussianBlur(cv2.resize(src_frame(pts[0]["r0"]), (W, int(478 * scale))), (0, 0), 6)
            elif kind == "lesson":
                voice = data["voice"]
                fr = []
                for f in data["freeze"]:
                    raw = src_frame(f["t"])
                    im = draw_video(raw, tr, f["t"], scale, trail_s=0.0)
                    if f.get("mark_px"):               # e.g. where the shuttle lies (proxy pixels)
                        mx, my = (np.array(f["mark_px"]) * scale).astype(int)
                        cv2.circle(im, (mx, my), 30, (0, 230, 255), 4, cv2.LINE_AA)
                        im = text(im, [(f.get("mark_label", "volant"), (mx, my - 42), 32, (0, 230, 255), FONT_B, "ms", 5)])
                    fr.append((im, f["caption"]))
                    if f.get("zoom"):                  # 16:9 close-up on one player (the far one is tiny)
                        b = tr.box_near(f["zoom"], f["t"])
                        if b is None:
                            print(f"WARNING: no {f['zoom']} box near {f['t']} s: zoom left out")
                        else:
                            fr.append((player_zoom(raw, b, (W, int(W * 9 / 16))), f.get("zoom_caption")))
            elif kind == "outro":
                voice = coach["outro"]["voice"]
            if voice:
                a = audio[voice][0]
                s0 = int((t_out + 0.4) * sr)
                track[s0:s0 + len(a)] += a[:len(track) - s0]
            for i in range(n):
                tt = i / FPS
                if kind == "rally":
                    p, t0, t1 = data
                    st = min(t0 + tt * args.speed, t1)
                    frame = src_frame(st)
                    canvas = np.full((H, W, 3), BG, np.uint8)
                    vid = draw_video(frame, tr, st, scale)
                    canvas[VID_Y:VID_Y + vid.shape[0]] = vid[:H - VID_Y]
                    items = []
                    done = p["r1"] is not None and st >= p["r1"] - 0.25
                    score = p["after"] if done and p["after"] else p["before"]
                    who = "toi" if p["server"] == "R" else "l'adversaire"
                    el = max(0.0, min(st, p["r1"] or st) - p["r0"])
                    banner = None
                    if done and p["winner"]:
                        banner = ("POINT POUR TOI", YOU) if p["winner"] == "R" else ("POINT POUR L'ADVERSAIRE", OPP)
                    score_bar(canvas, items, score, f"Point {p['n']} · service : {who} · échange {el:.1f} s", banner)
                    items.append((f"x{args.speed:g}", (W - 30, VID_Y + 30), 30, INK2, FONT_B, "rm"))
                    draw_map(canvas, tr, st, 120, MAP_Y)
                    vr = tr.at("r", st)[1]
                    sx = 120 + court.WIDTH * 56 + 70
                    items += [("Ta vitesse", (sx, MAP_Y + 40), 34, INK2, FONT_R, "lm"),
                              (f"{vr:.1f} m/s" if vr is not None and np.isfinite(vr) else "–", (sx, MAP_Y + 95), 60, YOU, FONT_D, "lm"),
                              ("Durée de l'échange", (sx, MAP_Y + 190), 34, INK2, FONT_R, "lm"),
                              (f"{el:.1f} s", (sx, MAP_Y + 245), 60, INK, FONT_D, "lm"),
                              ("+ = ta base", (sx, MAP_Y + 340), 30, INK2, FONT_R, "lm"),
                              ("profondeur adverse", (sx, CAP_Y - 110), 26, INK2, FONT_R, "lm"),
                              ("approximative", (sx, CAP_Y - 78), 26, INK2, FONT_R, "lm")]
                    if p["rdur"] is None and st > t1 - 1.0:
                        captions(items, "La vidéo s'arrête pendant cet échange.")
                elif kind == "intro":
                    canvas = np.full((H, W, 3), BG, np.uint8)
                    canvas[VID_Y + 300:VID_Y + 300 + bg.shape[0]] = bg
                    items = [(coach["intro"]["title"], (540, 260), 72, INK, FONT_B, "mm"),
                             (coach["intro"]["subtitle"], (540, 360), 34, INK2, FONT_R, "mm")]
                elif kind == "lesson":
                    z = 1.0 + 0.04 * tt / dur                 # slow push-in
                    ims = []
                    for im, _ in fr:
                        hh, ww = im.shape[:2]
                        zi = cv2.resize(im, None, fx=z, fy=z, interpolation=cv2.INTER_LINEAR)
                        oy, ox = (zi.shape[0] - hh) // 2, (zi.shape[1] - ww) // 2
                        ims.append(zi[oy:oy + hh, ox:ox + ww])
                    canvas, items = card(data["title"], [], img=ims[0], img2=ims[1] if len(ims) > 1 else None,
                                         caps=(fr[0][1], fr[1][1] if len(fr) > 1 else None),
                                         stat=data["stat"], fix=data["fix"])
                else:
                    canvas, items = card(coach["outro"]["title"], [f"• {x}" for x in coach["outro"]["items"]],
                                         sub=coach["outro"]["note"])
                    last = [p for p in pts if p["after"]][-1]["after"].split("-")
                    items += [("Score de la vidéo", (540, 760), 36, INK2, FONT_R, "mm"),
                              ("TOI", (300, 900), 44, YOU, FONT_B, "mm"), ("ADV", (780, 900), 44, OPP, FONT_B, "mm"),
                              (f"{last[0]}  –  {last[1]}", (540, 905), 130, INK, FONT_D, "mm"),
                              ("Analyse 100 % locale et gratuite", (540, 1500), 30, INK2, FONT_R, "mm")]
                    if not pts[-1]["r1"]:
                        items.append((f"(la vidéo s'arrête pendant le {pts[-1]['n']}e échange)", (540, 1020), 30, INK2, FONT_R, "mm"))
                    elif not pts[-1]["winner"]:
                        items.append(("(résultat du dernier échange inconnu)", (540, 1020), 30, INK2, FONT_R, "mm"))
                if voice and 0.4 <= tt <= 0.4 + voice_len(voice):
                    captions(items, sentence_at(voice, tt - 0.4))
                ff.stdin.write(text(canvas, items).tobytes())
            t_out += n / FPS
            print(f"  {kind:6s} {dur:5.1f} s  (at {t_out:5.1f} s)", flush=True)
        ff.stdin.close()
        ff.wait()

        import soundfile as sf
        peak = np.abs(track).max() or 1.0
        sf.write(str(wav), track[:int(t_out * sr)] / peak * 0.9, sr)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp / "video.mp4"), "-i", str(wav), "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(tmp / "reel.mp4")],
                       check=True)
        Path(tmp / "reel.mp4").replace(args.out)      # an existing reel is only replaced by a complete one
    finally:
        if ff is not None and ff.poll() is None:   # stopped by an error: free the file before deleting it
            ff.kill()
            ff.wait()
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"reel -> {args.out}  ({t_out:.1f} s)")


if __name__ == "__main__":
    main()
