"""Fast checks that need no video and no model: python -m unittest discover tests"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")


class I18n(unittest.TestCase):
    def test_same_keys_and_shapes(self):
        from bball.i18n import TEXT
        en, fr = TEXT["en"], TEXT["fr"]
        self.assertEqual(set(en), set(fr))
        for k in en:
            self.assertIs(type(en[k]), type(fr[k]), k)
            if isinstance(en[k], (dict, tuple, list)):
                self.assertEqual(len(en[k]), len(fr[k]), k)
            if isinstance(en[k], dict):
                self.assertEqual(set(en[k]), set(fr[k]), k)

    def test_formatting(self):
        from bball.i18n import Text
        self.assertEqual(Text("fr").num(1.25, 1), "1,2")
        self.assertEqual(Text("en").num(1.25, 2), "1.25")
        self.assertEqual(Text("en").plural(1, "point_won"), "1 point won")
        self.assertEqual(Text("fr").plural(3, "point_won"), "3 points gagnés")
        self.assertEqual(Text("en").pick(3, "h_lessons", n="three"), "The three lessons")

    def test_run_messages_same_keys(self):
        import run as runpy
        self.assertEqual(set(runpy.MSG["en"]), set(runpy.MSG["fr"]))
        for k, v in runpy.MSG["en"].items():
            if isinstance(v, dict):
                self.assertEqual(set(v), set(runpy.MSG["fr"][k]), k)


class Score(unittest.TestCase):
    def test_service_court_rule(self):
        # 0-0: O serves from the right; O wins (serves again at 0-1 from the left); R wins (serves at 1-1 from the left)
        serves = [{"t": 1, "server": "O", "side": "right"}, {"t": 2, "server": "O", "side": "left"},
                  {"t": 3, "server": "R", "side": "left"}, {"t": 4, "server": "R", "side": "left"}]
        with tempfile.TemporaryDirectory() as d:
            Path(d, "serves.json").write_text(json.dumps({"serves": serves}))
            r = run("s5_score.py", Path(d, "serves.json"), Path(d, "score.json"))
            self.assertEqual(r.returncode, 0, r.stderr)
            score = json.loads(Path(d, "score.json").read_text())
        self.assertEqual([p["after"] for p in score["points"]][:3], ["0-1", "1-1", "2-1"])
        self.assertEqual(score["contradictions"], [4])      # at 2-1 R must serve from the right


class Coach(unittest.TestCase):
    def fake_analysis(self, d, lang):
        """Minimal report / motion / points: the player loses most points on his serve."""
        pts, rows, motion = [], [], []
        servers = ["R", "R", "O", "R", "O", "R", "O", "O"]
        winners = ["O", "O", "R", "O", "R", "O", "O", "R"]
        score = [0, 0]
        for n, (s, w) in enumerate(zip(servers, winners), 1):
            t = 10.0 * n
            before = f"{score[0]}-{score[1]}"
            score[0 if w == "R" else 1] += 1
            p = {"n": n, "t": t, "r0": t, "r1": t + 4, "rdur": 4.0, "server": s, "winner": w}
            pts.append(p)
            rows.append(dict(p, before=before, after=f"{score[0]}-{score[1]}", you_at_end={"x": 3.0, "depth": 2.0}))
            motion.append({"n": n, "winner": w, "R": {"trips": []}})
        a = {"avg_speed": 1.0, "peak_speed": 2.5, "distance_m": 30.0, "live_s": 30.0, "back_share": 0.1,
             "front_share": 0.0, "trips": 0, "median_dist_1s_after_trip": None}
        summary = {"on_your_serve": {"won": 0, "lost": 4}, "on_his_serve": {"won": 3, "lost": 1},
                   "pose_you": {}, "examples": {}, "movement_you": {"all": a, "you_won": a, "you_lost": a},
                   "rally_s": {"median": 4.0}}
        Path(d, "report.json").write_text(json.dumps({"points": rows, "summary": summary, "limits": []}))
        Path(d, "motion.json").write_text(json.dumps({"points": motion}))
        Path(d, "points.json").write_text(json.dumps({"points": pts}))
        Path(d, "analysis.json").write_text(json.dumps({"params": {"lang": lang}}))

    def test_rules_find_the_serve(self):
        for lang in ("en", "fr"):
            with tempfile.TemporaryDirectory() as d:
                self.fake_analysis(d, lang)
                r = run("s10_coach.py", d)
                self.assertEqual(r.returncode, 0, r.stderr)
                c = json.loads(Path(d, "coaching.json").read_text())
            self.assertEqual(c["generated"], "auto")
            self.assertEqual([l_["id"] for l_ in c["lessons"]], ["serve"])
            self.assertEqual(c["lessons"][0]["after_point"], 1)       # first point lost on his serve
            self.assertEqual(len(c["pdf"]["lessons"]), 1)
            def strings(x):
                if isinstance(x, str):
                    yield x
                elif isinstance(x, dict):
                    for k, v in x.items():
                        if k != "evidence":
                            yield from strings(v)
                elif isinstance(x, list):
                    for v in x:
                        yield from strings(v)
            unfilled = [s for s in strings(c) if "{" in s or "}" in s]
            self.assertEqual(unfilled, [])                              # every displayed template filled


class Benchmark(unittest.TestCase):
    def test_tv_rallies(self):
        import numpy as np
        from s12_benchmark import tv_rallies
        t = np.arange(0, 30, 0.1)
        view = (t >= 2) & (t < 14)                       # main view 2-14 s, then a close-up
        speed = np.where((t >= 5) & (t < 12), 2.0, 0.3)   # still serve preparation, rally 5-12 s
        (t0, t1, cut0, cut1), = tv_rallies(t, view, speed)
        self.assertAlmostEqual(t0, 5.0, delta=0.11)
        self.assertAlmostEqual(t1, 12.2, delta=0.11)       # last fast frame + 0.3 s
        self.assertFalse(cut0 or cut1)


class Shuttle(unittest.TestCase):
    def test_blob_and_gaps(self):
        import numpy as np

        import s13_shuttle as s
        heat = np.zeros((288, 512), np.float32)
        self.assertIsNone(s.blob_centre(heat))
        heat[10:13, 20:23] = 0.9                       # small blob
        heat[100:110, 200:210] = 0.8                   # largest blob wins
        self.assertEqual(s.blob_centre(heat), (205, 105))
        seen = [True, True, False, False, True, False, True]
        ys = [100] * 7
        self.assertEqual(s.inpaint_mask(ys, seen, 14).tolist(), [0, 0, 1, 1, 0, 1, 0])
        long_gap = [True, True] + [False] * 20 + [True]
        self.assertFalse(s.inpaint_mask([100] * 23, long_gap, 14).any())   # too long: left empty
        self.assertFalse(s.inpaint_mask([5] * 7, seen, 14).any())          # near the top edge: left empty
        self.assertFalse(s.inpaint_mask(ys, seen, 14, shots=[0, 0, 0, 0, 1, 1, 1])[2:4].any())  # across a cut

    def test_backgrounds_follow_the_camera(self):
        import numpy as np

        import s13_shuttle as s
        shots = np.array([0] * 60 + [1] * 90)
        moving = np.array([False] * 60 + [False] * 30 + [True] * 60)
        self.assertEqual(s.segments(shots, moving, 30, "global"), [(0, 149, 0, 149)])
        segs = s.segments(shots, moving, 30, "shot")
        self.assertEqual(segs[:2], [(0, 59, 0, 59), (60, 89, 60, 89)])     # one per still stretch of each shot
        self.assertEqual(segs[2:], [(90, 119, 60, 149), (120, 149, 90, 149)])  # moving: per second, same shot only


class CoachTexts(unittest.TestCase):
    def test_side_rule_texts(self):
        import s10_coach
        for lang, S in s10_coach.L.items():
            # the side words must not share a key with the side lesson texts (that clash crashed the rule)
            self.assertEqual(set(S["side_word"]), {"right", "left"}, lang)
            for key in ("title", "stat", "voice", "caption"):
                S["side"][key].format(i=1, N="1", side=S["side_word"]["right"], k=3, nl=5, kw=1, nw=6, list="2",
                                      k2=2, mmss="2:41")


class Settings(unittest.TestCase):
    def test_parse_and_num(self):
        import run as runpy
        self.assertEqual(runpy.num(12.0), "12")
        self.assertEqual(runpy.num(12), "12")
        self.assertEqual(runpy.parse_value("shorts_hue", "[170,10]", "en"), [170, 10])
        self.assertEqual(runpy.parse_value("accept_score", "[7]", "en"), [7])
        self.assertEqual(runpy.parse_value("camera", "raised", "en"), "haute")
        with self.assertRaises(SystemExit):
            runpy.parse_value("accept_score", "true", "en")
        with self.assertRaises(SystemExit):
            runpy.parse_value("lang", "de", "en")


if __name__ == "__main__":
    unittest.main()
