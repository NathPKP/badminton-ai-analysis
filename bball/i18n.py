"""Fixed texts of the report, the reel and run.py, in English ("en") and French ("fr").

T = Text(lang); T("key", **values) formats one text; T.num / T.date / T.count / T.plural follow the
language's conventions (decimal comma, number words, plurals).
"""
import datetime

LANGS = ("en", "fr")
VOICE = {"en": ("af_heart", "en-us"), "fr": ("ff_siwis", "fr-fr")}      # Kokoro voice, language code

TEXT = {
    "en": {
        # ---- PDF report (first person: the player's own report)
        "title": "Video analysis of my badminton match",
        "subtitle": "Singles · {n} rallies filmed ({min} min {sec:02d} of video) · report of {date} · "
                    "automatic AI analysis, run locally",
        "k_score": "Score on the video",
        "stop_unfinished": " (the video stops during rally {n})",
        "stop_unknown": " (result of the last rally unknown)",
        "k_rally": "Rally length",
        "v_rally": "median {med} s ({lo} to {hi} s)",
        "k_speed": "My speed in play",
        "v_speed": "{avg} m/s on average, peak {peak} m/s; {dist} m covered in {live} s of actual play",
        "k_serve": "On serve",
        "v_serve": "when I serve: {won} out of {n1}; when my opponent serves: {w2} out of {n2}",
        "point_won": ("point won", "points won"),
        "point": ("point", "points"),
        "h_summary": "In short",
        "summary_intro": ("{N} point to work on comes out of the analysis:",
                          "{N} points to work on come out of the analysis:"),
        "few": "With only {n} rallies, these are trends, not certainties. ",
        "each_ref": "Each finding below points to a precise moment of the video (minute:second), so it can be checked.",
        "h_lessons": ("The lesson", "The {n} lessons"),
        "finding": "Finding.",
        "work": "To work on.",
        "table_head": ["#", "Video", "Server", "Score", "Length", "Won by", "How the point ends"],
        "who": {"R": "me", "O": "opp.", None: "?"},
        "end_first": {"my_winner": "my winning shot (or his fault at the net)",
                      "his_fault": "his fault (shuttle out)",
                      "his_winner": "his winning shot (or my fault at the net)",
                      "my_out": "my shot out"},
        "uncertain": " (uncertain)",
        "h_table": "The match, point by point",
        "table_note": "Score: me – opponent. “How the point ends” combines the winner (who serves next) and the side "
                      "where the shuttle came down: for example, a shuttle on my side on a point I lost = his winning "
                      "shot or my fault at the net.",
        "h_move": "Movement",
        "speed": "Average speed in play: <b>{v} m/s</b>",
        "speed_usual": ", the usual value in singles (about 1 m/s)",
        "speed_same": ", the same on points won and lost.",
        "speed_diff": " ({w} m/s on points won, {l} on points lost).",
        "zones": "I spend {b} % of playing time in the back of the court and ",
        "front_rare": "almost never in front of the short service line ({f} %).",
        "front": "{f} % in front of the short service line.",
        "zone_main": " My main area is {across}, {depth} (+).",
        "across": {"center": "the centre", "left": "the left side", "right": "the right side"},
        "depth": {"behind": "a little behind my base", "ahead": "a little ahead of my base", "on": "around my base"},
        "recovery": "One second after being pulled away from my base, I am still {l} m from it on points lost, "
                    "against {w} m on points won ({trips}{sure}).",
        "trips_same": "{n} each",
        "trips_diff": "{l} on points lost, {w} on points won",
        "trip": ("move", "moves"),
        "trips_sure": ": to be confirmed on more points",
        "cap_heat": "Time spent in each place of my half during play (darker = longer).",
        "h_bench": "Compared with professional players",
        "bench_intro": "The same measures on two BWF singles finals filmed for TV ({n} rallies, both players), "
                       "computed exactly like mine:",
        "bench_head": ["Per rally (median)", "Me", "Pros"],
        "bench_rows": ["Rally length", "Average speed", "Peak speed", "Distance per 10 s of play",
                       "Distance to my base 1 s after being pulled out"],
        "bench_note": "Medians per rally (hence an average speed different from the one above, over all the play). Pro "
                      "rally ends are found automatically, about 1 s late, and a raised TV camera measures more precisely "
                      "than a phone on the floor: orders of magnitude, not a ranking.",
        "heat_net": "net",
        "heat_base": "my baseline (camera side)",
        "h_method": "How it is measured, and the limits",
        "camera_sol": "phone on the floor behind my baseline",
        "camera_haute": "raised camera behind my baseline",
        "method": "The video ({camera}) is analysed frame by frame: players detected 10 times per second, court "
                  "calibrated to work in metres, posture from a skeleton model, serves and rally ends checked frame by "
                  "frame, score rebuilt with the rule “the winner serves” and checked with the service-court rule "
                  "({check}). Everything runs on a PC, with free tools.",
        "check_ok": "{n} serves out of {n} consistent",
        "check_accepted": ("{ok} serves out of {n} consistent; serve {list} breaks the rule and was kept as read "
                           "(a fault, a let, or a reading error)",
                           "{ok} serves out of {n} consistent; serves {list} break the rule and were kept as read "
                           "(faults, lets, or reading errors)"),
        "check_bad": ("{ok} serves out of {n} consistent; serve {list} must be checked again",
                      "{ok} serves out of {n} consistent; serves {list} must be checked again"),
        "limits": "Limits.",
        "limit_sep": "; ",
        "page": "page {n}",
        "mark": "shuttle",
        # ---- report.md / report.json (second person: talking to the player)
        "end_second": {"my_winner": "your winning shot (or his fault at the net)",
                       "his_fault": "his fault (shuttle out or serve)",
                       "his_winner": "his winning shot (or your fault at the net)",
                       "my_out": "your shot out"},
        "note_unfinished": "the video stops during the rally",
        "limit_depth": "opponent depth biased (camera on the floor): his court zones are not used",
        "limit_few": "only {n} rallies: trends, not certainties",
        "limit_guessed": "side of the shuttle inferred from the pick-up (not seen directly) for point(s) {list}",
        "md_title": "# Match analysis",
        "md_points": "**{won} points won, {lost} lost** over {n} finished rallies (score {a} → {b}).",
        "md_head": "| # | Serve | Score | Length | Won by | Shuttle landed | End of point |",
        "md_who": {"R": "you", "O": "opponent", None: "?"},
        "md_findings": "## Key findings",
        "md_serve": "- On your serve: {w1} won, {l1} lost. On his: {w2} won, {l2} lost.",
        "md_length": "- Rallies of 5 s or less: {a}. 5 to 10 s: {b}. Over 10 s: {c}.",
        "md_speed": "- Average speed in play: {v} m/s; 1 s after a trip out of position you are still {w} m from "
                    "your base on points won, {l} m on points lost.",
        "md_racket": "- Racket up in the back of the court: {w} of the time on points won, {l} on points lost.",
        "md_limits": "## Limits",
        # ---- reel
        "you_tag": "YOU",
        "opp_tag": "OPP",
        "point_you": "POINT FOR YOU",
        "point_opp": "POINT FOR YOUR OPPONENT",
        "bar": "Point {n} · serve: {who} · rally {el} s",
        "bar_you": "you",
        "bar_opp": "your opponent",
        "your_speed": "Your speed",
        "rally_len": "Rally length",
        "your_base": "+ = your base",
        "opp_depth": ("opponent depth", "approximate"),
        "video_stops": "The video stops during this rally.",
        "video_score": "Score on the video",
        "local": "100 % local and free analysis",
        "reel_unfinished": "(the video stops during rally {n})",
        "reel_unknown": "(result of the last rally unknown)",
        "to_work": "To work on",
    },
    "fr": {
        "title": "Analyse vidéo de mon match de badminton",
        "subtitle": "Simple · {n} échanges filmés ({min} min {sec:02d} de vidéo) · rapport du {date} · "
                    "analyse automatique par IA, faite en local",
        "k_score": "Score sur la vidéo",
        "stop_unfinished": " (la vidéo s'arrête pendant le {n}<super>e</super> échange)",
        "stop_unknown": " (résultat du dernier échange inconnu)",
        "k_rally": "Durée des échanges",
        "v_rally": "médiane {med} s (de {lo} à {hi} s)",
        "k_speed": "Ma vitesse en jeu",
        "v_speed": "{avg} m/s en moyenne, pointe à {peak} m/s ; {dist} m parcourus en {live} s de jeu effectif",
        "k_serve": "Au service",
        "v_serve": "quand je sers : {won} sur {n1} ; quand il sert : {w2} sur {n2}",
        "point_won": ("point gagné", "points gagnés"),
        "point": ("point", "points"),
        "h_summary": "En bref",
        "summary_intro": ("{N} point à travailler ressort de l'analyse :",
                          "{N} points à travailler ressortent de l'analyse :"),
        "few": "Sur {n} échanges seulement, ce sont des tendances, pas des certitudes. ",
        "each_ref": "Chaque constat ci-dessous renvoie à un moment précis de la vidéo (minute:seconde) pour pouvoir le vérifier.",
        "h_lessons": ("L'enseignement", "Les {n} enseignements"),
        "finding": "Constat.",
        "work": "À travailler.",
        "table_head": ["#", "Vidéo", "Service", "Score", "Durée", "Gagné par", "Comment le point se termine"],
        "who": {"R": "moi", "O": "adv.", None: "?"},
        "end_first": {"my_winner": "mon coup gagnant (ou sa faute au filet)",
                      "his_fault": "sa faute (volant dehors)",
                      "his_winner": "son coup gagnant (ou ma faute au filet)",
                      "my_out": "ma frappe sortie"},
        "uncertain": " (incertain)",
        "h_table": "Le match point par point",
        "table_note": "Score : moi – adversaire. « Comment le point se termine » combine le gagnant (celui qui sert "
                      "ensuite) et le côté où le volant est tombé : par exemple, un volant de mon côté sur un point "
                      "perdu = son coup gagnant ou ma faute au filet.",
        "h_move": "Déplacements",
        "speed": "Vitesse moyenne en jeu : <b>{v} m/s</b>",
        "speed_usual": ", la valeur habituelle en simple (environ 1 m/s)",
        "speed_same": ", identique sur les points gagnés et perdus.",
        "speed_diff": " ({w} m/s sur les points gagnés, {l} sur les points perdus).",
        "zones": "Je passe {b} % du temps de jeu au fond du court et ",
        "front_rare": "presque jamais devant la ligne de service court ({f} %).",
        "front": "{f} % devant la ligne de service court.",
        "zone_main": " Ma zone principale est {across}, {depth} (+).",
        "across": {"center": "le centre", "left": "le côté gauche", "right": "le côté droit"},
        "depth": {"behind": "un peu en arrière de ma base", "ahead": "un peu en avant de ma base", "on": "autour de ma base"},
        "recovery": "Une seconde après avoir été déplacé loin de ma base, je suis encore à {l} m de celle-ci sur les "
                    "points perdus, contre {w} m sur les points gagnés ({trips}{sure}).",
        "trips_same": "{n} de chaque côté",
        "trips_diff": "{l} sur les points perdus, {w} sur les points gagnés",
        "trip": ("déplacement", "déplacements"),
        "trips_sure": " : à confirmer sur plus de points",
        "cap_heat": "Temps passé à chaque endroit de ma moitié de terrain pendant le jeu (plus foncé = plus longtemps).",
        "h_bench": "Comparaison avec des joueurs professionnels",
        "bench_intro": "Les mêmes mesures sur deux finales BWF de simple filmées pour la télévision ({n} échanges, "
                       "les deux joueurs), calculées exactement comme les miennes :",
        "bench_head": ["Par échange (médiane)", "Moi", "Pros"],
        "bench_rows": ["Durée de l'échange", "Vitesse moyenne", "Vitesse de pointe", "Distance par 10 s de jeu",
                       "Distance à ma base 1 s après un déplacement"],
        "bench_note": "Médianes par échange (d'où une vitesse moyenne différente de celle ci-dessus, calculée sur tout "
                      "le jeu). Chez les pros, la fin des échanges est trouvée automatiquement, environ 1 s trop tard, et "
                      "une caméra TV en hauteur mesure plus finement qu'un téléphone au sol : des ordres de grandeur, pas "
                      "un classement.",
        "heat_net": "filet",
        "heat_base": "ma ligne de fond (côté caméra)",
        "h_method": "Comment c'est mesuré, et les limites",
        "camera_sol": "téléphone posé au sol derrière ma ligne de fond",
        "camera_haute": "caméra surélevée derrière ma ligne de fond",
        "method": "La vidéo ({camera}) est analysée image par image : détection des joueurs 10 fois par seconde, "
                  "calibrage du terrain pour passer en mètres, posture avec un modèle de squelette, services et fins "
                  "d'échange relus image par image, score reconstitué avec la règle « le gagnant sert » et contrôlé "
                  "par la règle du côté de service ({check}). Tout tourne sur un PC, avec des outils gratuits.",
        "check_ok": "{n} services sur {n} cohérents",
        "check_accepted": ("{ok} services sur {n} cohérents ; le service {list} contredit la règle et a été gardé tel "
                           "quel (faute, let, ou erreur de lecture)",
                           "{ok} services sur {n} cohérents ; les services {list} contredisent la règle et ont été "
                           "gardés tels quels (fautes, lets, ou erreurs de lecture)"),
        "check_bad": ("{ok} services sur {n} cohérents ; le service {list} est à revérifier",
                      "{ok} services sur {n} cohérents ; les services {list} sont à revérifier"),
        "limits": "Limites.",
        "limit_sep": " ; ",
        "page": "page {n}",
        "mark": "volant",
        "end_second": {"my_winner": "ton coup gagnant (ou sa faute au filet)",
                       "his_fault": "sa faute (volant dehors ou service)",
                       "his_winner": "son coup gagnant (ou ta faute au filet)",
                       "my_out": "ta frappe sortie"},
        "note_unfinished": "la vidéo s'arrête pendant l'échange",
        "limit_depth": "profondeur de l'adversaire biaisée (caméra au sol) : ses zones ne sont pas utilisées",
        "limit_few": "{n} échanges seulement : des tendances, pas des certitudes",
        "limit_guessed": "côté du volant déduit du ramassage (pas vu directement) pour le(s) point(s) {list}",
        "md_title": "# Analyse du match",
        "md_points": "**{won} points gagnés, {lost} perdus** sur {n} échanges terminés (score {a} → {b}).",
        "md_head": "| # | Service | Score | Durée | Gagné par | Volant tombé | Fin du point |",
        "md_who": {"R": "toi", "O": "adversaire", None: "?"},
        "md_findings": "## Ce qui ressort",
        "md_serve": "- Sur ton service : {w1} gagné(s), {l1} perdu(s). Sur le sien : {w2} gagné(s), {l2} perdu(s).",
        "md_length": "- Échanges de 5 s ou moins : {a}. De 5 à 10 s : {b}. Plus de 10 s : {c}.",
        "md_speed": "- Vitesse moyenne en jeu : {v} m/s ; 1 s après une sortie de position, tu es encore à {w} m de "
                    "ta base sur les points gagnés, {l} m sur les points perdus.",
        "md_racket": "- Raquette haute au fond du court : {w} du temps sur les points gagnés, {l} sur les points perdus.",
        "md_limits": "## Limites",
        "you_tag": "TOI",
        "opp_tag": "ADV",
        "point_you": "POINT POUR TOI",
        "point_opp": "POINT POUR L'ADVERSAIRE",
        "bar": "Point {n} · service : {who} · échange {el} s",
        "bar_you": "toi",
        "bar_opp": "l'adversaire",
        "your_speed": "Ta vitesse",
        "rally_len": "Durée de l'échange",
        "your_base": "+ = ta base",
        "opp_depth": ("profondeur adverse", "approximative"),
        "video_stops": "La vidéo s'arrête pendant cet échange.",
        "video_score": "Score de la vidéo",
        "local": "Analyse 100 % locale et gratuite",
        "reel_unfinished": "(la vidéo s'arrête pendant le {n}e échange)",
        "reel_unknown": "(résultat du dernier échange inconnu)",
        "to_work": "À travailler",
    },
}

NUMBER_WORDS = {"en": ["No", "One", "Two", "Three", "Four", "Five", "Six"],
                "fr": ["Aucun", "Un", "Deux", "Trois", "Quatre", "Cinq", "Six"]}
MONTHS = {"en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                 "November", "December"],
          "fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
                 "novembre", "décembre"]}


class Text:
    def __init__(self, lang):
        if lang not in LANGS:
            raise SystemExit(f"unknown language {lang!r}: use one of {', '.join(LANGS)}")
        self.lang = lang

    def __call__(self, key, **kw):
        s = TEXT[self.lang][key]
        return s.format(**kw) if kw and isinstance(s, str) else s

    def num(self, x, d=1):
        s = f"{x:.{d}f}"
        return s.replace(".", ",") if self.lang == "fr" else s

    def count(self, n):
        """Number as a word at the start of a sentence (Three), digits past six."""
        words = NUMBER_WORDS[self.lang]
        return words[n] if n < len(words) else str(n)

    def plural(self, n, key):
        """'1 point won' / '3 points won' from a (singular, plural) entry."""
        one, many = TEXT[self.lang][key]
        return f"{n} {one if n <= 1 else many}"

    def pick(self, count, key, /, **kw):
        """Singular or plural version of a whole sentence."""
        one, many = TEXT[self.lang][key]
        return (one if count <= 1 else many).format(**kw)

    def date(self, d=None):
        d = d or datetime.date.today()
        return f"{d.day} {MONTHS[self.lang][d.month - 1]} {d.year}"


def analysis_lang(work):
    """Language of an analysis folder: analysis.json params.lang, French for folders made before it existed."""
    import json
    from pathlib import Path
    meta = Path(work) / "analysis.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8")).get("params", {}).get("lang", "fr")
    return "en"
