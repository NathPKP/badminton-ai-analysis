# Guide d'utilisation (français)

Analyse de match de badminton 100 % locale, d'après la description de [KiranYelisetti](https://github.com/KiranYelisetti/badminton-ai-analysis) (étapes 0 à 11). Tout tourne sur le PC, sans GPU.

- Présentation générale (anglais) : [README](../README.md)
- Détails techniques : [PIPELINE.md](PIPELINE.md)

## Installation

Il faut Python 3.12 (`winget install Python.Python.3.12`) et ffmpeg (`winget install Gyan.FFmpeg`). Ensuite, depuis le dossier du projet :

```
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Le script fait quatre choses :
1. il crée `.venv` ;
2. il installe les paquets ;
3. il télécharge les modèles dans `models\` (environ 420 Mo) : `yolox_s.onnx` pour la détection, `pose_landmarker_heavy.task` pour la posture, `kokoro-v1.0.onnx` et `voices-v1.0.bin` pour la voix de la vidéo ;
4. il lance `run.py doctor`, qui vérifie tout et dit comment corriger ce qui manque.

Si Windows Defender (accès contrôlé aux dossiers) est actif, autorise-y Python et ffmpeg ; `doctor` le détecte.

## Analyser une vidéo : `run.py`

Chaque analyse a son propre dossier `analyses\<date>_<nom>\`, avec tous ses fichiers. Une nouvelle analyse n'écrase jamais une ancienne.

Toutes les commandes se lancent dans PowerShell, depuis le dossier du projet, avec le Python du projet : `.venv\Scripts\python.exe`, et non `python`.

Nouvelle analyse (le nom est libre ; `--start` est la fin de l'échauffement, en secondes ; `--lang fr` donne un rapport et une vidéo en français, l'anglais étant la langue par défaut) :

```
.venv\Scripts\python.exe run.py new C:\chemin\video.mp4 match-paul --start 12 --lang fr
```

Reprendre là où ça s'est arrêté :

```
.venv\Scripts\python.exe run.py run match-paul
```

Voir l'état de chaque étape, puis la liste de toutes les analyses :

```
.venv\Scripts\python.exe run.py status match-paul
.venv\Scripts\python.exe run.py list
```

Le dossier s'appelle `<date de la vidéo>_<nom>`, par exemple `2026-10-28_match-paul`. Dans les commandes, un morceau unique du nom suffit (`match-paul`).

`run` lance toutes les étapes à faire, dans l'ordre. Il ne refait que ce qui a changé : le contenu d'un fichier d'entrée ou un réglage. Il s'arrête quand quelqu'un doit intervenir et explique quoi faire ; on relance `run` ensuite. Il y a quatre arrêts possibles :

1. **Calibrage du terrain.** Une fenêtre s'ouvre et tu cliques les points du terrain (environ 2 minutes).
   - Si la caméra était au même endroit qu'une analyse précédente, ajoute `--calib-from <analyse>` à `new`. Les points sont repris sans fenêtre ; il ne reste qu'à vérifier l'image `contact\s2_calib_check.jpg`.
   - Si le terrain jaune ne tombe pas sur les lignes, `run.py run <analyse> --redo calib` rouvre la fenêtre avec les points déjà placés.
2. **`review.json`** (qui sert, et quand chaque échange se termine). Claude le remplit en lisant les planches `contact\timeline`. Un brouillon, `review.todo.json`, est pré-rempli avec les services détectés automatiquement. Demande à Claude : « lis les planches de analyses\<dossier> et écris review.json ».
3. **Contrôle du score.** Si un service contredit la règle du côté de service, `run` s'arrête et indique lesquels. En général, c'est qu'un serveur est mal lu dans `review.json`. Si c'était vraiment une exception (faute, let), `run.py set <analyse> accept_score=[7]` accepte ce service-là, et lui seul.
4. **`coaching.json`** (les conseils, le texte de la voix et la partie `pdf` du rapport écrit).
   - Par défaut (`coach=auto`), `s10_coach.py` l'écrit à partir des chiffres, avec des règles, et le réécrit quand les chiffres changent. `run` ne s'arrête pas.
   - Pour des conseils plus précis, demande à Claude Code : « écris coaching.json pour analyses\<dossier> ». Un fichier modifié est gardé tel quel.
   - Avec `coach=ai`, `run` s'arrête et attend ce fichier. Si les chiffres changent ensuite (par exemple après une correction de `review.json`), il s'arrête pour faire relire les conseils. S'ils restent justes : `run.py run <analyse> --accept coaching`.

Claude Code lit le fichier [`CLAUDE.md`](../CLAUDE.md) du projet, qui lui explique comment faire ces lectures et les vérifier.

À la fin, le dossier de l'analyse contient `rapport_match.pdf` (à envoyer par mail) et `video\reel.mp4` (un Reel vertical commenté).

### Réglages

On les change avec `run.py set <analyse> clé=valeur`. Les étapes concernées sont refaites au prochain `run`.

| Réglage | Rôle |
|---|---|
| `start=12` | fin de l'échauffement, en secondes (obligatoire) |
| `lang=fr` | langue du rapport, de la vidéo et des messages : `en` ou `fr` (les analyses créées avant ce réglage sont en français) |
| `coach=auto` | `auto` : conseils écrits par des règles ; `ai` : attendre un coaching.json écrit par un assistant IA |
| `shorts_hue=[100,135]` | couleur de ton short, qui sert à te reconnaître : bleu `[100,135]` (défaut), rouge `[170,10]`, vert `[40,85]`. C'est une teinte OpenCV de 0 à 179 ; ne marche pas pour un short noir, blanc ou gris |
| `thr=0.82` | seuil de flexion de l'étape 9 (dépend de l'angle de la caméra) |
| `last_winner=R` | gagnant du dernier point, si la vidéo le montre (`R` = toi, `O` = adversaire) |
| `accept_score=[7]` | services qui contredisent la règle du côté mais sont justes (faute, let…) |
| `camera=raised` | caméra surélevée : retire du rapport la limite « caméra au sol » (défaut : `floor` ; `sol` et `haute` marchent aussi) |
| `speed=1.3` | vitesse de lecture des échanges dans la vidéo |
| `"report_date=27 octobre 2026"` | date écrite sur le PDF (par défaut, le jour de création de l'analyse ; les guillemets sont nécessaires à cause des espaces) |

Exemple : `.venv\Scripts\python.exe run.py set match-paul start=15 shorts_hue=[170,10]`. Une valeur vide (`thr=`) supprime le réglage.

`run.py run <analyse> --redo <étape>` refait une étape et tout ce qui en dépend. Étapes possibles : `proxy`, `detect`, `calib`, `players`, `sheets`, `serves`, `score`, `rallies`, `movement`, `pose`, `pickups`, `report`, `pdf`, `video`. Une étape interrompue ou en échec est refaite au `run` suivant.

### Historique

Avant qu'une étape remplace un fichier de résultat, l'ancienne version est copiée dans `history\<date-heure>\` du dossier de l'analyse. Chaque version de `calib.json`, `review.json` et `coaching.json` est gardée de la même façon.

Ne sont pas gardés :
- `proxy.mp4`, refait à partir de la vidéo d'origine ;
- les planches et les figures, refaites à partir des résultats.

Pour une sauvegarde hors du PC, copie le dossier `analyses\` (OneDrive, disque externe). Chaque analyse est complète dans son dossier, `proxy.mp4` compris : elle n'a plus besoin de la vidéo d'origine, sauf pour refaire l'étape 0.

## Contenu d'un dossier d'analyse

| Fichier | Étape | Contenu |
|---|---|---|
| `analysis.json` | | vidéo d'origine et réglages |
| `proxy.mp4` | 0 | copie de travail à 30 images/s |
| `dets.json` | 1 | personnes détectées, 10 fois/s |
| `calib.json` | 2 | calibrage du terrain (clics + homographie) |
| `players.json` | 3 | positions des deux joueurs en mètres |
| `serves_auto.json`, `contact\timeline\` | 4a | services détectés automatiquement, planches de lecture |
| `review.json` | lecture | services et fins d'échange lus à l'œil |
| `serves.json`, `score.json` | 4, 5 | services, score et contrôle de cohérence |
| `points.json` | 6 | un enregistrement par point, avec sa durée |
| `motion.json`, `figures\` | 7 | déplacements, cartes de chaleur |
| `pose_stats.json`, `pose_frames.json` | 8 | posture |
| `pickups.json` | 9 | côté où tombe le volant |
| `report.json`, `report.md` | 10 | tous les chiffres |
| `coaching.json` | conseils | enseignements, texte de la voix, partie `pdf` du rapport |
| `rapport_match.pdf`, `video\reel.mp4` | 10, 11 | les deux rendus |
| `.pipeline.json`, `history\` | | état des étapes, anciennes versions |

Dans `coaching.json`, la partie `pdf` contient les textes du rapport écrit, à la première personne : constats, images, fins de points connues, limites. Sans elle, le PDF reprend les enseignements de la vidéo, tutoiement compris. Le format est décrit en tête de `s10_pdf.py`.

Exemples complets :
- [`examples\coaching.example.json`](../examples/coaching.example.json) ;
- [`examples\review.example.json`](../examples/review.example.json) pour le format de `review.json`.

Chaque étape reste un script autonome (`s0_proxy.py` … `s11_video.py`). `run.py` les lance avec les bons chemins et affiche chaque commande lancée.

## Limites (caméra au sol)

- La profondeur de l'adversaire est faussée : ses zones du terrain ne sont pas utilisées.
- Quand le volant tombe près du filet, on ne voit pas toujours de quel côté.
- Sur peu d'échanges, on obtient des tendances, pas des certitudes.

## Pour la prochaine vidéo

1. Caméra surélevée (2 m ou plus) derrière ta ligne de fond, avec les 4 coins et les lignes de service visibles. Ajoute alors `--camera raised` à `new`.
2. Caméra immobile pendant tout le match, et vidéo originale (pas une copie compressée).
3. Au calibrage, clique aussi des points du fond adverse (ligne de service long, coins du fond).
4. Revérifie le seuil de l'étape 9 (`thr`) : l'étape 9 affiche son accord avec `review.json`.
