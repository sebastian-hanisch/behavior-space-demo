"""Konstanten der Verhaltensraum-Demo: Labyrinth und Roboter wie in Stück 1 und 2, zwölf Verhaltensbeschreibungen, Studien-Achsen, Regler, Voreinstellungen.
Längen in Zellen (Zellraster N × N, Zelle (i, j) belegt den Bereich [i, i + 1) × [j, j + 1))."""


def fmt_int(n):
    """Ganzzahl mit Leerzeichen als Tausendertrenner (10000 -> 10 000), wie in der README; so lässt sich ein halber Median (1 308.5) nicht mit einem Tausenderpunkt verwechseln."""
    return f"{n:,}".replace(",", " ")


def fmt_pct(x, digits=0):
    """Anteil als Prozent mit Leerzeichen (0.086 -> "9 %", mit digits=1 "8.6 %")."""
    return f"{x:.{digits}%}".replace("%", " %")


def fmt_med(x):
    """Median der Evaluationen: ganze Zahl mit Leerzeichen als Tausendertrenner, ein halber Wert behält seine Stelle (988.5, 1 308.5); None = kein Lauf war erfolgreich."""
    if x is None:
        return "–"
    return fmt_int(int(x)) if x == int(x) else f"{x:,.1f}".replace(",", " ")


N = 24                                       # Kantenlänge des Labyrinths in Zellen (Rand ist Wand)
START = (3.5, 3.5)                           # Mitte der Startzelle (3, 3)
GOAL = (3.5, 20.5)                           # Mitte der Zielzelle (3, 20)
SUCCESS_RADIUS = 2.0                         # Erfolg: Endposition höchstens so weit vom Ziel
START_HEADING = 0.7853981633974483           # Anfangsrichtung π/4
STEPS = 140                                  # Länge der Steuerfolge
TURN_MAX = 0.6                               # Drehung je Schritt (rad), Gen ∈ [−1, 1]
SPEED_MAX = 0.7                              # Fahrstrecke je Schritt (Zellen), Gen ∈ [0, 1]

POP = 100                                    # Individuen je Generation
ELITE = 2
TOURNAMENT = 3
MUT_SIGMA = 0.25                             # Standardabweichung der Mutation je verändertem Gen
MUT_RATE = 0.15                              # Anteil der Gene, die je Kind mutieren

LEVELS = (0, 1, 2, 3, 4)
LEVEL_LABELS = {0: "0: freies Feld", 1: "1: Öffnung links", 2: "2: Öffnung in der Mitte", 3: "3: Öffnung rechts", 4: "4: Schlangenlinie"}
GAP = 3                                      # Breite einer Öffnung in der Wand (Zellen)

# Auswahl: nach Fitness (Bezug) oder nach Novelty in einem der zwölf Verhaltensräume. Alle Novelty-Läufe ohne Archiv (Stück 2: das Archiv bringt hier nichts).
FITNESS = "fitness"
BASELINE = "end"                             # Bezug: die Endposition wie in Stück 2
K_DEFAULT = 15                               # Nachbarn der Novelty (k)

# Gruppen der Beschreibungen: wo im Raum, ohne Ortsbezug, mit Aufgabenwissen
GROUPS = ("space", "blind", "knowledge")
GROUP_LABELS = {"space": "mit Ortsbezug", "blind": "ohne Ortsbezug", "knowledge": "mit Aufgabenwissen"}
DESCRIPTORS = ("end", "end_heading", "coarse", "mid", "way4", "way8", "x", "y", "path_turn", "gene_means", "bfs", "euclid")
DESCRIPTOR_GROUP = {"end": "space", "end_heading": "space", "coarse": "space", "mid": "space", "way4": "space", "way8": "space", "x": "space", "y": "space",
                    "path_turn": "blind", "gene_means": "blind", "bfs": "knowledge", "euclid": "knowledge"}
DESCRIPTOR_DIM = {"end": 2, "end_heading": 4, "coarse": 2, "mid": 2, "way4": 8, "way8": 16, "x": 1, "y": 1, "path_turn": 2, "gene_means": 2, "bfs": 1, "euclid": 1}
DESCRIPTOR_LABELS = {
    "end": "Endposition (Bezug)", "end_heading": "Endposition und Endrichtung", "coarse": "grobe Endposition (4×4-Blöcke)", "mid": "Position nach der halben Fahrt",
    "way4": "4 Wegpunkte der Fahrt", "way8": "8 Wegpunkte der Fahrt", "x": "nur x der Endposition", "y": "nur y der Endposition",
    "path_turn": "Weglänge und Gesamtdrehung", "gene_means": "Mittelwerte der Gene", "bfs": "wahrer Restweg zum Ziel", "euclid": "Luftlinie zum Ziel",
}
HEADING_SCALE = 3.0                          # Endrichtung als (cos, sin) mal 3, damit sie neben Zellen als Längen zählt
TURN_SCALE = 0.2                             # Gesamtdrehung (Summe |Drehgen|) mal 0.2
GENE_SCALE = 20.0                            # Mittelwerte der Gene mal 20
COARSE_BLOCK = 4                             # Kantenlänge der groben Blöcke (Zellen)

STUDY_SEEDS = 20                             # Läufe je Zelle der Studie
STUDY_GENS = 400                             # Generationen je Lauf (bei POP = 100: 40 000 Evaluationen)
STUDY_BUDGET = POP * STUDY_GENS
STUDY_LEVELS = (3, 4)                        # Stufen der Beschreibungs-Studie
LENGTH_STEPS = (140, 180, 220, 300)          # Längen der Steuerfolge im Genomlängen-Experiment (Stufe 4, Fitness und Endposition)
LENGTH_SELECTIONS = (FITNESS, BASELINE)

GENS_MIN, GENS_MAX, GENS_STEP, DEFAULT_GENS = 10, 200, 10, 100      # Regler: Generationen des Live-Laufs
K_MIN, K_MAX, K_STEP = 1, 30, 1
DEFAULT_LEVEL = 3
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_DESCRIPTOR = "mid"

PRESET_ORDER = ("Falle: Endposition gegen halbe Fahrt", "Ohne Ortsbezug", "Mit Aufgabenwissen", "Schlangenlinie")


def _preset(level=DEFAULT_LEVEL, gens=DEFAULT_GENS, descriptor=DEFAULT_DESCRIPTOR, k=K_DEFAULT):
    return {"level": level, "gens": gens, "seed": DEFAULT_SEED, "descriptor": descriptor, "k": k}


PRESETS = {
    "Falle: Endposition gegen halbe Fahrt": _preset(),
    "Ohne Ortsbezug": _preset(descriptor="gene_means", gens=200),
    "Mit Aufgabenwissen": _preset(descriptor="bfs"),
    "Schlangenlinie": _preset(level=4, gens=200, descriptor="bfs"),
}
# Zahlen aus der vorgerechneten Studie (20 Läufe je Zelle); tests/test_claims.py rechnet jede nach
PRESET_HELP = {
    "Falle: Endposition gegen halbe Fahrt": "Stufe 3: mit der Position nach halber Fahrt statt der Endposition löst Novelty ebenfalls 20 von 20 Läufen, im Median aber erst nach 4 478 statt 988.5 Evaluationen; die Fitness schafft 11 von 20.",
    "Ohne Ortsbezug": "Mittelwerte der Gene statt eines Ortes: Stufe 3 gelingt in 18 von 20 Läufen (Median 9 579.5 Evaluationen), und die Suche besucht im Mittel nur 385.6 von 465 Zellen.",
    "Mit Aufgabenwissen": "Der wahre Restweg zum Ziel als Beschreibung löst Stufe 3 in 20 von 20 Läufen (Median 790, der kleinste aller zwölf Mediane), Stufe 4 trotzdem in 0 von 20 (Restweg im Mittel 24.75 Zellen).",
    "Schlangenlinie": "Stufe 4: keine der zwölf Beschreibungen löst einen einzigen von 240 Läufen, auch nicht der wahre Restweg zum Ziel; mit 300 statt 140 Schritten löst Novelty auf der Endposition 2 von 20.",
}
