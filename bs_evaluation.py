"""Auswertung: Täuschungsmaße je Stufe (exakt), Lösbarkeitsnachweis, Live-Paar (Bezug neben gewählter Beschreibung), Lesen und Aggregieren der vorgerechneten Studie.
Die Studie (Beschreibungen × Stufen, Genomlängen-Experiment) wird beim Bau gerechnet (`generate_precomputed.py`) und hier nur gelesen."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import bs_constants as C
import bs_maze as M
import bs_robot as R
import bs_search as S

PRECOMPUTED_PATH = Path(__file__).resolve().parent / "precomputed_sweep.json"
SELECTIONS = (C.FITNESS,) + C.DESCRIPTORS               # Zeilen der Studie: Fitness als Bezug, dann die zwölf Beschreibungen


def level_metrics(level):
    """Exakte Täuschungsmaße der Stufe: kürzester Weg (Zellen), Luftlinie Start–Ziel, Umwegfaktor, Fitness-Distanz-Korrelation, lokale Minima (Zahl), Abstand des tiefsten Minimums zum Ziel, freie Zellen."""
    g = M.make_maze(level)
    path = M.shortest_path(g)
    return {"path": len(path) - 1, "straight": float(np.hypot(C.START[0] - C.GOAL[0], C.START[1] - C.GOAL[1])), "detour": M.detour_ratio(g), "fdc": M.fitness_distance_correlation(g),
            "minima": len(M.local_minima(g)), "trap": M.trap_distance(g), "free": int((~g).sum())}


def solvability(level):
    """Lösbarkeitsnachweis: eine aus dem kürzesten Pfad abgeleitete Steuerfolge fährt in diesem Labyrinth ins Ziel. Gibt (Abstand am Ende, benötigte Schritte) zurück."""
    g = M.make_maze(level)
    genome = R.controller_from_path(M.shortest_path(g))
    end = R.simulate(genome[None], g)
    return float(R.distance_to_goal(end)[0]), int((np.abs(genome).sum(axis=1) > 0).sum())


def live_run(level, gens, seed, descriptor, k):
    """Ein Lauf mit der Beschreibung `descriptor` auf der gewählten Stufe, samt bester Fahrt (Positionen je Schritt)."""
    g = M.make_maze(level)
    res = S.run_search(g, seed, descriptor, gens=gens, k=k)
    _, track = R.simulate(res.best_genome[None], g, trajectories=True)
    return res, track[0]


def live_pair(level, gens, seed, descriptor, k):
    """Derselbe Seed, dieselbe Stufe: der Lauf mit der Endposition (Bezug) und der Lauf mit der gewählten Beschreibung. Gibt ((Lauf, Fahrt) des Bezugs, (Lauf, Fahrt) der Beschreibung) zurück."""
    return live_run(level, gens, seed, C.BASELINE, k), live_run(level, gens, seed, descriptor, k)


@lru_cache(maxsize=1)
def load_precomputed():
    return json.loads(PRECOMPUTED_PATH.read_text(encoding="utf-8"))


def success_count(runs, budget=None):
    """Zahl der Läufe, die den Zielkreis erreichten (höchstens `budget` Evaluationen, wenn angegeben)."""
    return sum(1 for r in runs if r is not None and (budget is None or r <= budget))


def median_solved(runs):
    """Median der Evaluationen bis zum Erfolg über die erfolgreichen Läufe (None, wenn kein Lauf erfolgreich war)."""
    ok = [r for r in runs if r is not None]
    return float(np.median(ok)) if ok else None


def level_info(pre, level):
    return pre["levels"][str(level)]["metrics"]


def study_cell(pre, level, selection):
    return pre["levels"][str(level)]["selections"][selection]


def length_cell(pre, selection, steps):
    return pre["length"]["cells"][selection][str(steps)]
