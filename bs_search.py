"""Suche mit Fitness oder mit Novelty in einem wählbaren Verhaltensraum: derselbe Genetische Algorithmus wie in Stück 1 und 2 (Turnier 3, die zwei Besten bleiben, Mutation ohne Kreuzung); gewählt wird nach

* **Fitness:** −Abstand der Endposition zum Ziel (Bezug), oder
* **Novelty** (Lehman und Stanley 2008, 2011): mittlerer Abstand der Verhaltensbeschreibung eines Individuums zu den `k` nächsten Beschreibungen der Population (ohne Archiv; in Stück 2 brachte das Archiv hier nichts).
  Die Beschreibung (`bs_descriptors.py`) ist die einzige Stellschraube; das Ziel geht nur in die Messung ein (Erfolg: eine Endposition höchstens 2 Zellen vom Ziel).

Zufall nur über `numpy.random.default_rng(seed)` (PCG64, plattformstabil)."""

from dataclasses import dataclass, field

import numpy as np

import bs_constants as C
import bs_descriptors as D
import bs_maze as M
import bs_robot as R

NEEDS_TRACK = ("mid", "way4", "way8", "path_turn")        # Beschreibungen, die mehr als die Endposition aus der Fahrt brauchen


@dataclass
class SearchResult:
    best_distance: list                  # kleinster Abstand zum Ziel je Generation
    visited_cells: list                  # Zahl der bisher besuchten Zellen (Endpositionen) nach jeder Generation
    best_progress: list                  # kleinster wahrer Restweg (Zellen) aller bisher besuchten Zellen nach jeder Generation
    first_solved: object                 # Zahl der Evaluationen bis zum ersten Individuum im Zielkreis (None: nie)
    end_positions: list = field(default_factory=list)    # Endpositionen (pop, 2) je Generation
    last_scores: object = None           # Auswahlwert (Novelty bzw. −Abstand) jedes Individuums der letzten Generation
    visited_grid: object = None          # bool-Raster der besuchten Zellen
    best_genome: object = None           # Steuerfolge mit dem kleinsten Abstand über den ganzen Lauf
    evaluations: int = 0


def next_generation(rng, genomes, score, sigma=C.MUT_SIGMA, rate=C.MUT_RATE):
    """Neue Population: die ELITE höchsten `score` bleiben unverändert, der Rest sind Mutationen von Turnier-Siegern (Turniergröße TOURNAMENT). Größer ist besser."""
    pop = len(genomes)
    order = np.argsort(-score, kind="stable")
    nxt = [genomes[i].copy() for i in order[:C.ELITE]]
    while len(nxt) < pop:
        cand = rng.integers(0, pop, C.TOURNAMENT)
        parent = cand[np.argmax(score[cand])]
        child = genomes[parent] + rng.normal(0.0, sigma, genomes[parent].shape) * (rng.random(genomes[parent].shape) < rate)
        nxt.append(R.clip_genomes(child))
    return np.array(nxt)


def novelty_scores(points, k):
    """Novelty jedes Punkts: mittlerer Abstand zu den `k` nächsten anderen Punkten (euklidisch, beliebige Dimension). Der Punkt selbst zählt nicht, ein anderer Punkt am selben Ort zählt mit Abstand 0;
    gibt es weniger als k andere Punkte, gehen alle ein."""
    if len(points) < 2 or k < 1:
        raise ValueError("Novelty braucht mindestens zwei Punkte und k >= 1")
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    m = min(k + 1, d.shape[1])
    d = np.partition(d, m - 1, axis=1)[:, :m]            # die m kleinsten Abstände je Zeile (statt die ganze Zeile zu sortieren; gleiches Ergebnis)
    d.sort(axis=1)
    return d[:, 1:k + 1].mean(axis=1)


def run_search(grid, seed, selection=C.BASELINE, gens=C.STUDY_GENS, pop=C.POP, k=C.K_DEFAULT, sigma=C.MUT_SIGMA, rate=C.MUT_RATE, keep_positions=True, stop_when_solved=False):
    """Ein Lauf mit der Auswahl `selection`: "fitness" oder eine der Beschreibungen aus C.DESCRIPTORS. Zählt Evaluationen (pop je Generation) bis die erste Endposition im Zielkreis liegt, und verfolgt, wie viele
    Zellen die Suche besucht hat und wie nah dem Ziel (entlang des wahren Wegs) sie gekommen ist."""
    if selection != C.FITNESS and selection not in D.FUNCTIONS:
        raise ValueError(f"unbekannte Auswahl: {selection}")
    rng = np.random.default_rng(seed)
    genomes = R.random_genomes(rng, pop)
    true_path = M.bfs_distances(grid, M.cell_of(C.GOAL))
    visited = np.zeros(grid.shape, bool)
    res = SearchResult(best_distance=[], visited_cells=[], best_progress=[], first_solved=None)
    best_d, best_p = np.inf, np.inf
    track_needed = selection in NEEDS_TRACK
    for gen in range(gens):
        if track_needed:
            end, track = R.simulate(genomes, grid, trajectories=True)
        else:
            end = R.simulate(genomes, grid)
            track = end[:, None, :]
        dist = R.distance_to_goal(end)
        res.evaluations += pop
        if res.first_solved is None and dist.min() < C.SUCCESS_RADIUS:
            res.first_solved = gen * pop + int(np.argmax(dist < C.SUCCESS_RADIUS)) + 1
        idx = np.clip(end.astype(int), 0, grid.shape[0] - 1)
        visited[idx[:, 0], idx[:, 1]] = True
        best_p = min(best_p, float(true_path[idx[:, 0], idx[:, 1]].min()))
        res.best_distance.append(float(dist.min()))
        res.visited_cells.append(int(visited.sum()))
        res.best_progress.append(best_p)
        if keep_positions:
            res.end_positions.append(end.copy())
        if dist.min() < best_d:
            best_d = float(dist.min())
            res.best_genome = genomes[int(np.argmin(dist))].copy()
        score = -dist if selection == C.FITNESS else novelty_scores(D.describe(selection, track, genomes, grid), k)
        res.last_scores = score
        if stop_when_solved and res.first_solved is not None:
            break
        genomes = next_generation(rng, genomes, score, sigma, rate)
    res.visited_grid = visited
    return res
