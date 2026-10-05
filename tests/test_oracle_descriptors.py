"""Orakel-Tests: die Beschreibungen gegen eine skalare Neuberechnung der Fahrt nur aus der Beschreibung der Kinematik (eine Schleife je Individuum und Schritt, `math` statt NumPy), und die Novelty gegen SciPys `cKDTree`
und eine skalare Definition (mehrere Dimensionen, k, doppelte Punkte)."""

import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

import bs_constants as C
import bs_descriptors as D
import bs_maze as M
import bs_robot as R
import bs_search as S
from test_oracle_maze import scipy_distances


def scalar_track(genome, grid):
    """Alle Positionen der Fahrt (Schritt 0 bis STEPS), skalar: Drehung, Fahrt, Stillstand an belegten Zellen und am Rand."""
    n = grid.shape[0]
    x, y = C.START
    heading = C.START_HEADING
    out = [(x, y)]
    for turn, speed in genome:
        heading += turn * C.TURN_MAX
        nx = x + speed * C.SPEED_MAX * math.cos(heading)
        ny = y + speed * C.SPEED_MAX * math.sin(heading)
        if 0 <= nx < n and 0 <= ny < n and not grid[int(nx), int(ny)]:
            x, y = nx, ny
        out.append((x, y))
    return out


def scalar_describe(name, genome, grid):
    pts = scalar_track(genome, grid)
    ex, ey = pts[-1]
    steps = len(genome)
    if name == "end":
        return [ex, ey]
    if name == "end_heading":
        h = C.START_HEADING + sum(t * C.TURN_MAX for t, _ in genome)
        return [ex, ey, 3 * math.cos(h), 3 * math.sin(h)]
    if name == "coarse":
        return [(math.floor(ex / 4) + 0.5) * 4, (math.floor(ey / 4) + 0.5) * 4]
    if name == "mid":
        return list(pts[steps // 2])
    if name in ("way4", "way8"):
        parts = 4 if name == "way4" else 8
        return [c for i in range(1, parts + 1) for c in pts[int(steps * i / parts)]]
    if name == "x":
        return [ex]
    if name == "y":
        return [ey]
    if name == "path_turn":
        return [sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])), 0.2 * sum(abs(t) for t, _ in genome)]
    if name == "gene_means":
        return [20 * sum(t for t, _ in genome) / steps, 20 * sum(s for _, s in genome) / steps]
    if name == "bfs":
        return [float(scipy_distances(grid, M.cell_of(C.GOAL))[int(ex), int(ey)])]
    if name == "euclid":
        return [math.hypot(ex - C.GOAL[0], ey - C.GOAL[1])]
    raise ValueError(name)


@pytest.mark.parametrize("level", [0, 3, 4])
@pytest.mark.parametrize("name", C.DESCRIPTORS)
def test_every_description_agrees_with_the_scalar_reference(name, level):
    grid = M.make_maze(level)
    genomes = R.random_genomes(np.random.default_rng(200 + level), 12)
    _, track = R.simulate(genomes, grid, trajectories=True)
    fast = D.describe(name, track, genomes, grid)
    slow = np.array([scalar_describe(name, g, grid) for g in genomes])
    assert fast == pytest.approx(slow, abs=1e-8)


def kdtree_novelty(points, k):
    dist, _ = cKDTree(points).query(points, k=min(k + 1, len(points)))
    return dist[:, 1:].mean(axis=1)


def scalar_novelty(points, k):
    out = []
    for i, p in enumerate(points):
        near = sorted(math.dist(p, q) for j, q in enumerate(points) if j != i)[:k]
        out.append(sum(near) / len(near))
    return np.array(out)


@pytest.mark.parametrize("dim", [1, 2, 4, 16])
@pytest.mark.parametrize("k", [1, 3, 15, 30])
def test_novelty_agrees_with_a_kd_tree_and_with_the_scalar_definition(dim, k):
    pts = np.random.default_rng(dim * 100 + k).uniform(0, 24, (40, dim))
    mine = S.novelty_scores(pts, k)
    assert mine == pytest.approx(kdtree_novelty(pts, k), abs=1e-10) and mine == pytest.approx(scalar_novelty(pts, k), abs=1e-10)


def test_novelty_with_many_duplicate_points_agrees_too():
    """Gestrandete Roboter enden auf demselben Punkt; solche Gleichstände sind der Normalfall an Wänden."""
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 24, (6, 2))[rng.integers(0, 6, 50)]
    for k in (1, 5, 15):
        assert S.novelty_scores(pts, k) == pytest.approx(kdtree_novelty(pts, k), abs=1e-12) and S.novelty_scores(pts, k) == pytest.approx(scalar_novelty(pts, k), abs=1e-12)


def test_visited_cells_agree_with_a_set_of_cell_tuples_over_the_stored_end_positions():
    grid = M.make_maze(3)
    res = S.run_search(grid, 5, "mid", gens=30, keep_positions=True)
    seen, counts = set(), []
    for end in res.end_positions:
        seen.update((int(x), int(y)) for x, y in end)
        counts.append(len(seen))
    assert res.visited_cells == counts and res.visited_cells[-1] == int(res.visited_grid.sum()) and all(not grid[c] for c in seen)


def test_best_progress_agrees_with_scipys_breadth_first_search():
    grid = M.make_maze(4)
    res = S.run_search(grid, 2, "fitness", gens=30, keep_positions=True)
    ref = scipy_distances(grid, M.cell_of(C.GOAL))
    best, series = np.inf, []
    for end in res.end_positions:
        best = min(best, min(ref[int(x), int(y)] for x, y in end))
        series.append(best)
    assert res.best_progress == pytest.approx(series, abs=0)
