"""Suche: Novelty von Hand, Auswahl nach Novelty statt Fitness, Reproduzierbarkeit je Auswahl, Zählungen (Evaluationen, besuchte Zellen, Restweg), wer das Ziel sieht."""

import numpy as np
import pytest

import bs_constants as C
import bs_maze as M
import bs_robot as R
import bs_search as S

LINE = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
SELECTIONS = (C.FITNESS,) + C.DESCRIPTORS


def test_novelty_with_one_neighbour_is_the_distance_to_the_nearest_other_point():
    assert S.novelty_scores(LINE, 1) == pytest.approx([1.0, 1.0, 2.0, 7.0])


def test_novelty_with_two_neighbours_is_the_mean_of_the_two_nearest_distances():
    assert S.novelty_scores(LINE, 2) == pytest.approx([2.0, 1.5, 2.5, 8.0])               # Punkt 0: 1 und 3; Punkt 1: 1 und 2; Punkt 2: 2 und 3; Punkt 3: 7 und 9


def test_the_point_itself_does_not_count_but_a_duplicate_does():
    assert S.novelty_scores(np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 0.0]]), 1) == pytest.approx([0.0, 0.0, 5.0])


def test_k_larger_than_the_population_uses_all_other_points():
    assert S.novelty_scores(LINE, 30) == pytest.approx([(1 + 3 + 10) / 3, (1 + 2 + 9) / 3, (3 + 2 + 7) / 3, (10 + 9 + 7) / 3])


def test_novelty_works_in_higher_dimensions_by_the_euclidean_distance():
    pts = np.array([[0.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 12.0, 0.0]])             # Abstände 5, 12, 13
    assert S.novelty_scores(pts, 1) == pytest.approx([5.0, 5.0, 12.0])


def test_novelty_needs_at_least_two_points_and_k_one():
    with pytest.raises(ValueError):
        S.novelty_scores(np.array([[1.0, 1.0]]), 1)
    with pytest.raises(ValueError):
        S.novelty_scores(LINE, 0)


def test_next_generation_keeps_the_elite_unchanged_and_keeps_the_size():
    genomes = R.random_genomes(np.random.default_rng(1), 10)
    score = np.array([0.0, 5.0, 1.0, 9.0, 2.0, 3.0, 4.0, 8.0, 7.0, 6.0])
    nxt = S.next_generation(np.random.default_rng(2), genomes, score)
    assert nxt.shape == genomes.shape and (nxt[0] == genomes[3]).all() and (nxt[1] == genomes[7]).all()


def test_without_mutation_every_child_is_a_copy_of_a_parent():
    genomes = R.random_genomes(np.random.default_rng(1), 8)
    nxt = S.next_generation(np.random.default_rng(2), genomes, np.arange(8, dtype=float), rate=0.0)
    assert all(any((child == parent).all() for parent in genomes) for child in nxt)


def test_the_tournament_prefers_higher_scores():
    genomes = R.random_genomes(np.random.default_rng(5), 60)
    nxt = S.next_generation(np.random.default_rng(6), genomes, np.arange(60, dtype=float), rate=0.0)
    parents = [next(i for i in range(60) if (child == genomes[i]).all()) for child in nxt[C.ELITE:]]
    assert len(parents) == 58 and np.mean(parents) > 36


def test_unknown_selection_is_refused():
    with pytest.raises(ValueError):
        S.run_search(M.make_maze(0), 1, "zufall", gens=2)


@pytest.mark.parametrize("selection", SELECTIONS)
def test_runs_are_reproducible_and_depend_on_the_seed(selection):
    grid = M.make_maze(2)
    a, b, c = (S.run_search(grid, s, selection, gens=12, keep_positions=False) for s in (4, 4, 5))
    assert a.best_distance == b.best_distance and a.first_solved == b.first_solved and a.visited_cells == b.visited_cells and a.best_distance != c.best_distance


@pytest.mark.parametrize("selection", SELECTIONS)
def test_counts_and_series_have_the_right_shape(selection):
    res = S.run_search(M.make_maze(3), 3, selection, gens=25, keep_positions=True)
    assert res.evaluations == 25 * C.POP and len(res.best_distance) == len(res.visited_cells) == len(res.best_progress) == len(res.end_positions) == 25
    assert all(b >= a for a, b in zip(res.visited_cells, res.visited_cells[1:])) and res.visited_cells[-1] == int(res.visited_grid.sum())
    assert all(b <= a for a, b in zip(res.best_progress, res.best_progress[1:])) and res.end_positions[0].shape == (C.POP, 2) and res.last_scores.shape == (C.POP,)


def test_the_track_is_only_simulated_for_descriptions_that_need_it():
    """Die Beschreibungen ohne Fahrt-Bedarf liefern dieselben Läufe, ob man die Fahrt mitrechnet oder nicht (Gegenprobe: gleiche Endpositionen wie die Fahrt-Rechnung)."""
    grid = M.make_maze(3)
    for name in C.DESCRIPTORS:
        assert (name in S.NEEDS_TRACK) == (name in ("mid", "way4", "way8", "path_turn"))
    genomes = R.random_genomes(np.random.default_rng(3), 20)
    end = R.simulate(genomes, grid)
    end2, _ = R.simulate(genomes, grid, trajectories=True)
    assert (end == end2).all()


def test_the_fitness_selection_never_gets_worse_because_the_elite_survives():
    res = S.run_search(M.make_maze(3), 3, "fitness", gens=40, keep_positions=False)
    assert all(b <= a + 1e-12 for a, b in zip(res.best_distance, res.best_distance[1:]))


def test_the_baseline_equals_the_second_piece():
    """Beschreibung „Endposition“ gleich Novelty ohne Archiv aus Stück 2 (novelty-search-demo): dieselben Evaluationen bis zum Erfolg auf Stufe 3, Seeds 0 bis 5."""
    runs = [S.run_search(M.make_maze(3), s, "end", keep_positions=False).first_solved for s in range(6)]
    assert runs == [1246, 1737, 515, 631, 2035, 1641]


def test_only_the_fitness_and_the_knowledge_descriptions_see_the_goal():
    """Mit verschobenem Ziel bleiben alle Läufe gleich, außer Fitness, `bfs` und `euclid` (und die Messung des Erfolgs)."""
    grid = M.make_maze(3)

    def positions(selection):
        return S.run_search(grid, 2, selection, gens=12, keep_positions=True).end_positions

    base = {s: positions(s) for s in SELECTIONS}
    old_goal = C.GOAL
    try:
        C.GOAL = (20.5, 20.5)
        moved = {s: positions(s) for s in SELECTIONS}
    finally:
        C.GOAL = old_goal
    for s in SELECTIONS:
        same = all((a == b).all() for a, b in zip(base[s], moved[s]))
        assert same == (s not in (C.FITNESS, "bfs", "euclid")), s


def test_first_solved_counts_evaluations_by_hand_on_the_free_field():
    res = S.run_search(M.make_maze(0), 3, "end", gens=30, keep_positions=True)
    for gen, end in enumerate(res.end_positions):
        hit = np.flatnonzero(R.distance_to_goal(end) < C.SUCCESS_RADIUS)
        if len(hit):
            assert res.first_solved == gen * C.POP + int(hit[0]) + 1
            break
    else:
        assert res.first_solved is None


def test_stop_when_solved_ends_the_run_early():
    res = S.run_search(M.make_maze(0), 3, "end", gens=300, keep_positions=False, stop_when_solved=True)
    assert res.first_solved is not None and len(res.best_distance) < 300 and len(res.best_distance) == (res.first_solved - 1) // C.POP + 1
