"""Auswertung: Täuschungsmaße je Stufe, Lösbarkeit, Live-Paar, Hilfsfunktionen der Studie, Vollständigkeit und Stimmigkeit der vorgerechneten Datei (inklusive Neurechnung einzelner Läufe)."""

import numpy as np
import pytest

import bs_constants as C
import bs_evaluation as E
import bs_maze as M
import bs_search as S

PRE = E.load_precomputed()


def test_level_metrics_by_hand():
    m0, m3 = E.level_metrics(0), E.level_metrics(3)
    assert m0["path"] == 17 and m0["straight"] == pytest.approx(17.0) and m0["detour"] == pytest.approx(1.0) and m0["minima"] == 0 and m0["trap"] is None and m0["free"] == 22 * 22
    assert m3["path"] == 49 and m3["detour"] == pytest.approx(49 / 17) and m3["minima"] == 1 and m3["trap"] == pytest.approx(9.0) and m3["free"] == 22 * 22 - 19


@pytest.mark.parametrize("level", list(C.LEVELS))
def test_every_level_is_solvable_within_the_step_budget(level):
    dist, steps = E.solvability(level)
    assert dist < 0.5 and steps <= C.STEPS


def test_live_pair_runs_the_baseline_and_the_chosen_description_from_the_same_start():
    (base, bt), (desc, dt) = E.live_pair(2, 12, 5, "mid", 15)
    assert len(base.best_distance) == len(desc.best_distance) == 12 and bt.shape == dt.shape == (C.STEPS + 1, 2) and tuple(bt[0]) == tuple(dt[0]) == C.START
    assert (base.end_positions[0] == desc.end_positions[0]).all()                                      # gleiche Anfangspopulation
    assert np.hypot(*(dt[-1] - np.array(C.GOAL))) == pytest.approx(min(desc.best_distance)) and np.hypot(*(bt[-1] - np.array(C.GOAL))) == pytest.approx(min(base.best_distance))


def test_live_pair_with_the_baseline_gives_two_identical_runs():
    (a, _), (b, _) = E.live_pair(2, 12, 5, C.BASELINE, 15)
    assert a.best_distance == b.best_distance and a.first_solved == b.first_solved and (a.last_scores == b.last_scores).all()


def test_success_helpers_by_hand():
    runs = [100, None, 500, 50, None]
    assert E.success_count(runs) == 3 and E.success_count(runs, 100) == 2 and E.success_count(runs, 10) == 0
    assert E.median_solved(runs) == 100.0 and E.median_solved([None, None]) is None and E.median_solved([10, 20]) == 15.0


def test_the_study_covers_both_levels_and_every_selection():
    assert set(PRE["levels"]) == {"3", "4"} and PRE["seeds"] == C.STUDY_SEEDS and PRE["budget"] == C.STUDY_BUDGET and E.SELECTIONS == (C.FITNESS,) + C.DESCRIPTORS
    for lv in C.STUDY_LEVELS:
        assert set(PRE["levels"][str(lv)]["selections"]) == set(E.SELECTIONS)
        for s in E.SELECTIONS:
            c = E.study_cell(PRE, lv, s)
            assert len(c["runs"]) == len(c["cells"]) == len(c["progress"]) == C.STUDY_SEEDS and len(c["mean_visited"]) == len(c["mean_progress"]) == C.STUDY_GENS
            assert all(r is None or 1 <= r <= C.STUDY_BUDGET for r in c["runs"]) and max(c["cells"]) <= E.level_info(PRE, lv)["free"]


def test_the_stored_level_metrics_equal_the_exact_ones():
    for lv in C.STUDY_LEVELS:
        stored, fresh = E.level_info(PRE, lv), E.level_metrics(lv)
        for key in ("path", "straight", "detour", "fdc", "minima", "free"):
            assert stored[key] == pytest.approx(fresh[key])
        assert stored["trap"] == pytest.approx(fresh["trap"])


@pytest.mark.parametrize("level,selection,seed", [(3, "fitness", 0), (3, "end", 3), (3, "mid", 2), (4, "way4", 1), (3, "bfs", 5), (3, "path_turn", 7), (4, "end_heading", 4)])
def test_a_study_run_is_reproduced_by_a_fresh_run_with_the_same_seed(level, selection, seed):
    """Gespeicherte Zahl der Evaluationen bis zum Erfolg, besuchte Zellen und Restweg stimmen mit einem frischen Lauf überein."""
    res = S.run_search(M.make_maze(level), seed, selection, gens=C.STUDY_GENS, keep_positions=False)
    c = E.study_cell(PRE, level, selection)
    assert res.first_solved == c["runs"][seed] and res.visited_cells[-1] == c["cells"][seed] and res.best_progress[-1] == c["progress"][seed]


def test_the_mean_curves_never_get_worse():
    c = E.study_cell(PRE, 3, "mid")
    assert 1 <= c["mean_visited"][0] <= C.POP and all(b >= a for a, b in zip(c["mean_visited"], c["mean_visited"][1:])) and all(b <= a + 1e-9 for a, b in zip(c["mean_progress"], c["mean_progress"][1:]))


def test_the_baseline_curves_equal_the_second_piece_start():
    """Die Endposition ist die Beschreibung aus Stück 2: dieselben Läufe, also hier und dort derselbe Median und dieselbe Abdeckung (geprüft in test_claims)."""
    assert E.median_solved(E.study_cell(PRE, 3, "end")["runs"]) == 988.5 and C.BASELINE == "end"


def test_the_length_experiment_covers_every_cell():
    assert PRE["length"]["steps"] == list(C.LENGTH_STEPS) and set(PRE["length"]["cells"]) == set(C.LENGTH_SELECTIONS)
    for sel in C.LENGTH_SELECTIONS:
        for st in C.LENGTH_STEPS:
            c = E.length_cell(PRE, sel, st)
            assert len(c["runs"]) == len(c["cells"]) == len(c["progress"]) == C.STUDY_SEEDS


def test_the_length_experiment_at_140_steps_equals_the_level_study():
    for sel in C.LENGTH_SELECTIONS:
        assert E.length_cell(PRE, sel, 140)["runs"] == E.study_cell(PRE, 4, sel)["runs"] and E.length_cell(PRE, sel, 140)["progress"] == E.study_cell(PRE, 4, sel)["progress"]


def test_a_length_run_is_reproduced_by_a_fresh_run_with_the_same_seed(monkeypatch):
    """Stufe 4, 300 Schritte, Endposition, Seed 2: dieselben Endwerte wie gespeichert."""
    monkeypatch.setattr(C, "STEPS", 300)
    res = S.run_search(M.make_maze(4), 2, C.BASELINE, gens=C.STUDY_GENS, keep_positions=False)
    c = E.length_cell(PRE, C.BASELINE, 300)
    assert res.first_solved == c["runs"][2] and res.visited_cells[-1] == c["cells"][2] and res.best_progress[-1] == c["progress"][2]
