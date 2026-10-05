"""JEDE Zahl aus README und App-Texten wird hier nachgerechnet: Täuschungsmaße exakt aus dem Labyrinth, Studienzahlen aus der vorgerechneten Datei (20 Läufe je Zelle, 40 000 Evaluationen, Seeds 0 bis 19; Genomlängen-Experiment
mit je 400 Generationen). Die Datei ist fest; ändern sich die Zahlen nach einer neuen Rechnung, müssen README und Hilfetexte nachgezogen werden. Längen in Zellen, Aufwand in Evaluationen."""

import numpy as np
import pytest

import bs_constants as C
import bs_evaluation as E

PRE = E.load_precomputed()
D = C.DESCRIPTORS


def cell(level, sel):
    return E.study_cell(PRE, level, sel)


def runs(level, sel):
    return cell(level, sel)["runs"]


def mean(level, sel, key):
    return float(np.mean(cell(level, sel)[key]))


def test_exact_deception_measures_of_the_levels_as_in_the_first_pieces():
    """README: kürzester Weg 49 / 77 Zellen (Stufe 3 / 4), Umwegfaktor 2.88 / 4.53, Korrelation 0.69 / 0.68, Fallen 1 / 2, freie Zellen 465 / 427."""
    m3, m4 = E.level_metrics(3), E.level_metrics(4)
    assert (m3["path"], m4["path"]) == (49, 77) and (round(m3["detour"], 2), round(m4["detour"], 2)) == (2.88, 4.53)
    assert (round(m3["fdc"], 2), round(m4["fdc"], 2)) == (0.69, 0.68) and (m3["minima"], m4["minima"]) == (1, 2) and (m3["free"], m4["free"]) == (465, 427)


def test_the_fitness_reference_equals_the_first_pieces():
    """README: die Fitness löst Stufe 3 in 11 von 20 Läufen (Median 1 267, schlimmster Lauf 3 355; besucht im Mittel 343.9 von 465 Zellen, kleinster Restweg 5.55, schlechtester Lauf 17) und Stufe 4 in 0 von 20 (Restweg 34.75)."""
    assert E.success_count(runs(3, "fitness")) == 11 and E.median_solved(runs(3, "fitness")) == 1267.0 and max(r for r in runs(3, "fitness") if r is not None) == 3355
    assert round(mean(3, "fitness", "cells"), 2) == 343.9 and round(mean(3, "fitness", "progress"), 2) == 5.55 and max(cell(3, "fitness")["progress"]) == 17
    assert E.success_count(runs(4, "fitness")) == 0 and round(mean(4, "fitness", "progress"), 2) == 34.75


def test_the_baseline_equals_the_second_piece():
    """README: die Endposition löst Stufe 3 in 20 von 20 Läufen, Median 988.5 (schnellster 77, langsamster 3 373), alle 465 Zellen in jedem Lauf besucht; Stufe 4: 0 von 20, Restweg 26.25, besuchte Zellen 264.15 von 427."""
    ok = [r for r in runs(3, "end") if r is not None]
    assert len(ok) == 20 and E.median_solved(runs(3, "end")) == 988.5 and (min(ok), max(ok)) == (77, 3373) and cell(3, "end")["cells"] == [465] * 20
    assert E.success_count(runs(4, "end")) == 0 and round(mean(4, "end", "progress"), 2) == 26.25 and round(mean(4, "end", "cells"), 2) == 264.15


def test_success_and_median_on_level_three_for_all_twelve_descriptions():
    """README: Erfolge von 20 / Median auf Stufe 3: Endposition 20 / 988.5, Endposition und Endrichtung 20 / 801.5, grobe Endposition 20 / 1 077, Position nach halber Fahrt 20 / 4 478, 4 Wegpunkte 20 / 1 087.5, 8 Wegpunkte 20 / 1 488,
    nur x 17 / 16 498, nur y 20 / 1 401.5, Weglänge und Drehung 19 / 4 040, Mittelwerte der Gene 18 / 9 579.5, wahrer Restweg 20 / 790, Luftlinie 20 / 1 717.5."""
    expect = {"end": (20, 988.5), "end_heading": (20, 801.5), "coarse": (20, 1077.0), "mid": (20, 4478.0), "way4": (20, 1087.5), "way8": (20, 1488.0), "x": (17, 16498.0), "y": (20, 1401.5),
              "path_turn": (19, 4040.0), "gene_means": (18, 9579.5), "bfs": (20, 790.0), "euclid": (20, 1717.5)}
    assert set(expect) == set(D)
    for d, (n, med) in expect.items():
        assert (E.success_count(runs(3, d)), E.median_solved(runs(3, d))) == (n, med), d


def test_the_groups_on_level_three():
    """README: 9 der 12 Beschreibungen lösen alle 20 Läufe: sieben der acht mit Ortsbezug (nicht: nur x) und beide mit Aufgabenwissen; keine ohne Ortsbezug. Der Median der neun reicht von 790 bis 4 478, das ist der Faktor 5.7;
    unter den ortsbezogenen von 801.5 bis 4 478."""
    full = [d for d in D if E.success_count(runs(3, d)) == 20]
    assert len(full) == 9 and set(D) - set(full) == {"x", "path_turn", "gene_means"}
    assert [d for d in full if C.DESCRIPTOR_GROUP[d] == "space"] == ["end", "end_heading", "coarse", "mid", "way4", "way8", "y"] and {"bfs", "euclid"} <= set(full)
    meds = [E.median_solved(runs(3, d)) for d in full]
    assert (min(meds), max(meds)) == (790.0, 4478.0) and round(max(meds) / min(meds), 1) == 5.7
    space = [E.median_solved(runs(3, d)) for d in full if C.DESCRIPTOR_GROUP[d] == "space"]
    assert (min(space), max(space)) == (801.5, 4478.0)


def test_the_slow_and_the_failing_descriptions_in_detail():
    """README: nur x: schlimmster Lauf 39 751, besuchte Zellen 401.9, Restweg 1.65 (schlechtester Lauf 8); Mittelwerte der Gene: 385.6, 1.6 (schlechtester 3), schlimmster Lauf 37 373; Weglänge und Drehung: 441.05, 0.65 (4), schlimmster Lauf 36 919;
    Position nach halber Fahrt: schlimmster Lauf 14 233, 462.85 Zellen."""
    assert max(r for r in runs(3, "x") if r is not None) == 39751 and round(mean(3, "x", "cells"), 2) == 401.9 and round(mean(3, "x", "progress"), 2) == 1.65 and max(cell(3, "x")["progress"]) == 8
    assert max(r for r in runs(3, "gene_means") if r is not None) == 37373 and round(mean(3, "gene_means", "cells"), 2) == 385.6 and round(mean(3, "gene_means", "progress"), 2) == 1.6 and max(cell(3, "gene_means")["progress"]) == 3
    assert max(r for r in runs(3, "path_turn") if r is not None) == 36919 and round(mean(3, "path_turn", "cells"), 2) == 441.05 and round(mean(3, "path_turn", "progress"), 2) == 0.65 and max(cell(3, "path_turn")["progress"]) == 4
    assert max(r for r in runs(3, "mid") if r is not None) == 14233 and round(mean(3, "mid", "cells"), 2) == 462.85


def test_visited_cells_on_level_three():
    """README: alle 465 Zellen in jedem Lauf: Endposition, Endposition und Endrichtung, grobe Endposition, wahrer Restweg, Luftlinie; im Mittel 464.95 (4 und 8 Wegpunkte), 463.95 (nur y)."""
    for d in ("end", "end_heading", "coarse", "bfs", "euclid"):
        assert cell(3, d)["cells"] == [465] * 20, d
    assert round(mean(3, "way4", "cells"), 2) == round(mean(3, "way8", "cells"), 2) == 464.95 and round(mean(3, "y", "cells"), 2) == 463.95


def test_level_four_is_solved_by_no_description():
    """README: Stufe 4: 0 von 20 Läufen für jede der 12 Beschreibungen und die Fitness, also 0 von 240 (Beschreibungen) beziehungsweise 0 von 260."""
    assert all(E.success_count(runs(4, s)) == 0 for s in E.SELECTIONS) and sum(len(runs(4, d)) for d in D) == 240 and sum(len(runs(4, s)) for s in E.SELECTIONS) == 260


def test_level_four_progress_per_description():
    """README: kleinster wahrer Restweg im Mittel (bester Lauf) auf Stufe 4: Endposition 26.25 (22), Endposition und Endrichtung 27.2 (25), grobe Endposition 27.1 (23), halbe Fahrt 29.0 (23), 4 Wegpunkte 25.6 (20),
    8 Wegpunkte 27.1 (20), nur x 37.6 (31), nur y 29.0 (23), Weglänge und Drehung 34.3 (21), Gene 38.7 (33), wahrer Restweg 24.75 (19), Luftlinie 27.85 (25); Fitness 34.75 (22)."""
    expect = {"end": (26.25, 22), "end_heading": (27.2, 25), "coarse": (27.1, 23), "mid": (29.0, 23), "way4": (25.6, 20), "way8": (27.1, 20), "x": (37.6, 31), "y": (29.0, 23),
              "path_turn": (34.3, 21), "gene_means": (38.7, 33), "bfs": (24.75, 19), "euclid": (27.85, 25), "fitness": (34.75, 22)}
    for s, (m, best) in expect.items():
        assert (round(mean(4, s, "progress"), 2), min(cell(4, s)["progress"])) == (m, best), s


def test_the_knowledge_does_not_solve_level_four_and_helps_only_a_little_on_level_three():
    """README: mit dem wahren Restweg als Beschreibung bleibt Stufe 4 ungelöst (Restweg 24.75 gegen 26.25 bei der Endposition, besuchte Zellen 283.3 gegen 264.15); auf Stufe 3 ist der Median 790 gegen 988.5, also das 1.25-fache."""
    assert round(mean(4, "bfs", "cells"), 2) == 283.3 and round(mean(4, "end", "cells"), 2) == 264.15
    assert round(E.median_solved(runs(3, "end")) / E.median_solved(runs(3, "bfs")), 2) == 1.25


def test_the_genome_length_experiment():
    """README: Stufe 4, Restweg im Mittel bei 140 / 180 / 220 / 300 Schritten: Fitness 34.75 / 25.05 / 19.95 / 17.8, Novelty (Endposition) 26.25 / 21.25 / 15.65 / 9.05; Erfolge Fitness 0 / 0 / 0 / 0, Novelty 0 / 0 / 0 / 2 (Median 30 402,
    bester Restweg im besten Lauf 2); besuchte Zellen bei 300 Schritten: Fitness 315.95, Novelty 348.95."""
    steps = PRE["length"]["steps"]
    assert steps == [140, 180, 220, 300]
    fit = [round(float(np.mean(E.length_cell(PRE, "fitness", s)["progress"])), 2) for s in steps]
    end = [round(float(np.mean(E.length_cell(PRE, "end", s)["progress"])), 2) for s in steps]
    assert fit == [34.75, 25.05, 19.95, 17.8] and end == [26.25, 21.25, 15.65, 9.05]
    assert [E.success_count(E.length_cell(PRE, "fitness", s)["runs"]) for s in steps] == [0, 0, 0, 0]
    assert [E.success_count(E.length_cell(PRE, "end", s)["runs"]) for s in steps] == [0, 0, 0, 2]
    assert E.median_solved(E.length_cell(PRE, "end", 300)["runs"]) == 30402.0 and min(E.length_cell(PRE, "end", 300)["progress"]) == 2
    assert round(float(np.mean(E.length_cell(PRE, "fitness", 300)["cells"])), 2) == 315.95 and round(float(np.mean(E.length_cell(PRE, "end", 300)["cells"])), 2) == 348.95


def test_the_two_runs_that_solve_level_four_with_300_steps():
    """README: bei 300 Schritten lösen zwei Läufe Stufe 4 mit der Endposition; das ist eine Stichprobe von 20, also 10 % mit einer Standardabweichung von etwa 7 Prozentpunkten."""
    ok = [r for r in E.length_cell(PRE, "end", 300)["runs"] if r is not None]
    assert len(ok) == 2 and all(r > 20_000 for r in ok)
    assert round(100 * (0.1 * 0.9 / 20) ** 0.5) == 7


def test_preset_help_quotes_the_study():
    """PRESET_HELP: halbe Fahrt (20 von 20, Median 4 478 gegen 988.5, Fitness 11 von 20); Gene (18 von 20, Median 9 579.5, 385.6 von 465 Zellen); wahrer Restweg (20 von 20, Median 790 als kleinster Median, Stufe 4 0 von 20, Restweg 24.75);
    Schlangenlinie (0 von 240 Läufen, 300 Schritte: 2 von 20)."""
    assert (E.success_count(runs(3, "mid")), E.median_solved(runs(3, "mid")), E.median_solved(runs(3, "end")), E.success_count(runs(3, "fitness"))) == (20, 4478.0, 988.5, 11)
    assert (E.success_count(runs(3, "gene_means")), E.median_solved(runs(3, "gene_means")), round(mean(3, "gene_means", "cells"), 1)) == (18, 9579.5, 385.6)
    assert E.median_solved(runs(3, "bfs")) == min(E.median_solved(runs(3, d)) for d in D if E.success_count(runs(3, d)) == 20) == 790.0
    assert E.success_count(runs(4, "bfs")) == 0 and round(mean(4, "bfs", "progress"), 2) == 24.75
    assert sum(E.success_count(runs(4, d)) for d in D) == 0 and sum(len(runs(4, d)) for d in D) == 240 and E.success_count(E.length_cell(PRE, "end", 300)["runs"]) == 2
    assert C.STUDY_BUDGET == 40_000
    for name, text in C.PRESET_HELP.items():
        assert text and name in C.PRESETS
