"""Abbildungen: gesperrte Achsen, Spuren, Beschriftungen, Farben der Gruppen."""

import numpy as np

import bs_constants as C
import bs_evaluation as E
import bs_maze as M
import bs_visualization as V

PRE = E.load_precomputed()


def _locked(fig):
    return all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())


def test_all_charts_lock_their_axes():
    grid = M.make_maze(3)
    (base, bt), (desc, dt) = E.live_pair(3, 10, 1, "way4", 15)
    figs = [V.build_run_figure(grid, base.end_positions[-1], bt, base.visited_grid, base.last_scores), V.build_run_figure(grid, desc.end_positions[-1], dt, desc.visited_grid, desc.last_scores),
            V.build_progress_chart(base.best_distance, desc.best_distance, "way4", 9.0), V.build_progress_chart(base.best_distance, desc.best_distance, C.BASELINE, None),
            V.build_coverage_chart(base.visited_cells, desc.visited_cells, "way4", 465), V.build_median_chart(PRE, 3), V.build_progress_bar_chart(PRE, 3), V.build_progress_bar_chart(PRE, 4), V.build_length_chart(PRE)]
    assert all(_locked(f) for f in figs)


def test_run_figure_colors_the_end_positions_by_their_novelty():
    grid = M.make_maze(3)
    (base, bt), _ = E.live_pair(3, 10, 1, "mid", 15)
    fig = V.build_run_figure(grid, base.end_positions[-1], bt, base.visited_grid, base.last_scores)
    pts = next(t for t in fig.data if (t.name or "").startswith("Endpositionen"))
    assert list(pts.marker.color) == list(base.last_scores) and len(pts.x) == C.POP and pts.marker.showscale


def test_run_figure_marks_walls_and_visited_cells_with_different_values():
    grid = M.make_maze(3)
    (base, bt), _ = E.live_pair(3, 10, 1, "mid", 15)
    z = np.array(V.build_run_figure(grid, base.end_positions[-1], bt, base.visited_grid, base.last_scores).data[0].z).T
    assert (z[grid] == 2.0).all() and (z[base.visited_grid & ~grid] == 1.0).all() and np.isnan(z[~grid & ~base.visited_grid]).all()


def test_the_baseline_curve_is_dashed_when_the_baseline_itself_is_chosen():
    fig = V.build_progress_chart([5.0, 4.0], [5.0, 4.0], C.BASELINE, None)
    assert fig.data[1].line.dash == "dash" and V.build_progress_chart([5.0, 4.0], [6.0, 3.0], "mid", None).data[1].line.dash == "solid"


def test_median_chart_has_one_bar_per_selection_in_the_group_colors():
    fig = V.build_median_chart(PRE, 3)
    bars = fig.data[0]
    assert len(bars.y) == 13 and bars.y[0] == "Fitness (Bezug)" and bars.marker.color[0] == V.FITNESS_COLOR
    assert [bars.marker.color[1 + i] for i in range(12)] == [V.GROUP_COLORS[C.DESCRIPTOR_GROUP[d]] for d in C.DESCRIPTORS]
    assert "11 von 20" in bars.text[0] and "20 von 20 · Median 988.5" in bars.text[1]


def test_median_chart_on_level_four_has_no_bar_length_and_says_zero_of_twenty():
    bars = V.build_median_chart(PRE, 4).data[0]
    assert all(x == 0 for x in bars.x) and all(t == "0 von 20" for t in bars.text)


def test_progress_bar_chart_reports_the_mean_remaining_path():
    bars = V.build_progress_bar_chart(PRE, 4).data[0]
    assert bars.text[C.DESCRIPTORS.index("end") + 1] == "26.25"
    assert bars.text[0] == "34.75"


def test_length_chart_has_a_line_for_the_fitness_and_one_for_novelty():
    fig = V.build_length_chart(PRE)
    assert [t.name for t in fig.data] == ["Fitness (Bezug)", "Endposition (Bezug)"] and list(fig.data[0].x) == list(C.LENGTH_STEPS)
    assert fig.data[1].text[-1] == "2 von 20" and fig.data[0].text[-1] == "0 von 20"
