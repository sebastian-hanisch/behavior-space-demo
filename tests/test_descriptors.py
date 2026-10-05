"""Verhaltensbeschreibungen von Hand: eine Fahrt mit bekannten Werten (Tempo 0.1 geradeaus: Weg 9.8 Zellen, nach t Schritten genau t · 0.07 Zellen in Richtung π/4), Stillstand, reine Drehung, Dimensionen, Gruppen."""

import math

import numpy as np
import pytest

import bs_constants as C
import bs_descriptors as D
import bs_maze as M
import bs_robot as R

FREE = M.make_maze(0)
COS = math.cos(C.START_HEADING)                      # = sin(π/4)
STEP = 0.1 * C.SPEED_MAX                             # 0.07 Zellen je Schritt


def _run(genome, grid=FREE):
    genomes = np.asarray(genome, float)[None]
    end, track = R.simulate(genomes, grid, trajectories=True)
    return track, genomes, grid


def slow_straight():
    g = np.zeros((C.STEPS, 2))
    g[:, 1] = 0.1
    return _run(g)


def standing():
    return _run(np.zeros((C.STEPS, 2)))


def test_the_slow_straight_ride_ends_where_the_arithmetic_says():
    track, g, grid = slow_straight()
    expected = C.START[0] + C.STEPS * STEP * COS
    assert tuple(track[0, -1]) == pytest.approx((expected, C.START[1] + C.STEPS * STEP * COS), abs=1e-9)


def test_end():
    track, g, grid = slow_straight()
    assert D.describe("end", track, g, grid)[0] == pytest.approx(track[0, -1])


def test_end_heading_adds_the_final_direction_scaled_by_three():
    track, g, grid = slow_straight()
    out = D.describe("end_heading", track, g, grid)[0]
    assert out[2:] == pytest.approx([3 * COS, 3 * COS]) and out[:2] == pytest.approx(track[0, -1])
    turn = np.zeros((C.STEPS, 2))
    turn[:, 0] = 0.5                                                           # Drehgen 0.5 mal 140 Schritte mal 0.6 rad
    t2, g2, gr2 = _run(turn)
    heading = C.START_HEADING + C.STEPS * 0.5 * C.TURN_MAX
    out2 = D.describe("end_heading", t2, g2, gr2)[0]
    assert out2[2:] == pytest.approx([3 * math.cos(heading), 3 * math.sin(heading)]) and out2[:2] == pytest.approx(C.START)


def test_coarse_rounds_the_end_position_to_the_middle_of_its_four_by_four_block():
    track, g, grid = slow_straight()
    x = C.START[0] + C.STEPS * STEP * COS                                     # 10.43...
    assert D.describe("coarse", track, g, grid)[0] == pytest.approx([10.0, 10.0]) and 8 <= x < 12


def test_mid_is_the_position_after_half_the_ride():
    track, g, grid = slow_straight()
    half = C.START[0] + (C.STEPS // 2) * STEP * COS
    assert D.describe("mid", track, g, grid)[0] == pytest.approx([half, half - C.START[0] + C.START[1]])


def test_waypoints_are_the_positions_after_equal_parts_of_the_ride():
    track, g, grid = slow_straight()
    w4 = D.describe("way4", track, g, grid)[0].reshape(4, 2)
    w8 = D.describe("way8", track, g, grid)[0].reshape(8, 2)
    steps4 = [35, 70, 105, 140]
    steps8 = [17, 35, 52, 70, 87, 105, 122, 140]                               # abgerundet: 17.5 -> 17, 52.5 -> 52, ...
    assert w4[:, 0] == pytest.approx([C.START[0] + t * STEP * COS for t in steps4]) and w8[:, 0] == pytest.approx([C.START[0] + t * STEP * COS for t in steps8])
    assert w4[-1] == pytest.approx(track[0, -1]) and w8[-1] == pytest.approx(track[0, -1])
    assert D.describe("way4", track, g, grid).shape == (1, 8) and D.describe("way8", track, g, grid).shape == (1, 16)


def test_x_and_y_are_the_single_coordinates():
    track, g, grid = slow_straight()
    assert D.describe("x", track, g, grid)[0] == pytest.approx([track[0, -1, 0]]) and D.describe("y", track, g, grid)[0] == pytest.approx([track[0, -1, 1]])


def test_path_turn_is_the_driven_length_and_a_fifth_of_the_summed_turning():
    track, g, grid = slow_straight()
    assert D.describe("path_turn", track, g, grid)[0] == pytest.approx([C.STEPS * STEP, 0.0])
    spin = np.zeros((C.STEPS, 2))
    spin[:, 0] = -1.0                                                          # Dauerdrehung, kein Tempo
    t2, g2, gr2 = _run(spin)
    assert D.describe("path_turn", t2, g2, gr2)[0] == pytest.approx([0.0, 0.2 * C.STEPS])


def test_gene_means_scale_the_two_gene_means_by_twenty():
    g = np.zeros((C.STEPS, 2))
    g[:, 0], g[:, 1] = 0.25, 0.5
    track, genomes, grid = _run(g)
    assert D.describe("gene_means", track, genomes, grid)[0] == pytest.approx([5.0, 10.0])


def test_bfs_is_the_true_way_to_the_goal_from_the_end_cell():
    track, g, grid = slow_straight()                                          # Endzelle (10, 10) im freien Feld: |10 - 3| + |10 - 20| = 17
    assert D.describe("bfs", track, g, grid)[0] == pytest.approx([17.0])
    wall = M.make_maze(3)
    below = np.array([[[3.5, 11.5]] * 2])                                      # unter der Wand links: 16 zur Öffnung, 2 hindurch, 23 zum Ziel
    assert D.describe("bfs", below, np.zeros((1, 2, 2)), wall)[0] == pytest.approx([41.0])
    above = np.array([[[3.5, 13.5]] * 2])
    assert D.describe("bfs", above, np.zeros((1, 2, 2)), wall)[0] == pytest.approx([7.0])


def test_euclid_is_the_straight_line_to_the_goal():
    track, g, grid = slow_straight()
    e = track[0, -1]
    assert D.describe("euclid", track, g, grid)[0] == pytest.approx([math.hypot(e[0] - C.GOAL[0], e[1] - C.GOAL[1])])


def test_standing_still_gives_the_start_everywhere():
    track, g, grid = standing()
    for name in ("end", "mid", "coarse"):
        assert D.describe(name, track, g, grid)[0] == pytest.approx([3.5, 3.5] if name != "coarse" else [2.0, 2.0])
    assert D.describe("way4", track, g, grid)[0] == pytest.approx([3.5] * 8) and D.describe("way8", track, g, grid)[0] == pytest.approx([3.5] * 16)
    assert D.describe("path_turn", track, g, grid)[0] == pytest.approx([0.0, 0.0]) and D.describe("gene_means", track, g, grid)[0] == pytest.approx([0.0, 0.0])


def test_dimensions_groups_and_labels_are_complete_and_consistent():
    track, g, grid = slow_straight()
    assert set(C.DESCRIPTORS) == set(D.FUNCTIONS) == set(C.DESCRIPTOR_GROUP) == set(C.DESCRIPTOR_DIM) == set(C.DESCRIPTOR_LABELS) and len(C.DESCRIPTORS) == 12
    for name in C.DESCRIPTORS:
        out = D.describe(name, track, g, grid)
        assert out.shape == (1, C.DESCRIPTOR_DIM[name]) and np.isfinite(out).all()
    assert set(C.DESCRIPTOR_GROUP.values()) == set(C.GROUPS) and C.BASELINE in C.DESCRIPTORS


def test_unknown_description_is_refused():
    track, g, grid = standing()
    with pytest.raises(ValueError):
        D.describe("farbe", track, g, grid)


def test_descriptions_work_for_whole_populations():
    grid = M.make_maze(3)
    genomes = R.random_genomes(np.random.default_rng(3), 17)
    _, track = R.simulate(genomes, grid, trajectories=True)
    for name in C.DESCRIPTORS:
        out = D.describe(name, track, genomes, grid)
        assert out.shape == (17, C.DESCRIPTOR_DIM[name])
        single = D.describe(name, track[5:6], genomes[5:6], grid)
        assert single[0] == pytest.approx(out[5])                                # eine Zeile hängt nur von ihrer eigenen Fahrt ab
