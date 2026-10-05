"""Presets: Vollständigkeit, gültige Werte, Permalink-Konstanten, Formatierer."""

import pytest

import bs_constants as C
import bs_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER) and len(C.PRESETS) == 4
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for preset in C.PRESETS.values():
        assert preset["level"] in C.LEVELS and C.GENS_MIN <= preset["gens"] <= C.GENS_MAX and P.snap_gens(preset["gens"]) == preset["gens"]
        assert preset["descriptor"] in C.DESCRIPTORS and C.K_MIN <= preset["k"] <= C.K_MAX
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(preset[key])


def test_preset_names_state_the_values_they_set():
    assert C.PRESETS["Falle: Endposition gegen halbe Fahrt"]["descriptor"] == "mid" and C.PRESETS["Falle: Endposition gegen halbe Fahrt"]["level"] == 3
    assert C.PRESETS["Ohne Ortsbezug"]["descriptor"] == "gene_means" and C.DESCRIPTOR_GROUP["gene_means"] == "blind"
    assert C.PRESETS["Mit Aufgabenwissen"]["descriptor"] == "bfs" and C.DESCRIPTOR_GROUP["bfs"] == "knowledge"
    assert C.PRESETS["Schlangenlinie"]["level"] == 4


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Falle: Endposition gegen halbe Fahrt"]
    assert (p["level"], p["gens"], p["seed"], p["descriptor"], p["k"]) == (C.DEFAULT_LEVEL, C.DEFAULT_GENS, C.DEFAULT_SEED, C.DEFAULT_DESCRIPTOR, C.K_DEFAULT)


def test_bounds_and_url_params():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("gens_slider") == (10, 200) and P.bounds("level_select") == (0, 4) and P.bounds("k_slider") == (1, 30)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS) == 5


@pytest.mark.parametrize("value,expected", [(0, 10), (14, 10), (16, 20), (104, 100), (106, 110), (999, 200)])
def test_generations_snap_to_the_step_from_the_lower_bound(value, expected):
    assert P.snap_gens(value) == expected


@pytest.mark.parametrize("value", C.DESCRIPTORS)
def test_descriptor_caster_accepts_every_description(value):
    assert P.SETTING_SPECS["descriptor_select"].caster(value) == value


@pytest.mark.parametrize("value", ["fitness", "zufall", "", 3, None, ["end"]])
def test_descriptor_caster_refuses_everything_else(value):
    """Die Fitness ist keine Beschreibung und im Live-Lauf nicht wählbar."""
    with pytest.raises(ValueError):
        P.SETTING_SPECS["descriptor_select"].caster(value)


def test_the_constant_lists_are_consistent():
    assert len(C.DESCRIPTORS) == len(set(C.DESCRIPTORS)) == 12 and set(C.GROUP_LABELS) == set(C.GROUPS)
    assert [sum(1 for d in C.DESCRIPTORS if C.DESCRIPTOR_GROUP[d] == g) for g in C.GROUPS] == [8, 2, 2]


def test_formatters():
    assert C.fmt_int(40000) == "40 000" and C.fmt_pct(0.55) == "55 %" and C.fmt_pct(0.0898, 1) == "9.0 %"
    assert C.fmt_med(None) == "–" and C.fmt_med(988.5) == "988.5" and C.fmt_med(1195.0) == "1 195" and C.fmt_med(1308.5) == "1 308.5" and C.fmt_med(63.5) == "63.5"
