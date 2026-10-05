"""AppTest-Rauchtests: Voreinstellung, jedes Preset, alle Stufen und Beschreibungen, Randwerte, lokale Regler, Würfel-Knopf, Permalink-Grenzen, Abschnitte, Footer."""

import random
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import bs_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_the_reference_values():
    at = _run()
    _ok(at)
    assert _metric(at, "Kürzester Weg") == "49 Zellen" and _metric(at, "Fitness-Distanz-Korrelation") == "0.69" and _metric(at, "Fallen (lokale Minima)") == "1" and _metric(at, "Freie Zellen") == "465"
    assert _metric(at, "Bezug: Ergebnis") in ("Ziel erreicht", "Ziel verfehlt") and _metric(at, "Beschreibung: Ergebnis") in ("Ziel erreicht", "Ziel verfehlt")
    assert len(at.get("plotly_chart")) == 7


def test_all_sections_are_present():
    at = _run()
    assert [s.value for s in at.subheader] == ["📐 Ein Lauf: Bezug gegen gewählte Beschreibung", "🔬 Wie viel hängt an der Beschreibung? Die Studie über 20 Läufe",
                                               "🔬 Hängt Stufe 4 am Suchraum? Die Länge der Steuerfolge", "🚧 Wo die Annahmen enden"]
    assert any(m.value.startswith("## 🔗 Wie täuschend ist diese Stufe?") for m in at.markdown)
    assert any("Diese Demo ist Teil des Portfolios" in c.value for c in at.caption)


def test_the_description_table_lists_all_twelve():
    at = _run()
    table = next(m.value for m in at.markdown if m.value.startswith("| Gruppe | Beschreibung |"))
    assert table.count("\n| ") == 12 and all(C.DESCRIPTOR_LABELS[d] in table for d in C.DESCRIPTORS)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert (at.session_state["level_select"], at.session_state["gens_slider"], at.session_state["descriptor_select"], at.session_state["k_slider"]) == (p["level"], p["gens"], p["descriptor"], p["k"])
    assert at.metric


@pytest.mark.parametrize("level", list(C.LEVELS))
def test_every_level_runs(level):
    at = _run(level_select=level, gens_slider=10)
    _ok(at)
    assert _metric(at, "Bezug: besuchte Zellen").endswith(f"von {_metric(at, 'Freie Zellen')}")


@pytest.mark.parametrize("descriptor", C.DESCRIPTORS)
def test_every_description_runs(descriptor):
    at = _run(descriptor_select=descriptor, gens_slider=20)
    _ok(at)
    assert len(at.get("plotly_chart")) == 7 and _metric(at, "Beschreibung: besuchte Zellen")


def test_choosing_the_baseline_itself_says_so_and_gives_equal_metrics():
    at = _run(descriptor_select=C.BASELINE, gens_slider=30)
    _ok(at)
    assert any("Gewählt ist der Bezug selbst" in c.value for c in at.caption)
    for a, b in (("Bezug: besuchte Zellen", "Beschreibung: besuchte Zellen"), ("Bezug: Ergebnis", "Beschreibung: Ergebnis"), ("Bezug: kleinster Restweg", "Beschreibung: kleinster Restweg")):
        assert _metric(at, a) == _metric(at, b)


def test_free_field_both_searches_reach_the_goal():
    at = _run(level_select=0)
    _ok(at)
    assert _metric(at, "Bezug: Ergebnis") == "Ziel erreicht" and _metric(at, "Beschreibung: Ergebnis") == "Ziel erreicht" and _metric(at, "Fallen (lokale Minima)") == "0"


def test_the_serpentine_fails_for_both_searches():
    at = _run(level_select=4, gens_slider=100)
    _ok(at)
    assert _metric(at, "Fallen (lokale Minima)") == "2" and _metric(at, "Kürzester Weg") == "77 Zellen"
    assert _metric(at, "Bezug: Ergebnis") == "Ziel verfehlt" and _metric(at, "Beschreibung: Ergebnis") == "Ziel verfehlt"


def test_the_solvability_line_names_the_steps_used():
    at = _run(level_select=4)
    assert any("136 von 140 Schritten benutzt" in c.value for c in at.caption)


@pytest.mark.parametrize("kw", [dict(gens_slider=10), dict(gens_slider=200), dict(k_slider=1), dict(k_slider=30), dict(level_select=3, gens_slider=200, seed_input=0),
                                dict(level_select=1, gens_slider=10, seed_input=999999, k_slider=1, descriptor_select="way8")])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_the_study_table_lists_every_selection_with_both_levels():
    at = _run()
    table = next(m.value for m in at.markdown if m.value.startswith("| Auswahl | Gruppe |"))
    assert table.count("\n| ") == 13 and "| 11 von 20 |" in table and table.count("| 0 von 20 |") == 13


def test_the_study_info_names_the_nine_full_solvers_and_the_zero_on_level_four():
    at = _run()
    text = next(i.value for i in at.info if i.value.startswith("**Stufe 3:**"))
    assert "9 der 12 Beschreibungen lösen alle 20 Läufe" in text and "Faktor 5.7" in text and "0 von 240" not in text and "auch nur einen von 240 Läufen" in text


def test_the_length_info_names_the_two_successes_at_300_steps():
    at = _run()
    assert any("bei 300 Schritten löst Novelty 2 von 20 Läufen (Median 30 402 Evaluationen)" in i.value for i in at.info)


def test_dice_button_changes_the_seed_and_the_run(monkeypatch):
    """Der Würfel zieht sonst einen unseeded Zufalls-Seed; deshalb ist der gewürfelte Seed im Test fest."""
    monkeypatch.setattr(random, "randint", lambda a, b: 508145)
    at = _run(level_select=2)
    old_seed = at.session_state["seed_input"]
    old = (_metric(at, "Bezug: Evaluationen bis zum Erfolg"), _metric(at, "Beschreibung: Evaluationen bis zum Erfolg"))
    next(b for b in at.button if b.label == "🎲 Neuen Lauf würfeln").click().run()
    _ok(at)
    assert at.session_state["seed_input"] == 508145 != old_seed
    assert (_metric(at, "Bezug: Evaluationen bis zum Erfolg"), _metric(at, "Beschreibung: Evaluationen bis zum Erfolg")) != old


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["level"] = "9"
    at.query_params["gens"] = "104"
    at.query_params["k"] = "99"
    at.query_params["descriptor"] = "way8"
    at.run()
    _ok(at)
    assert at.session_state["level_select"] == 4 and at.session_state["gens_slider"] == 100 and at.session_state["k_slider"] == 30 and at.session_state["descriptor_select"] == "way8"


def test_permalink_ignores_garbage():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["level"] = "viele"
    at.query_params["seed"] = "nan"
    at.query_params["descriptor"] = "fitness"
    at.query_params["k"] = "0"
    at.run()
    _ok(at)
    assert at.session_state["level_select"] == C.DEFAULT_LEVEL and at.session_state["seed_input"] == C.DEFAULT_SEED and at.session_state["descriptor_select"] == C.DEFAULT_DESCRIPTOR
    assert at.session_state["k_slider"] == C.K_MIN                                            # 0 liegt unter der Grenze und wird auf 1 gehoben


def test_no_sentence_wide_comma_replacement_in_the_app_source():
    """Regressionsschutz: `.replace(",", ".")` auf einem ganzen (verketteten) Satz macht aus Kommas im Fließtext Punkte; Tausender nur über `fmt_int`."""
    source = Path(APP).read_text(encoding="utf-8")
    assert '.replace(",", ".")' not in source
