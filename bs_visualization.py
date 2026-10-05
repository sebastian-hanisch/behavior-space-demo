"""Plotly-Abbildungen der Verhaltensraum-Demo: Lauf im Labyrinth (Endpositionen nach Novelty gefärbt, besuchte Zellen), Verlauf des Abstands und der besuchten Zellen, Vergleich der zwölf Beschreibungen,
Restweg über der Genomlänge. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen. Das Raster `grid[i, j]` wird mit i als x-Achse gezeichnet."""

import numpy as np
import plotly.graph_objects as go

import bs_constants as C
import bs_evaluation as E

WALL_COLOR = "#3a3f47"
VISITED_COLOR = "#d6e4f5"
START_COLOR, GOAL_COLOR, PATH_COLOR, TRAP_COLOR = "#54a24b", "#e45756", "#f58518", "#b279a2"
BASE_COLOR, DESC_COLOR = "#e45756", "#4c78a8"
GROUP_COLORS = {"space": "#4c78a8", "blind": "#f58518", "knowledge": "#54a24b"}
FITNESS_COLOR = "#e45756"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25, top=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _arena_axes(fig):
    """Beide Achsen 0 bis N mit gleichem Maßstab (quadratische Zellen), Zeichenfläche passt sich der kleineren Seite an (constrain=domain)."""
    fig.update_xaxes(range=[0, C.N], showgrid=False, zeroline=False, title_text="x (Zellen)", constrain="domain")
    fig.update_yaxes(range=[0, C.N], showgrid=False, zeroline=False, title_text="y (Zellen)", scaleanchor="x", scaleratio=1, constrain="domain")


def _markers(fig):
    fig.add_trace(go.Scatter(x=[C.START[0]], y=[C.START[1]], mode="markers", marker=dict(color=START_COLOR, size=13, symbol="square", line=dict(color="white", width=1.5)), name="Start"))
    fig.add_trace(go.Scatter(x=[C.GOAL[0]], y=[C.GOAL[1]], mode="markers", marker=dict(color=GOAL_COLOR, size=16, symbol="star", line=dict(color="white", width=1.5)), name="Ziel"))
    fig.add_shape(type="circle", xref="x", yref="y", x0=C.GOAL[0] - C.SUCCESS_RADIUS, x1=C.GOAL[0] + C.SUCCESS_RADIUS, y0=C.GOAL[1] - C.SUCCESS_RADIUS, y1=C.GOAL[1] + C.SUCCESS_RADIUS,
                  line=dict(color=GOAL_COLOR, dash="dot", width=1.5))


def build_run_figure(grid, end_positions, track, visited, scores):
    """Ein Lauf: Wände (dunkel), alle besuchten Zellen (hellblau), Endpositionen der letzten Generation gefärbt nach ihrer Novelty in der gewählten Beschreibung (hell = neuartig) und die beste Fahrt des Laufs (orange)."""
    z = np.where(grid, 2.0, np.where(visited, 1.0, np.nan))
    fig = go.Figure(go.Heatmap(z=z.T, x0=0.5, dx=1, y0=0.5, dy=1, zmin=1, zmax=2, colorscale=[[0, VISITED_COLOR], [0.5, VISITED_COLOR], [0.5, WALL_COLOR], [1, WALL_COLOR]], showscale=False, hoverinfo="skip",
                               xgap=0, ygap=0))
    fig.add_trace(go.Scatter(x=end_positions[:, 0], y=end_positions[:, 1], mode="markers", name="Endpositionen der letzten Generation (Farbe: Novelty)",
                             marker=dict(color=scores, colorscale="Viridis", size=7, line=dict(color="white", width=0.5), showscale=True,
                                         colorbar=dict(title=dict(text="Novelty", side="top"), orientation="h", thickness=10, len=0.9, y=-0.2, yanchor="top"))))
    fig.add_trace(go.Scatter(x=track[:, 0], y=track[:, 1], mode="lines", line=dict(color=PATH_COLOR, width=2.5), name="beste Fahrt"))
    _markers(fig)
    _arena_axes(fig)
    return _base(fig, 480, legend_y=-0.45, top=0)


def build_progress_chart(base_best, desc_best, descriptor, trap):
    """Kleinster Abstand zum Ziel je Generation, Bezug (Endposition) gegen die gewählte Beschreibung; gestrichelt: Zielkreis und (wenn vorhanden) die Tiefe der tiefsten Falle."""
    gens = list(range(1, len(base_best) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=base_best, mode="lines", line=dict(color=BASE_COLOR, width=2.5), name=C.DESCRIPTOR_LABELS[C.BASELINE]))
    fig.add_trace(go.Scatter(x=gens, y=desc_best, mode="lines", line=dict(color=DESC_COLOR, width=2.5, dash="dash" if descriptor == C.BASELINE else "solid"), name=C.DESCRIPTOR_LABELS[descriptor]))
    fig.add_hline(y=C.SUCCESS_RADIUS, line=dict(color=GOAL_COLOR, dash="dot"), annotation_text="Zielkreis")
    if trap is not None:
        fig.add_hline(y=trap, line=dict(color=TRAP_COLOR, dash="dash"), annotation_text="tiefste Falle")
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="bester Abstand zum Ziel (Zellen)", rangemode="tozero")
    return _base(fig, 320, legend_y=-0.35)


def build_coverage_chart(base_visited, desc_visited, descriptor, free):
    """Anteil der freien Zellen, die die Suche bis zur jeweiligen Generation besucht hat (Endpositionen), Bezug gegen gewählte Beschreibung."""
    gens = list(range(1, len(base_visited) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=[v / free for v in base_visited], mode="lines", line=dict(color=BASE_COLOR, width=2.5), name=C.DESCRIPTOR_LABELS[C.BASELINE]))
    fig.add_trace(go.Scatter(x=gens, y=[v / free for v in desc_visited], mode="lines", line=dict(color=DESC_COLOR, width=2.5, dash="dash" if descriptor == C.BASELINE else "solid"), name=C.DESCRIPTOR_LABELS[descriptor]))
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="besuchte Zellen (Anteil der freien)", tickformat=".0%", range=[0, 1.03])
    return _base(fig, 320, legend_y=-0.35)


def _label(selection):
    return "Fitness (Bezug)" if selection == C.FITNESS else C.DESCRIPTOR_LABELS[selection]


def _color(selection):
    return FITNESS_COLOR if selection == C.FITNESS else GROUP_COLORS[C.DESCRIPTOR_GROUP[selection]]


def build_median_chart(pre, level):
    """Median der Evaluationen bis zum Erfolg je Auswahl (nur über die erfolgreichen Läufe); Beschriftung: Zahl der erfolgreichen Läufe von 20. Farbe nach Gruppe der Beschreibung."""
    n = pre["seeds"]
    rows = [(s, E.median_solved(E.study_cell(pre, level, s)["runs"]), E.success_count(E.study_cell(pre, level, s)["runs"])) for s in E.SELECTIONS]
    fig = go.Figure(go.Bar(y=[_label(s) for s, _, _ in rows], x=[m or 0 for _, m, _ in rows], orientation="h", marker_color=[_color(s) for s, _, _ in rows],
                           text=[f"{c} von {n}" + ("" if m is None else f" · Median {C.fmt_med(m)}") for _, m, c in rows], textposition="outside", cliponaxis=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Median der Evaluationen bis zum Erfolg (nur erfolgreiche Läufe)", range=[0, max((m or 0) for _, m, _ in rows) * 1.55])
    return _base(fig, 430)


def build_progress_bar_chart(pre, level):
    """Mittlerer kleinster wahrer Restweg am Ende des Laufs (Zellen) je Auswahl; 0 heißt: das Ziel wurde in jedem Lauf besucht."""
    rows = [(s, float(np.mean(E.study_cell(pre, level, s)["progress"]))) for s in E.SELECTIONS]
    fig = go.Figure(go.Bar(y=[_label(s) for s, _ in rows], x=[v for _, v in rows], orientation="h", marker_color=[_color(s) for s, _ in rows], text=[f"{v:.2f}" for _, v in rows], textposition="outside", cliponaxis=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text=f"mittlerer kleinster wahrer Restweg (Zellen), Start: {E.level_info(pre, level)['path']}", range=[0, max(v for _, v in rows) * 1.2 + 1])
    return _base(fig, 430)


def build_length_chart(pre):
    """Kleinster wahrer Restweg auf Stufe 4 über der Länge der Steuerfolge, Fitness gegen Novelty auf der Endposition; Beschriftung: erfolgreiche Läufe von 20."""
    n = pre["seeds"]
    steps = list(pre["length"]["steps"])
    fig = go.Figure()
    for sel, color in ((C.FITNESS, FITNESS_COLOR), (C.BASELINE, DESC_COLOR)):
        cells = [E.length_cell(pre, sel, st) for st in steps]
        fig.add_trace(go.Scatter(x=steps, y=[float(np.mean(c["progress"])) for c in cells], mode="lines+markers+text", line=dict(color=color, width=2.5), name=_label(sel),
                                 text=[f"{E.success_count(c['runs'])} von {n}" for c in cells], textposition="top center"))
    fig.update_xaxes(title_text="Länge der Steuerfolge (Schritte)", tickvals=steps)
    fig.update_yaxes(title_text="mittlerer kleinster wahrer Restweg (Zellen)", rangemode="tozero")
    return _base(fig, 340, legend_y=-0.3)
