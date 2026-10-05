"""Verhaltensraum – was Novelty Search als „anders“ ansieht - Stück 3 der Konzepte-Linie "Novelty Search und Quality-Diversity"
Sebastian Hanisch - Operations Research und Machine Learning

Derselbe Hof-Roboter, dasselbe täuschende Labyrinth und dieselbe Novelty-Auswahl wie in Stück 2 - nur die Verhaltensbeschreibung wechselt: in welchem Raum misst Novelty, wie verschieden zwei Fahrten sind?
Zwölf Beschreibungen in drei Gruppen. Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import bs_constants as C
import bs_evaluation as E
import bs_maze as M
from bs_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params)
from bs_visualization import (build_coverage_chart, build_length_chart, build_median_chart, build_progress_bar_chart, build_progress_chart, build_run_figure)

st.set_page_config(page_title="Verhaltensraum – Sebastian Hanisch", layout="wide")

PRE = E.load_precomputed()
N = PRE["seeds"]


@st.cache_data(show_spinner=False)
def _pair(level, gens, seed, descriptor, k):
    return E.live_pair(level, gens, seed, descriptor, k)


@st.cache_data(show_spinner=False)
def _solvability(level):
    return E.solvability(level)


def _evals(x):
    return C.fmt_med(x)


def _cells(n):
    """Zellenzahl mit richtiger Einzahl (1 Zelle, 0 Zellen, 16 Zellen)."""
    return f"{n:.0f} Zelle" if round(n) == 1 else f"{n:.0f} Zellen"


def _mean(cell, key):
    return sum(cell[key]) / len(cell[key])


st.title("🧭 Verhaltensraum – was Novelty Search als „anders“ ansieht")
st.markdown(
    """
Novelty Search belohnt, **anders** zu sein, aber „anders“ gibt es nicht von selbst: jemand muss festlegen, **in welchem Raum** zwei Fahrten verglichen werden. Bisher war das die Endposition des Roboters, eine Wahl von Hand. Dieses Stück
tauscht sie aus: **zwölf Verhaltensbeschreibungen** in drei Gruppen (mit Ortsbezug, ohne Ortsbezug, mit Aufgabenwissen), alle im selben Labyrinth mit derselben Suche. Es misst, **wie viel an dieser Wahl hängt**: für die Falle auf Stufe 3 viel
weniger als erwartet, solange die Beschreibung etwas über den Ort sagt, aber nichts für die Schlangenlinie auf Stufe 4, nicht einmal, wenn die Beschreibung die Antwort schon enthält.
"""
)
st.caption(
    "Stück 3 der Linie „Novelty Search und Quality-Diversity“; Labyrinth, Roboter und Suche wie in [Stück 2](https://sebastianhanisch-novelty-search-demo.streamlit.app/) "
    "(Novelty ohne Archiv, denn das Archiv brachte dort nichts)."
)

with st.expander("Die zwölf Beschreibungen", expanded=True):
    rows = []
    for g in C.GROUPS:
        for d in C.DESCRIPTORS:
            if C.DESCRIPTOR_GROUP[d] == g:
                rows.append(f"| {C.GROUP_LABELS[g]} | {C.DESCRIPTOR_LABELS[d]} | {C.DESCRIPTOR_DIM[d]} |")
    st.markdown("| Gruppe | Beschreibung | Zahl der Werte |\n|---|---|---|\n" + "\n".join(rows))
    st.markdown(
        """
Mit Ortsbezug sind Beschreibungen, die aus den **Positionen der Fahrt** entstehen (Endposition, Position nach halber Fahrt, Wegpunkte, grobe Gegend). *Ohne Ortsbezug* bedeutet, dass sie aus der **Steuerfolge** entstehen (Weglänge und
Drehung, Mittelwerte der Gene). *Mit Aufgabenwissen* benutzen das Labyrinth oder das Ziel (wahrer Weg, Luftlinie): sie sind kein fairer Verhaltensraum, sondern zeigen, was die Suche mit einem Wegweiser schaffen könnte. Alle Werte sind in Zellen
gemessen; Novelty ist der mittlere Abstand zu den k nächsten Nachbarn im jeweiligen Raum.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{name}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name] or None)

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    level = st.select_slider("Labyrinth (Stufe)", options=C.LEVELS, key="level_select", format_func=lambda v: C.LEVEL_LABELS[v], help="0 freies Feld bis 4 Schlangenlinie; je höher, desto länger der Umweg.")
    descriptor = st.selectbox("Verhaltensbeschreibung", options=C.DESCRIPTORS, key="descriptor_select", format_func=lambda v: C.DESCRIPTOR_LABELS[v],
                              help="In welchem Raum die Novelty misst. Daneben läuft immer der Bezug (Endposition, wie in Stück 2) mit gleichem Seed und gleicher Anfangspopulation.")
    k = st.slider("Nachbarn k für die Novelty", *bounds("k_slider"), step=C.K_STEP, key="k_slider", help="Die Novelty ist der mittlere Abstand zu den k nächsten Nachbarn im gewählten Raum.")
    gens = st.slider("Generationen im Live-Lauf", *bounds("gens_slider"), step=C.GENS_STEP, key="gens_slider", help=f"Je Generation {C.POP} Individuen, also {C.POP} Evaluationen.")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1, key="seed_input", help="Bestimmt die Anfangspopulation und alle Zufallsentscheidungen; beide Suchen starten gleich.")
    st.button("🎲 Neuen Lauf würfeln", on_click=randomize_seed)

level, descriptor, k, gens, seed = int(level), str(descriptor), int(k), int(gens), int(seed)
sync_query_params({"level_select": level, "descriptor_select": descriptor, "k_slider": k, "gens_slider": gens, "seed_input": seed})

grid = M.make_maze(level)
lm = E.level_metrics(level)
free = lm["free"]

st.markdown("---")
st.markdown("## 🔗 Wie täuschend ist diese Stufe?")
st.caption(f"{C.LEVEL_LABELS[level]}. Die Maße stammen aus Stück 1: sie sagen, wie sehr die Fitness „Abstand zum Ziel“ in die Irre führt; die Novelty benutzt sie nicht.")
m1 = st.columns(4)
m1[0].metric("Kürzester Weg", f"{lm['path']} Zellen", delta=f"{lm['detour']:.2f}-facher Umweg", delta_color="off")
m1[1].metric("Fitness-Distanz-Korrelation", f"{lm['fdc']:.2f}", help="1 = die Luftlinie führt zum Ziel; je kleiner, desto täuschender.")
m1[2].metric("Fallen (lokale Minima)", f"{lm['minima']}", help="Zellen, in denen die Luftlinie zum Ziel kleiner ist als in allen freien Nachbarzellen.")
m1[3].metric("Freie Zellen", f"{free}", help="So viele Zellen kann eine Endposition belegen.")
dist_ok, steps_used = _solvability(level)
st.caption(f"Lösbar? Ja: eine aus dem kürzesten Pfad abgeleitete Steuerfolge fährt ins Ziel (Abstand am Ende {dist_ok:.2f} Zellen, {steps_used} von {C.STEPS} Schritten benutzt).")

st.markdown("---")
st.subheader("📐 Ein Lauf: Bezug gegen gewählte Beschreibung")
with st.spinner(f"Lasse beide Suchen {gens} Generationen laufen …"):
    (base, base_track), (desc, desc_track) = _pair(level, gens, seed, descriptor, k)
base_ok, desc_ok = base.first_solved is not None, desc.first_solved is not None
same = descriptor == C.BASELINE
if same:
    st.caption("Gewählt ist der Bezug selbst: beide Läufe sind identisch. Wählen Sie links eine andere Beschreibung.")
c_base, c_desc = st.columns(2)
with c_base:
    st.markdown(f"**{C.DESCRIPTOR_LABELS[C.BASELINE]}**")
    b1 = st.columns(2)
    b1[0].metric("Bezug: Ergebnis", "Ziel erreicht" if base_ok else "Ziel verfehlt")
    b1[1].metric("Bezug: Evaluationen bis zum Erfolg", _evals(base.first_solved) if not base_ok else C.fmt_int(base.first_solved))
    b2 = st.columns(2)
    b2[0].metric("Bezug: besuchte Zellen", f"{base.visited_cells[-1]} von {free}")
    b2[1].metric("Bezug: kleinster Restweg", _cells(base.best_progress[-1]))
    st.plotly_chart(build_run_figure(grid, base.end_positions[-1], base_track, base.visited_grid, base.last_scores), width="stretch", key=f"run_base_{level}_{gens}_{seed}_{k}")
with c_desc:
    st.markdown(f"**{C.DESCRIPTOR_LABELS[descriptor]}** ({C.GROUP_LABELS[C.DESCRIPTOR_GROUP[descriptor]]}, {C.DESCRIPTOR_DIM[descriptor]} {'Wert' if C.DESCRIPTOR_DIM[descriptor] == 1 else 'Werte'}, k = {k})")
    d1 = st.columns(2)
    d1[0].metric("Beschreibung: Ergebnis", "Ziel erreicht" if desc_ok else "Ziel verfehlt")
    d1[1].metric("Beschreibung: Evaluationen bis zum Erfolg", _evals(desc.first_solved) if not desc_ok else C.fmt_int(desc.first_solved))
    d2 = st.columns(2)
    d2[0].metric("Beschreibung: besuchte Zellen", f"{desc.visited_cells[-1]} von {free}")
    d2[1].metric("Beschreibung: kleinster Restweg", _cells(desc.best_progress[-1]))
    st.plotly_chart(build_run_figure(grid, desc.end_positions[-1], desc_track, desc.visited_grid, desc.last_scores), width="stretch", key=f"run_desc_{level}_{gens}_{seed}_{descriptor}_{k}")
st.caption("Hellblau: alle Zellen, in denen eine Fahrt je endete. Punkte: die Endpositionen der letzten Generation, gefärbt nach ihrer Novelty in der jeweiligen Beschreibung (hell = neuartig, die Suche bevorzugt diese). Orange: die beste Fahrt.")
col_c, col_d = st.columns(2)
with col_c:
    st.markdown("**Bester Abstand zum Ziel**")
    st.plotly_chart(build_progress_chart(base.best_distance, desc.best_distance, descriptor, lm["trap"]), width="stretch", key=f"progress_{level}_{gens}_{seed}_{descriptor}_{k}")
with col_d:
    st.markdown("**Besuchte Zellen**")
    st.plotly_chart(build_coverage_chart(base.visited_cells, desc.visited_cells, descriptor, free), width="stretch", key=f"coverage_{level}_{gens}_{seed}_{descriptor}_{k}")
if same:
    pass
elif desc_ok and not base_ok:
    st.info(f"Mit „{C.DESCRIPTOR_LABELS[descriptor]}“ erreicht die Suche das Ziel nach {C.fmt_int(desc.first_solved)} Evaluationen, mit der Endposition in {gens} Generationen noch nicht (bester Abstand {min(base.best_distance):.2f} Zellen). "
            "Das ist ein Lauf; ob es typisch ist, zeigt die Studie unten.")
elif base_ok and not desc_ok:
    st.info(f"Mit der Endposition erreicht die Suche das Ziel nach {C.fmt_int(base.first_solved)} Evaluationen, mit „{C.DESCRIPTOR_LABELS[descriptor]}“ in {gens} Generationen noch nicht (bester Abstand {min(desc.best_distance):.2f} Zellen). "
            "Das ist ein Lauf; ob es typisch ist, zeigt die Studie unten.")
elif base_ok and desc_ok:
    st.info(f"Beide Suchen erreichen das Ziel: die Endposition nach {C.fmt_int(base.first_solved)}, „{C.DESCRIPTOR_LABELS[descriptor]}“ nach {C.fmt_int(desc.first_solved)} Evaluationen. Ein einzelner Lauf zeigt Streuung; die Studie unten vergleicht {N} Läufe.")
else:
    st.info(f"Keine der beiden Suchen erreicht das Ziel in {gens} Generationen (bester Abstand Endposition {min(base.best_distance):.2f}, Beschreibung {min(desc.best_distance):.2f} Zellen; kleinster Restweg "
            f"{base.best_progress[-1]:.0f} gegen {desc.best_progress[-1]:.0f} Zellen). Mehr Generationen oder ein anderer Seed können das ändern; die Studie unten fasst {N} Läufe zusammen.")

st.markdown("---")
st.subheader("🔬 Wie viel hängt an der Beschreibung? Die Studie über 20 Läufe")
st.markdown(
    f"Je Stufe und Auswahl {N} Läufe mit unterschiedlichem Seed, jeweils bis zu {C.fmt_int(PRE['budget'])} Evaluationen ({C.STUDY_GENS} Generationen), Novelty ohne Archiv mit k = {C.K_DEFAULT}. Die Fitness (Auswahl nach dem Abstand zum "
    f"Ziel) steht als Bezug aus Stück 1 dabei. **Erfolg** heißt: irgendein Individuum endet höchstens {C.SUCCESS_RADIUS:g} Zellen vom Ziel, **Restweg**: der kürzeste Weg in Zellen von der besten besuchten Zelle zum Ziel."
)
st.markdown("**Stufe 3 (die Falle): wie schnell?**")
st.plotly_chart(build_median_chart(PRE, 3), width="stretch", key="median_chart")
st.markdown("**Stufe 4 (Schlangenlinie): wie weit?**")
st.plotly_chart(build_progress_bar_chart(PRE, 4), width="stretch", key="progress_bar_chart_4")
rows = []
for s in E.SELECTIONS:
    c3, c4 = E.study_cell(PRE, 3, s), E.study_cell(PRE, 4, s)
    label = "Fitness (Bezug)" if s == C.FITNESS else C.DESCRIPTOR_LABELS[s]
    group = "–" if s == C.FITNESS else C.GROUP_LABELS[C.DESCRIPTOR_GROUP[s]]
    dim = "–" if s == C.FITNESS else str(C.DESCRIPTOR_DIM[s])
    rows.append(f"| {label} | {group} | {dim} | {E.success_count(c3['runs'])} von {N} | {_evals(E.median_solved(c3['runs']))} | {_mean(c3, 'cells'):.2f} | {_mean(c3, 'progress'):.2f} | "
                f"{E.success_count(c4['runs'])} von {N} | {_mean(c4, 'cells'):.2f} | {_mean(c4, 'progress'):.2f} |")
st.markdown("| Auswahl | Gruppe | Werte | Stufe 3: Erfolg | Median der Evaluationen bis zum Erfolg | besuchte Zellen (von 465) | Restweg | Stufe 4: Erfolg | besuchte Zellen (von 427) | Restweg |\n|---|---|---|---|---|---|---|---|---|---|\n"
            + "\n".join(rows))
s3 = {d: E.study_cell(PRE, 3, d) for d in C.DESCRIPTORS}
full = [d for d in C.DESCRIPTORS if E.success_count(s3[d]["runs"]) == N]
med_space = [E.median_solved(s3[d]["runs"]) for d in C.DESCRIPTORS if C.DESCRIPTOR_GROUP[d] == "space" and d in full]
fit3 = E.study_cell(PRE, 3, C.FITNESS)
meds_full = [E.median_solved(s3[d]['runs']) for d in full]
ratio = max(meds_full) / min(meds_full)
st.info(
    f"**Stufe 3:** {len(full)} der 12 Beschreibungen lösen alle {N} Läufe, darunter alle mit Aufgabenwissen und sieben der acht mit Ortsbezug; der Median liegt dabei zwischen {_evals(min(med_space))} und {_evals(max(med_space))} Evaluationen "
    f"für die ortsbezogenen (Endposition: {_evals(E.median_solved(s3['end']['runs']))}). Die Fitness schafft {E.success_count(fit3['runs'])} von {N}. Ohne Ortsbezug und mit nur einer Koordinate wird es schlechter: "
    f"nur x {E.success_count(s3['x']['runs'])} von {N} (Median {_evals(E.median_solved(s3['x']['runs']))}), Mittelwerte der Gene {E.success_count(s3['gene_means']['runs'])} von {N} (Median {_evals(E.median_solved(s3['gene_means']['runs']))}), "
    f"Weglänge und Drehung {E.success_count(s3['path_turn']['runs'])} von {N} (Median {_evals(E.median_solved(s3['path_turn']['runs']))}). Schon die Wahl des Verhaltensraums ändert den Aufwand also um "
    f"den Faktor {ratio:.1f} (Median {_evals(min(meds_full))} bis {_evals(max(meds_full))} unter den Beschreibungen mit 20 von 20). **Stufe 4:** keine der 12 Beschreibungen löst auch nur einen von {12 * N} Läufen, auch nicht der wahre Restweg zum Ziel."
)

st.markdown("---")
st.subheader("🔬 Hängt Stufe 4 am Suchraum? Die Länge der Steuerfolge")
st.markdown(
    f"Stufe 4 mit derselben Suche, aber längerer Steuerfolge (Schritte statt 140 bis 300), je {N} Läufe à {C.STUDY_GENS} Generationen. Die Fitness steht neben Novelty auf der Endposition. Die Abbildung zeigt, wie nah dem Ziel die Suche im Mittel kommt."
)
st.plotly_chart(build_length_chart(PRE), width="stretch", key="length_chart")
rows = []
for st_ in PRE["length"]["steps"]:
    f, e = E.length_cell(PRE, C.FITNESS, st_), E.length_cell(PRE, C.BASELINE, st_)
    rows.append(f"| {st_} | {E.success_count(f['runs'])} von {N} | {_mean(f, 'progress'):.2f} | {E.success_count(e['runs'])} von {N} | {_evals(E.median_solved(e['runs']))} | {_mean(e, 'progress'):.2f} |")
st.markdown("| Schritte | Fitness: Erfolg | Fitness: Restweg | Novelty (Endposition): Erfolg | Median der Evaluationen | Novelty: Restweg |\n|---|---|---|---|---|---|\n" + "\n".join(rows))
e300 = E.length_cell(PRE, C.BASELINE, 300)
st.info(
    f"Je länger die Steuerfolge, desto näher kommt die Suche dem Ziel, bei Novelty wie bei der Fitness; bei 300 Schritten löst Novelty {E.success_count(e300['runs'])} von {N} Läufen (Median {_evals(E.median_solved(e300['runs']))} Evaluationen), die Fitness "
    f"{E.success_count(E.length_cell(PRE, C.FITNESS, 300)['runs'])}. Der Engpass von Stufe 4 liegt demnach im **Suchraum** (die Steuerfolge muss die Schlangenlinie eng genug nachfahren, sonst gibt es nichts, was Novelty belohnen könnte), "
    "nicht im Verhaltensraum: eine bessere Beschreibung löst ihn nicht."
)

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Beschreibung ist von Hand gewählt** | Jede Wahl ist eine Annahme darüber, was an einer Fahrt wesentlich ist. Mit Ortsbezug trägt sie hier gut; ohne Ortsbezug wird die Suche langsamer oder bricht ab. | kein Folgestück |
| **Aufgabenwissen im Raum ist erlaubt** | Mit dem wahren Weg zum Ziel im Raum ist die Suche kein Entdecken mehr, sondern eine Zielsuche mit Umweg; sie löst Stufe 4 trotzdem nicht. | kein Folgestück |
| **Ein Maßstab für alle Werte** | Alle Werte sind in Zellen gemessen; die Maße (Drehung × 0.2, Gene × 20, Richtung × 3) sind von Hand gewählt. Andere Maßstäbe verschieben die Gewichte zwischen den Werten und damit die Zahlen. Nicht gemessen. | kein Folgestück |
| **Eine Suche mit fester Steuerfolge** | Stufe 4 hängt an der Länge der Steuerfolge, nicht an der Beschreibung: der Suchraum selbst ist der Engpass. | **Go-Explore** und **POET** (Folgestücke) |
| **Einzelne Lösung gesucht** | Die Suche liefert den Weg zum Ziel, nicht die Menge verschiedener guter Wege in einem Archiv nach Verhaltensbereichen. | **MAP-Elites** (Folgestück) |
"""
)
st.caption(
    "Verwandt im Portfolio: [deception-maze-demo](https://sebastianhanisch-deception-maze-demo.streamlit.app/) (Stück 1: das täuschende Labyrinth), "
    "[novelty-search-demo](https://sebastianhanisch-novelty-search-demo.streamlit.app/) (Stück 2: Novelty Search mit und ohne Archiv), "
    "[autoencoder-demo](https://sebastianhanisch-autoencoder-demo.streamlit.app/) (gelernte Beschreibungen verdichteter Daten)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Fahrt.** Positionen $p_0,\dots,p_T$ und Steuerfolge $g=(a_t, v_t)_{t=1}^T$ mit $T=140$ (Kinematik wie in Stück 1); Endposition $p_T$, Richtung $\theta_T = \theta_0 + 0.6\sum_t a_t$.

**Beschreibung.** Eine Abbildung $b(g)\in\mathbb{R}^d$, $d\in\{1,2,4,8,16\}$: Endposition $p_T$; $(p_T, 3\cos\theta_T, 3\sin\theta_T)$; $4\,(\lfloor p_T/4\rfloor + \tfrac12)$; $p_{\lfloor T/2\rfloor}$; $(p_{\lfloor jT/m\rfloor})_{j=1}^{m}$ für $m=4,8$;
$x_T$ oder $y_T$; $\bigl(\sum_t \lVert p_t-p_{t-1}\rVert,\ 0.2\sum_t |a_t|\bigr)$; $\bigl(20\,\bar a, 20\,\bar v\bigr)$; $d_{\text{Weg}}(\text{Zelle}(p_T))$ (Breitensuche zum Ziel); $\lVert p_T - z\rVert_2$.

**Novelty.** $\rho_k(i) = \frac{1}{k}\sum_{j\in N_k(i)}\lVert b(g_i)-b(g_j)\rVert_2$ mit den $k$ nächsten anderen Individuen $N_k(i)$ der Population (ein anderes Individuum am selben Ort zählt mit Abstand 0). Ausgewählt wird nach $\rho_k$ (Turnier 3, Elite 2,
Mutation Gauß 0.25 auf 15 % der Gene), kein Archiv.

**Messgrößen.** Erfolg: $\lVert p_T - z\rVert_2 < 2$. Besuchte Zellen: Zahl der Zellen, die mindestens eine Endposition enthielten. Restweg: $\min d_{\text{Weg}}(c)$ über die besuchten Zellen $c$.

Implementiert in `bs_descriptors.py` (die zwölf Beschreibungen), `bs_search.py` (Novelty, Lauf), `bs_maze.py` und `bs_robot.py` (wie Stück 1).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html))."
)
