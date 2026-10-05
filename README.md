# Verhaltensraum – was Novelty Search als „anders“ ansieht (Streamlit-Demo)

---

Interaktive Demo zum **Verhaltensraum von Novelty Search**. **Drittes Stück der Konzepte-Linie „Novelty Search und Quality-Diversity“** im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning): ein Verfahren, ein wachsendes Beispiel, jedes Folgestück hebt genau eine Annahme auf.

Derselbe Hof-Roboter, dasselbe täuschende Labyrinth und dieselbe Suche wie in [`deception-maze-demo`](https://github.com/sebastian-hanisch/deception-maze-demo) (Stück 1) und
[`novelty-search-demo`](https://github.com/sebastian-hanisch/novelty-search-demo) (Stück 2), aber die **Verhaltensbeschreibung** wechselt: In welchem Raum misst Novelty, wie verschieden zwei Fahrten sind? Bisher war es die Endposition, eine Wahl von
Hand. Das Stück vergleicht **zwölf Beschreibungen** in drei Gruppen (mit Ortsbezug, ohne Ortsbezug, mit Aufgabenwissen) und misst, **wie viel an dieser Wahl hängt**, und was sie **nicht** erklärt: den Engpass auf Stufe 4.

## Kernfrage

Wie stark hängt der Erfolg von Novelty Search an der gewählten Verhaltensbeschreibung, und was bleibt ungelöst, auch wenn die Beschreibung die Antwort schon enthält?

## Modell und Methodik

- **Labyrinth und Roboter** (`bs_maze.py`, `bs_robot.py`): unverändert aus Stück 1. 24 × 24 Zellen, Start (3.5, 3.5), Ziel (3.5, 20.5); Stufe 3 eine Querwand mit Öffnung rechts (Falle 9 Zellen vor dem Ziel), Stufe 4 drei Wände (Öffnungen rechts, links,
  rechts). Feste Steuerfolge aus 140 Befehlen, kein Sensor. **Erfolg:** eine Endposition höchstens 2 Zellen vom Ziel.
- **Suche** (`bs_search.py`): derselbe Genetische Algorithmus (100 Individuen, Turnier der Größe 3, die zwei Besten bleiben, Mutation Gauß σ = 0.25 auf 15 % der Gene, keine Kreuzung). Gewählt wird nach der **Fitness** (−Abstand zum Ziel, Bezug) oder nach
  **Novelty** (mittlerer Abstand zu den k = 15 nächsten Individuen der Population im gewählten Raum, **ohne Archiv**, denn das Archiv brachte in Stück 2 hier nichts).
- **Die zwölf Beschreibungen** (`bs_descriptors.py`; Werte in Zellen):
  - *mit Ortsbezug* (aus den Positionen der Fahrt): Endposition (2 Werte, der Bezug aus Stück 2); Endposition und Endrichtung (4: dazu 3·cos θ, 3·sin θ); grobe Endposition (2: Mitte des 4×4-Blocks); Position nach der halben Fahrt (2); 4 und 8 Wegpunkte
    (8 und 16 Werte: Positionen nach gleich langen Teilen der Fahrt); nur x und nur y der Endposition (je 1).
  - *ohne Ortsbezug* (aus der Steuerfolge): zurückgelegte Weglänge und 0.2 · Summe der Drehbeträge (2); 20 · Mittel der Drehgene und 20 · Mittel der Tempogene (2).
  - *mit Aufgabenwissen* (benutzt Labyrinth oder Ziel): wahrer Weg von der Endzelle zum Ziel (Breitensuche, 1); Luftlinie zum Ziel (1).
- **Messgrößen:** Evaluationen bis zum ersten Erfolg; **besuchte Zellen**; **wahrer Restweg** (kürzester Weg in Zellen von der besten besuchten Zelle zum Ziel).
- **Gegenproben:** (1) jede Beschreibung von Hand an einer Fahrt mit bekannten Werten (Tempo 0.1 geradeaus: Weg genau 9.8 Zellen, Wegpunkte nach t · 0.07 Zellen, wahrer Weg von zwei Zellen von Hand gezählt: 41 und 7); (2) **Orakel:** alle zwölf Beschreibungen
  gegen eine skalare Neuberechnung der Fahrt aus der Beschreibung der Kinematik (eine Schleife je Individuum und Schritt, `math` statt NumPy) auf drei Stufen, die Novelty gegen SciPys `cKDTree` und eine skalare Definition in 1, 2, 4 und 16 Dimensionen
  mit und ohne doppelte Punkte, der Restweg gegen SciPys Breitensuche; (3) Ziel wandert: nur Fitness, wahrer Restweg und Luftlinie ändern ihre Läufe, alle anderen Auswahlen nicht; (4) die Endposition reproduziert die Läufe aus Stück 2 (Seeds 0 bis 5
  auf Stufe 3: 1 246, 1 737, 515, 631, 2 035, 1 641 Evaluationen); (5) Studienläufe werden mit demselben Seed frisch reproduziert, auch bei 300 Schritten.
- **Vorgerechnete Studie** (`generate_precomputed.py` → `precomputed_sweep.json`, rund vier Minuten parallel): Stufe 3 und 4, je Fitness und zwölf Beschreibungen, 20 Läufe à 400 Generationen (40 000 Evaluationen); dazu Stufe 4 mit 140 / 180 / 220 / 300 Schritten
  für die Fitness und die Endposition (je 20 Läufe).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. 20 Läufe je Zelle (Seeds 0 bis 19), 40 000 Evaluationen, Novelty ohne Archiv mit k = 15. Der Median zählt nur die erfolgreichen Läufe.

| Auswahl | Gruppe | Werte | Stufe 3: Erfolg | Median | besuchte Zellen (von 465) | Restweg | Stufe 4: Erfolg | besuchte Zellen (von 427) | Restweg (bester Lauf) |
|---|---|---|---|---|---|---|---|---|---|
| Fitness (Bezug) | – | – | 11 von 20 | 1 267 | 343.90 | 5.55 | 0 von 20 | 223.80 | 34.75 (22) |
| Endposition (Bezug) | mit Ortsbezug | 2 | 20 von 20 | 988.5 | 465.00 | 0.00 | 0 von 20 | 264.15 | 26.25 (22) |
| Endposition und Endrichtung | mit Ortsbezug | 4 | 20 von 20 | 801.5 | 465.00 | 0.00 | 0 von 20 | 266.50 | 27.20 (25) |
| grobe Endposition (4×4-Blöcke) | mit Ortsbezug | 2 | 20 von 20 | 1 077 | 465.00 | 0.00 | 0 von 20 | 266.45 | 27.10 (23) |
| Position nach der halben Fahrt | mit Ortsbezug | 2 | 20 von 20 | 4 478 | 462.85 | 0.00 | 0 von 20 | 249.40 | 29.00 (23) |
| 4 Wegpunkte der Fahrt | mit Ortsbezug | 8 | 20 von 20 | 1 087.5 | 464.95 | 0.00 | 0 von 20 | 262.95 | 25.60 (20) |
| 8 Wegpunkte der Fahrt | mit Ortsbezug | 16 | 20 von 20 | 1 488 | 464.95 | 0.00 | 0 von 20 | 258.25 | 27.10 (20) |
| nur x der Endposition | mit Ortsbezug | 1 | 17 von 20 | 16 498 | 401.90 | 1.65 | 0 von 20 | 204.35 | 37.60 (31) |
| nur y der Endposition | mit Ortsbezug | 1 | 20 von 20 | 1 401.5 | 463.95 | 0.00 | 0 von 20 | 255.20 | 29.00 (23) |
| Weglänge und Gesamtdrehung | ohne Ortsbezug | 2 | 19 von 20 | 4 040 | 441.05 | 0.65 | 0 von 20 | 225.15 | 34.30 (21) |
| Mittelwerte der Gene | ohne Ortsbezug | 2 | 18 von 20 | 9 579.5 | 385.60 | 1.60 | 0 von 20 | 198.60 | 38.70 (33) |
| wahrer Restweg zum Ziel | mit Aufgabenwissen | 1 | 20 von 20 | 790 | 465.00 | 0.00 | 0 von 20 | 283.30 | 24.75 (19) |
| Luftlinie zum Ziel | mit Aufgabenwissen | 1 | 20 von 20 | 1 717.5 | 465.00 | 0.00 | 0 von 20 | 263.05 | 27.85 (25) |

| Frage | Befund |
|---|---|
| Wie viel hängt auf Stufe 3 an der Beschreibung? | **Weniger als erwartet, aber der Aufwand ändert sich um das Fünffache.** 9 der 12 Beschreibungen lösen alle 20 Läufe: sieben der acht mit Ortsbezug (nicht: nur x) und beide mit Aufgabenwissen. Ihr Median reicht von **790** (wahrer Restweg) bis **4 478** (Position nach der halben Fahrt), das ist der Faktor 5.7; unter den ortsbezogenen von 801.5 (Endposition und Endrichtung) bis 4 478. Die Endposition liegt mit 988.5 am unteren Ende. |
| Was bricht ohne Ortsbezug? | Es bricht nicht, aber es wird schlechter: Weglänge und Drehung 19 von 20 (Median 4 040, schlimmster Lauf 36 919), Mittelwerte der Gene 18 von 20 (Median 9 579.5, schlimmster Lauf 37 373; besucht im Mittel nur 385.6 von 465 Zellen). Auch eine einzelne Koordinate genügt nicht: nur x 17 von 20 (Median 16 498, schlimmster Lauf 39 751, 401.9 Zellen), nur y dagegen 20 von 20 (Median 1 401.5). |
| Was kostet die halbe Fahrt? | Die Position nach der halben Fahrt löst alle 20 Läufe, braucht aber im Median 4 478 statt 988.5 Evaluationen (schlimmster Lauf 14 233) und besucht 462.85 von 465 Zellen im Mittel: sie sieht vom Ende der Fahrt nichts. |
| Helfen mehr Werte (Wegpunkte)? | Nein: 4 Wegpunkte (8 Werte) 20 von 20, Median 1 087.5; 8 Wegpunkte (16 Werte) 20 von 20, Median 1 488; beide besuchen im Mittel 464.95 Zellen. Mehr Werte sind hier nicht schneller als die Endposition (988.5). |
| Hilft Aufgabenwissen? | **Auf Stufe 3 wenig:** der wahre Restweg ist mit Median 790 der schnellste, das ist nur das 1.25-fache der Endposition (988.5); die Luftlinie als einziger Wert braucht 1 717.5, löst aber alle 20 Läufe, obwohl sie die Zielnähe enthält (die Auswahl nach Fitness löst 11 von 20). **Auf Stufe 4 nicht:** 0 von 20. |
| Löst eine Beschreibung Stufe 4? | **Keine einzige.** 0 von 240 Läufen für die zwölf Beschreibungen (0 von 260 mit der Fitness). Der kleinste wahre Restweg liegt im Mittel zwischen 24.75 (wahrer Restweg; bester Lauf 19) und 38.7 (Mittelwerte der Gene); die Endposition erreicht 26.25, 4 Wegpunkte 25.6 (bester Lauf 20), die Fitness 34.75. |
| Woran hängt dann Stufe 4? | **Am Suchraum, nicht am Verhaltensraum.** Mit längerer Steuerfolge kommt die Suche voran: mittlerer Restweg bei 140 / 180 / 220 / 300 Schritten 34.75 / 25.05 / 19.95 / 17.8 (Fitness) und 26.25 / 21.25 / 15.65 / 9.05 (Novelty auf der Endposition). Bei 300 Schritten löst Novelty **2 von 20** Läufen (Median 30 402 Evaluationen, bester Restweg 2), die Fitness 0 von 20. Besuchte Zellen bei 300 Schritten: 348.95 (Novelty) gegen 315.95 (Fitness). |

## Befunde und Korrekturen gegenüber der Vorab-Messreihe

- **Die erwartete Hauptwirkung ist ausgeblieben.** Erwartet war, dass die Beschreibung der wichtigste Hebel von Novelty Search ist. Gemessen: Solange sie etwas über den Ort der Fahrt sagt, löst Novelty die Falle fast unabhängig von ihr
  (sieben Beschreibungen mit Ortsbezug und beide mit Aufgabenwissen: 20 von 20). Der Hebel liegt im Aufwand (Faktor 5.7), nicht in Gelingen oder Scheitern.
- **Selbst die Antwort im Verhaltensraum löst Stufe 4 nicht.** Der wahre Restweg zum Ziel ist ein Wegweiser, den die Fitness-Auswahl nie bekommt; als Beschreibung für Novelty löst er Stufe 4 trotzdem in 0 von 20 Läufen. Das widerlegt die Annahme, bessere
  Beschreibungen allein reichten. Die Folgerung: **gelernte** Beschreibungen (Folgestück) hätten auf diesem Labyrinth keinen Hebel auf Stufe 4, weil schon die richtige Beschreibung nicht hilft.
- **Der Engpass ist die Steuerfolge.** Die Suche kommt erst mit längerer Steuerfolge nennenswert voran; nur bei 300 Schritten löst Novelty überhaupt einzelne Läufe. Die Schlangenlinie braucht in dieser Parametrisierung eine Steuerfolge, die sie eng
  nachfährt (136 von 140 Schritten bei einer kontrollierten Fahrt), und kein Verhaltensmaß macht aus einer zu kurzen Folge eine lange.
- **Die Luftlinie als Verhalten ist kein Rückfall in die Fitness.** Die Fitness-Auswahl löst Stufe 3 in 11 von 20 Läufen; Novelty auf der Luftlinie (derselbe Wert, aber als „wie anders“ statt „wie nah“) in 20 von 20.

## Ehrliche Grenzen

- **Ein Maßstab für alle Werte.** Alle Beschreibungen sind in Zellen gemessen; die Faktoren (3 für die Richtung, 0.2 für die Gesamtdrehung, 20 für die Mittelwerte der Gene, 4 für die groben Blöcke) sind von Hand gewählt. Andere Maßstäbe verschieben die Gewichte
  zwischen den Werten und damit die Zahlen; **nicht gemessen**.
- **Zwölf Beschreibungen sind eine Auswahl, keine Vollständigkeit.** Andere Maße (etwa Geschwindigkeitsprofile, besuchte Zellen der Fahrt als Menge) sind nicht gerechnet.
- **20 Läufe je Zelle.** Erfolgsquoten wie 17, 18 oder 19 von 20 haben eine Standardabweichung von etwa 5 bis 8 Prozentpunkten; die Unterschiede zwischen 18, 19 und 20 von 20 liegen im Rauschen und werden nicht einzeln gedeutet. Die 2 von 20 bei 300
  Schritten sind 10 % mit einer Standardabweichung von etwa 7 Prozentpunkten: der Befund „löst nicht null“ ist belastbar, die genaue Quote nicht.
- **Vier Stützstellen für die Genomlänge** (140 / 180 / 220 / 300): wo genau die Schlangenlinie für die Suche lösbar wird, ist nicht gemessen, und bei 300 Schritten ist Stufe 4 nicht im Sinne der Linie „gelöst“, sondern in 2 von 20 Läufen gefunden.
- **Kein Archiv, ein einziges k.** Das Archiv brachte in Stück 2 nichts und fehlt hier; k = 15 ist fest. Ob andere Kombinationen die Reihenfolge der Beschreibungen ändern, ist nicht gemessen.
- **Ein einfacher GA ohne Kreuzung;** Kreuzung und andere Operatoren würden die Zahlen ändern (nicht gemessen).

## Verwandte Demos im Portfolio

- [`deception-maze-demo`](https://github.com/sebastian-hanisch/deception-maze-demo): Stück 1, das täuschende Labyrinth und die zielgetriebene Suche.
- [`novelty-search-demo`](https://github.com/sebastian-hanisch/novelty-search-demo): Stück 2, Novelty Search mit und ohne Archiv.
- [`autoencoder-demo`](https://github.com/sebastian-hanisch/autoencoder-demo): gelernte, verdichtete Beschreibungen von Daten.

## Bewusst nicht umgesetzt

Jede dieser Annahmen hebt ein Folgestück der Linie auf:

| Annahme | Folgestück |
|---|---|
| Die Beschreibung ist von Hand gewählt | kein Folgestück |
| Aufgabenwissen im Raum ist erlaubt | kein Folgestück |
| Ein Maßstab für alle Werte | kein Folgestück |
| Eine Suche mit fester Steuerfolge (Stufe 4 hängt am Suchraum) | Go-Explore und POET (Folgestücke) |
| Einzelne Lösung gesucht | MAP-Elites (Folgestück) |

## Tests

278 Tests, rund zwei Minuten: Labyrinth und Roboter (aus Stück 1), jede der zwölf Beschreibungen von Hand und gegen die skalare Kinematik, Novelty von Hand und gegen `cKDTree` und eine skalare Definition (1 bis 16 Dimensionen), Suche
(Reproduzierbarkeit je Auswahl, Zählung der Evaluationen, besuchte Zellen, Restweg, wer das Ziel sieht), Auswertung und Vollständigkeit der vorgerechneten Datei (inklusive Neurechnung einzelner Studien- und Längenläufe), Presets und Permalink, Diagramme
(gesperrte Achsen), AppTest-Rauchtests mit festem Würfel-Seed, der Smoke-Test der Portfolio-Vorlage, ein Quelltext-Test gegen Satz-Komma-Fehler und `test_claims.py` für jede Zahl dieser README.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `bs_maze.py`, `bs_robot.py` | Labyrinth, Breitensuche, Täuschungsmaße; Kinematik, Population, Pfad-Regler (wie Stück 1) |
| `bs_descriptors.py` | die zwölf Verhaltensbeschreibungen |
| `bs_search.py` | Suche mit Fitness oder Novelty in einem wählbaren Raum, besuchte Zellen, Restweg |
| `bs_evaluation.py` | Maße je Stufe, Lösbarkeit, Live-Paar, Zugriff auf die vorgerechnete Studie |
| `generate_precomputed.py` | rechnet die Studie vor → `precomputed_sweep.json` |
| `bs_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `bs_presets.py`, `bs_constants.py` | Presets, Permalink, Grenzen |
| `tests/` | siehe oben |

## Literatur

- Lehman, J., Stanley, K. O. (2011): Abandoning objectives: evolution through the search for novelty alone. *Evolutionary Computation* 19(2), 189–223 (Novelty als mittlerer Abstand zu den k nächsten Nachbarn im Verhaltensraum).
- Lehman, J., Stanley, K. O. (2008): Exploiting open-endedness to solve problems through the search for novelty. *Proceedings of the Eleventh International Conference on Artificial Life (ALIFE XI)*, MIT Press.

## Lokal ausführen

```
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -v`. Studie neu rechnen: `python generate_precomputed.py` (rund vier Minuten).

Gebaut mit Streamlit und Plotly (die Simulation rechnet NumPy).
