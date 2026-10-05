"""Verhaltensbeschreibungen: wie aus einer Fahrt (Steuerfolge und Positionen je Schritt) ein Punkt im Verhaltensraum wird. Novelty misst Abstände nur in diesem Raum; was die Beschreibung nicht enthält, kann die Suche nicht
unterscheiden. Alle Beschreibungen haben die Signatur `f(track, genomes, grid)` mit `track` (P, STEPS + 1, 2) und `genomes` (P, STEPS, 2) und geben ein Feld (P, Dimension) zurück; Längen in Zellen.

Mit Ortsbezug (aus den Positionen der Fahrt):
  end          Endposition (x, y), der Bezug aus Stück 2
  end_heading  Endposition und Endrichtung als (3 cos θ, 3 sin θ); θ = Anfangsrichtung + Summe der Drehungen
  coarse       Endposition auf die Mitte ihres 4×4-Blocks gerundet (nur Gegend, keine Lage im Block)
  mid          Position nach der halben Fahrt (Schritt STEPS // 2)
  way4, way8   Positionen nach 1/4, 2/4, 3/4, 4/4 (acht Zahlen) beziehungsweise nach 1/8 bis 8/8 (16 Zahlen) der Fahrt (Schritt = Brüche · STEPS, abgerundet)
  x, y         nur eine Koordinate der Endposition
Ohne Ortsbezug (aus der Steuerfolge, nicht aus dem Ort):
  path_turn    zurückgelegte Weglänge und 0.2 · Summe |Drehgen|
  gene_means   20 · Mittel der Drehgene und 20 · Mittel der Tempogene
Mit Aufgabenwissen (benutzt das Labyrinth oder das Ziel; die Suche bekäme sie ohne den Entwurf nicht):
  bfs          wahrer Weg (Zellen, Breitensuche) von der Endzelle zum Ziel
  euclid       Luftlinie der Endposition zum Ziel"""

import numpy as np

import bs_constants as C
import bs_maze as M


def _end(track):
    return track[:, -1, :]


def d_end(track, genomes, grid):
    return _end(track)


def d_end_heading(track, genomes, grid):
    heading = C.START_HEADING + (genomes[:, :, 0] * C.TURN_MAX).sum(axis=1)
    return np.column_stack([_end(track), C.HEADING_SCALE * np.cos(heading), C.HEADING_SCALE * np.sin(heading)])


def d_coarse(track, genomes, grid):
    return (np.floor(_end(track) / C.COARSE_BLOCK) + 0.5) * C.COARSE_BLOCK


def d_mid(track, genomes, grid):
    return track[:, genomes.shape[1] // 2, :]


def _waypoints(track, parts):
    steps = track.shape[1] - 1
    idx = [int(steps * i / parts) for i in range(1, parts + 1)]
    return track[:, idx, :].reshape(len(track), -1)


def d_way4(track, genomes, grid):
    return _waypoints(track, 4)


def d_way8(track, genomes, grid):
    return _waypoints(track, 8)


def d_x(track, genomes, grid):
    return _end(track)[:, :1]


def d_y(track, genomes, grid):
    return _end(track)[:, 1:]


def d_path_turn(track, genomes, grid):
    length = np.linalg.norm(np.diff(track, axis=1), axis=2).sum(axis=1)
    return np.column_stack([length, C.TURN_SCALE * np.abs(genomes[:, :, 0]).sum(axis=1)])


def d_gene_means(track, genomes, grid):
    return np.column_stack([C.GENE_SCALE * genomes[:, :, 0].mean(axis=1), C.GENE_SCALE * genomes[:, :, 1].mean(axis=1)])


def d_bfs(track, genomes, grid):
    true = M.bfs_distances(grid, M.cell_of(C.GOAL))
    idx = np.clip(_end(track).astype(int), 0, grid.shape[0] - 1)
    return true[idx[:, 0], idx[:, 1]][:, None]


def d_euclid(track, genomes, grid):
    end = _end(track)
    return np.hypot(end[:, 0] - C.GOAL[0], end[:, 1] - C.GOAL[1])[:, None]


FUNCTIONS = {"end": d_end, "end_heading": d_end_heading, "coarse": d_coarse, "mid": d_mid, "way4": d_way4, "way8": d_way8, "x": d_x, "y": d_y,
             "path_turn": d_path_turn, "gene_means": d_gene_means, "bfs": d_bfs, "euclid": d_euclid}


def describe(name, track, genomes, grid):
    """Verhaltensbeschreibung `name` für eine ganze Population: Feld (P, Dimension)."""
    if name not in FUNCTIONS:
        raise ValueError(f"unbekannte Beschreibung: {name}")
    return FUNCTIONS[name](track, genomes, grid)
