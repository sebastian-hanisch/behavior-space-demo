"""Rechnet die Studie vor (Build-Zeit, nicht in der App): `python generate_precomputed.py [Prozesse]` schreibt `precomputed_sweep.json`.

  levels  Stufe 3 und 4, je Auswahl (Fitness als Bezug und zwölf Verhaltensbeschreibungen mit Novelty ohne Archiv): 20 Läufe à 400 Generationen × 100 Individuen = 40 000 Evaluationen:
          Evaluationen bis zum ersten Erfolg (None = nie), besuchte Zellen und kleinster wahrer Restweg am Ende, mittlerer Verlauf beider Größen; dazu die exakten Täuschungsmaße der Stufe
  length  Stufe 4, Länge der Steuerfolge 140 / 180 / 220 / 300 Schritte, Auswahl Fitness und Novelty auf der Endposition: je 20 Läufe à 400 Generationen"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import bs_constants as C
import bs_evaluation as E
import bs_maze as M
import bs_search as S


def _level_task(args):
    level, selection, seed = args
    res = S.run_search(M.make_maze(level), seed, selection, gens=C.STUDY_GENS, keep_positions=False)
    return level, selection, seed, res.first_solved, res.visited_cells, res.best_progress


def _length_task(args):
    selection, steps, seed = args
    old = C.STEPS
    C.STEPS = steps                                      # Genomlänge gilt je Prozessaufruf; danach zurück
    try:
        res = S.run_search(M.make_maze(4), seed, selection, gens=C.STUDY_GENS, keep_positions=False)
    finally:
        C.STEPS = old
    return selection, steps, seed, res.first_solved, res.visited_cells, res.best_progress


def _curve(rows, index):
    arr = np.mean([r[index] for r in rows], axis=0)
    return [round(float(x), 3) for x in arr]


def main(workers):
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        level_res = list(ex.map(_level_task, [(lv, sel, s) for lv in C.STUDY_LEVELS for sel in E.SELECTIONS for s in range(C.STUDY_SEEDS)], chunksize=1))
        print(f"Beschreibungen fertig nach {time.time() - t0:.0f} s", flush=True)
        length_res = list(ex.map(_length_task, [(sel, st, s) for sel in C.LENGTH_SELECTIONS for st in C.LENGTH_STEPS for s in range(C.STUDY_SEEDS)], chunksize=1))
    levels = {}
    for lv in C.STUDY_LEVELS:
        selections = {}
        for sel in E.SELECTIONS:
            rows = sorted([r for r in level_res if r[0] == lv and r[1] == sel], key=lambda r: r[2])
            selections[sel] = {"runs": [r[3] for r in rows], "cells": [r[4][-1] for r in rows], "progress": [r[5][-1] for r in rows], "mean_visited": _curve(rows, 4), "mean_progress": _curve(rows, 5)}
        levels[str(lv)] = {"metrics": E.level_metrics(lv), "selections": selections}
    length = {}
    for sel in C.LENGTH_SELECTIONS:
        length[sel] = {}
        for st in C.LENGTH_STEPS:
            rows = sorted([r for r in length_res if r[0] == sel and r[1] == st], key=lambda r: r[2])
            length[sel][str(st)] = {"runs": [r[3] for r in rows], "cells": [r[4][-1] for r in rows], "progress": [r[5][-1] for r in rows]}
    data = {"seeds": C.STUDY_SEEDS, "budget": C.STUDY_BUDGET, "levels": levels, "length": {"steps": list(C.LENGTH_STEPS), "cells": length}}
    E.PRECOMPUTED_PATH.write_text(json.dumps(data), encoding="utf-8")
    print(f"fertig in {time.time() - t0:.0f} s -> {E.PRECOMPUTED_PATH}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else min(14, os.cpu_count() or 1))
