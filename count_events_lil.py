#!/usr/bin/env python3
"""
count_events_lil.py — read mass from pre-cut TSSA file and write a flat output.

Input:  tssa_target_ITCuts_offset_x25_y3_ylikeSign_cut.root
        Four trees, cuts already applied:
          tree_up_left, tree_up_right, tree_down_left, tree_down_right

Output: flat_tssa_mass.root
        Same four trees, each containing only rec_dimu_M.

Usage:
    conda run -n root_env python3 count_events_lil.py
"""

import os
import numpy as np
import uproot
import awkward as ak

INPUT  = "data/tssa_target_ITCuts_offset_x25_y3_ylikeSign_cut.root"
OUTPUT = "data/flat_tssa_mass.root"

MASS_MIN, MASS_MAX = 2.0, 5.9   # must match fit_common.py

TREES = [
    ("up",   "left",  "tree_up_left"),
    ("up",   "right", "tree_up_right"),
    ("down", "left",  "tree_down_left"),
    ("down", "right", "tree_down_right"),
]

BRANCHES = ["m", "x_st1_pos", "x_st1_neg", "py_st1_pos", "py_st1_neg"]


def main():
    print(f"  Input : {INPUT}")
    print(f"  Output: {OUTPUT}\n")

    collected = {}

    with uproot.open(INPUT) as fin:
        for spin, _side, tree_name in TREES:
            tree   = fin[tree_name]
            arrays = tree.arrays(BRANCHES, library="ak")
            mass      = ak.to_numpy(arrays["m"])
            x_cut     = (ak.to_numpy(arrays["x_st1_pos"]) < 25) & \
                        (ak.to_numpy(arrays["x_st1_neg"]) < 25)
            py_cut    = ak.to_numpy(arrays["py_st1_pos"]) * \
                        ak.to_numpy(arrays["py_st1_neg"]) < 0
            sel       = x_cut & py_cut
            mass_sel  = mass[sel]

            n_total  = len(mass)
            n_after  = int(sel.sum())
            in_mass  = (mass_sel > MASS_MIN) & (mass_sel < MASS_MAX)
            n_mass   = int(in_mass.sum())

            print(f"  {tree_name:<20s}: {n_total:>6,} total"
                  f"  | after x_st1+py cuts: {n_after:>5,}"
                  f"  | mass [{MASS_MIN},{MASS_MAX}] GeV: {n_mass:>5,}")

            collected[tree_name] = mass_sel

    print()

    # ── totals by spin ────────────────────────────────────────────────────────
    for spin in ("up", "down"):
        names = [t for _, _, t in TREES if t.startswith(f"tree_{spin}")]
        combined = np.concatenate([collected[n] for n in names])
        in_mass  = (combined > MASS_MIN) & (combined < MASS_MAX)
        print(f"  spin {spin:<4s} TOTAL        : {len(combined):>6,} total  |"
              f"  mass [{MASS_MIN}, {MASS_MAX}] GeV : {int(in_mass.sum()):>6,}")

    grand = np.concatenate(list(collected.values()))
    in_mass = (grand > MASS_MIN) & (grand < MASS_MAX)
    print(f"  ALL   spins  TOTAL     : {len(grand):>6,} total  |"
          f"  mass [{MASS_MIN}, {MASS_MAX}] GeV : {int(in_mass.sum()):>6,}")
    print()

    # ── write flat output ─────────────────────────────────────────────────────
    out_dir = os.path.dirname(OUTPUT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with uproot.recreate(OUTPUT) as fout:
        for tree_name, mass in collected.items():
            fout[tree_name] = {"m": mass.astype(np.float64)}

    print(f"  Written → {OUTPUT}")


if __name__ == "__main__":
    main()
