#!/usr/bin/env python3
"""
fit_mode_final.py — J/ψ simultaneous fit with additional track quality cuts.

Input: flat trees from convert_to_flat.py.

Additional cuts applied with numpy BEFORE any RooFit:
  - x_st1 < 25 cm  for both the positive and negative muon tracks
  - py_st1_pos × py_st1_neg < 0  (opposite-sign py at station 1)

Per-panel selection then applied:
  - mass window: 2.0–5.9 GeV
  - dimuon px direction: right (px < 0) or left (px > 0)
"""

import numpy as np
import uproot

from fit_common import PANELS, MASS_MIN, MASS_MAX, panel_suffix, fit_and_save

FILE_UP   = "/Users/spin/ana-spinquest-fit/data/flat_PM_up.root"
FILE_DOWN = "/Users/spin/ana-spinquest-fit/data/flat_PM_down.root"
OUTPUT    = "/Users/spin/ana-spinquest-fit/fit/fit_mode_final.pdf"

BRANCHES = [
    "rec_dimu_M",
    "rec_dimu_px",
    "rec_track_pos_x_st1",
    "rec_track_neg_x_st1",
    "rec_track_pos_py_st1",
    "rec_track_neg_py_st1",
]


def load_mass_arrays():
    """
    Read flat trees, apply extra track cuts with numpy, then return
    per-panel arrays of dimuon mass ready for RooFit.
    """
    raw = {}
    for spin, path in [("up", FILE_UP), ("down", FILE_DOWN)]:
        with uproot.open(path) as f:
            arr = f["tree"].arrays(BRANCHES, library="np")

        # --- extra cuts: pure numpy, before any RooFit ---
        x_cut  = (arr["rec_track_pos_x_st1"] < 25) & (arr["rec_track_neg_x_st1"] < 25)
        py_cut = arr["rec_track_pos_py_st1"] * arr["rec_track_neg_py_st1"] < 0
        keep   = x_cut & py_cut

        n_in  = len(arr["rec_dimu_M"])
        n_out = int(keep.sum())
        print(f"  [{spin}] {n_in:,} → {n_out:,} after x_st1 + py_st1 cuts")

        raw[spin] = {k: v[keep] for k, v in arr.items()}

    mass_arrays = {}
    for spin, side, _, _ in PANELS:
        arr = raw[spin]

        mass_cut = (arr["rec_dimu_M"] > MASS_MIN) & (arr["rec_dimu_M"] < MASS_MAX)
        px_cut   = arr["rec_dimu_px"] < 0 if side == "right" else arr["rec_dimu_px"] > 0

        mass_arrays[panel_suffix(spin, side)] = arr["rec_dimu_M"][mass_cut & px_cut]

    return mass_arrays


if __name__ == "__main__":
    print("\n── Applying extra cuts ───────────────────────────────────────")
    mass_arrays = load_mass_arrays()
    fit_and_save(mass_arrays, OUTPUT)
