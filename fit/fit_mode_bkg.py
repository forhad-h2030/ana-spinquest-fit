#!/usr/bin/env python3
"""
fit_mode_bkg.py — J/ψ simultaneous fit, base cuts only.

Input: flat trees from convert_to_flat.py (FPGA, z_vtx, y_st1, chi2, mass cuts
       already applied during conversion).

Per-panel selection applied here (before RooFit):
  - mass window: 2.0–5.9 GeV
  - dimuon px direction: right (px < 0) or left (px > 0)
"""

import numpy as np
import uproot

from fit_common import PANELS, MASS_MIN, MASS_MAX, panel_suffix, fit_and_save

FILE_UP   = "/Users/spin/ana-spinquest-fit/data/flat_PM_up.root"
FILE_DOWN = "/Users/spin/ana-spinquest-fit/data/flat_PM_down.root"
OUTPUT    = "/Users/spin/ana-spinquest-fit/fit/fit_mode_bkg.pdf"


def load_mass_arrays():
    """Read flat trees and return per-panel arrays of dimuon mass."""
    raw = {}
    for spin, path in [("up", FILE_UP), ("down", FILE_DOWN)]:
        with uproot.open(path) as f:
            raw[spin] = f["tree"].arrays(["rec_dimu_M", "rec_dimu_px"], library="np")

    mass_arrays = {}
    for spin, side, _, _ in PANELS:
        arr = raw[spin]

        mass_cut = (arr["rec_dimu_M"] > MASS_MIN) & (arr["rec_dimu_M"] < MASS_MAX)
        px_cut   = arr["rec_dimu_px"] < 0 if side == "right" else arr["rec_dimu_px"] > 0

        mass_arrays[panel_suffix(spin, side)] = arr["rec_dimu_M"][mass_cut & px_cut]

    return mass_arrays


if __name__ == "__main__":
    print("\n── Loading data ──────────────────────────────────────────────")
    mass_arrays = load_mass_arrays()
    fit_and_save(mass_arrays, OUTPUT)
