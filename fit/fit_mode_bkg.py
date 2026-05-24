#!/usr/bin/env python3
"""
fit_mode_bkg.py — J/ψ simultaneous fit, base cuts only.

Input: flat trees from convert_to_flat.py (FPGA, z_vtx, y_st1, chi2, mass cuts
       already applied during conversion).

Per-panel selection applied here (before RooFit):
  - mass window: 2.0–5.9 GeV
  - dimuon px direction: right (px < 0) or left (px > 0)

Data source (--data):
  flat  (default) — data/flat_PM_{up,down}.root
  runs            — data/flat_runs_{up,down}.root  (convert_to_flat_runs.py)
"""

import argparse

from fit_common import PANELS, MASS_MIN, MASS_MAX, panel_suffix, fit_and_save
from data_loader import load_spin_data, output_path, add_data_arg

_BASE_OUTPUT = "/Users/spin/ana-spinquest-fit/fit/fit_mode_bkg.pdf"

BRANCHES = ["rec_dimu_M", "rec_dimu_px"]


def load_mass_arrays(mode: str) -> dict:
    """Read flat trees and return per-panel arrays of dimuon mass."""
    raw = load_spin_data(mode, BRANCHES)

    mass_arrays = {}
    for spin, side, _, _ in PANELS:
        arr = raw[spin]

        mass_cut = (arr["rec_dimu_M"] > MASS_MIN) & (arr["rec_dimu_M"] < MASS_MAX)
        px_cut   = arr["rec_dimu_px"] < 0 if side == "right" else arr["rec_dimu_px"] > 0

        mass_arrays[panel_suffix(spin, side)] = arr["rec_dimu_M"][mass_cut & px_cut]

    return mass_arrays


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="J/ψ fit — base cuts only")
    add_data_arg(parser)
    args = parser.parse_args()

    out = output_path(_BASE_OUTPUT, args.data)
    print(f"\n── Data mode: {args.data!r}  →  {out}")
    print("── Loading data ──────────────────────────────────────────────")
    mass_arrays = load_mass_arrays(args.data)
    fit_and_save(mass_arrays, out)
