#!/usr/bin/env python3
"""
fit_mode_final.py — J/ψ simultaneous fit with additional track quality cuts.

Input: flat trees from convert_to_flat.py  (--data flat, default)
       or    convert_to_flat_runs.py        (--data runs).

Additional cuts applied with numpy BEFORE any RooFit:
  - x_st1 < 25 cm  for both the positive and negative muon tracks
  - py_st1_pos × py_st1_neg < 0  (opposite-sign py at station 1)

Per-panel selection then applied:
  - mass window: 2.0–5.9 GeV
  - dimuon px direction: right (px < 0) or left (px > 0)

Data source (--data):
  flat  (default) — data/flat_PM_{up,down}.root
  runs            — data/flat_runs_{up,down}.root  (convert_to_flat_runs.py)
"""

import argparse

from fit_common import PANELS, MASS_MIN, MASS_MAX, panel_suffix, fit_and_save
from data_loader import load_spin_data, output_path, add_data_arg

_BASE_OUTPUT = "/Users/spin/ana-spinquest-fit/fit/fit_mode_final.png"

BRANCHES = [
    "rec_dimu_M",
    "rec_dimu_px",
    "rec_track_pos_x_st1",
    "rec_track_neg_x_st1",
    "rec_track_pos_py_st1",
    "rec_track_neg_py_st1",
]


def load_mass_arrays(mode: str) -> dict:
    """
    Read flat trees, apply extra track cuts with numpy, then return
    per-panel arrays of dimuon mass ready for RooFit.
    """
    raw = load_spin_data(mode, BRANCHES)

    mass_arrays = {}
    for spin, side, _, _ in PANELS:
        arr = raw[spin]

        # --- extra cuts: pure numpy, before any RooFit ---
        x_cut  = (arr["rec_track_pos_x_st1"] < 25) & (arr["rec_track_neg_x_st1"] < 25)
        py_cut = arr["rec_track_pos_py_st1"] * arr["rec_track_neg_py_st1"] < 0
        keep   = x_cut & py_cut

        if side == "right":   # only print once per spin (first side)
            n_in  = len(arr["rec_dimu_M"])
            n_out = int(keep.sum())
            print(f"  [{spin}] {n_in:,} → {n_out:,} after x_st1 + py_st1 cuts")

        arr_cut  = {k: v[keep] for k, v in arr.items()}
        mass_cut = (arr_cut["rec_dimu_M"] > MASS_MIN) & (arr_cut["rec_dimu_M"] < MASS_MAX)
        px_cut   = arr_cut["rec_dimu_px"] < 0 if side == "right" else arr_cut["rec_dimu_px"] > 0

        mass_arrays[panel_suffix(spin, side)] = arr_cut["rec_dimu_M"][mass_cut & px_cut]

    return mass_arrays


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="J/ψ fit — base cuts + x_st1 + py_st1 track quality cuts")
    add_data_arg(parser)
    args = parser.parse_args()

    out = output_path(_BASE_OUTPUT, args.data)
    print(f"\n── Data mode: {args.data!r}  →  {out}")
    print("── Applying extra cuts ───────────────────────────────────────")
    mass_arrays = load_mass_arrays(args.data)
    fit_and_save(mass_arrays, out)
