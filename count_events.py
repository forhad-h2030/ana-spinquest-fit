#!/usr/bin/env python3
"""
count_events.py — cut-flow event counts per spin state and left/right category.

mode_bkg:   base cuts (already in flat trees) + mass window + px direction
mode_final: mode_bkg + x_st1 < 25 cm (both tracks) + py_st1_pos * py_st1_neg < 0
"""

import uproot

FILE_UP   = "/Users/spin/ana-spinquest-fit/data/flat_PM_up.root"
FILE_DOWN = "/Users/spin/ana-spinquest-fit/data/flat_PM_down.root"

MASS_MIN, MASS_MAX = 2.0, 5.9   # must match fit_common.py

BRANCHES = [
    "rec_dimu_M",
    "rec_dimu_px",
    "rec_track_pos_x_st1",
    "rec_track_neg_x_st1",
    "rec_track_pos_py_st1",
    "rec_track_neg_py_st1",
]

# ── load ──────────────────────────────────────────────────────────────────────
data = {}
for spin, path in [("up", FILE_UP), ("down", FILE_DOWN)]:
    with uproot.open(path) as f:
        data[spin] = f["tree"].arrays(BRANCHES, library="np")

# ── mode_bkg ──────────────────────────────────────────────────────────────────
print("── mode_bkg ──────────────────────────────────────────────────────")
print("   cuts: FPGA + z_vtx + y_st1 + chi2 (flat tree) + mass window + px")
print()

for spin in ("up", "down"):
    arr  = data[spin]
    mass = arr["rec_dimu_M"]
    px   = arr["rec_dimu_px"]

    in_mass = (mass > MASS_MIN) & (mass < MASS_MAX)

    print(f"  spin {spin}  ({len(mass):,} events in flat tree)")
    print(f"    mass [{MASS_MIN}, {MASS_MAX}] GeV : {int(in_mass.sum()):>6,}")
    print(f"      right  (px < 0)        : {int((in_mass & (px < 0)).sum()):>6,}")
    print(f"      left   (px > 0)        : {int((in_mass & (px > 0)).sum()):>6,}")
    print()

# ── mode_final ────────────────────────────────────────────────────────────────
print("── mode_final ────────────────────────────────────────────────────")
print("   cuts: mode_bkg + x_st1 < 25 cm (both tracks)")
print("                  + py_st1_pos * py_st1_neg < 0")
print()

for spin in ("up", "down"):
    arr  = data[spin]
    mass = arr["rec_dimu_M"]
    px   = arr["rec_dimu_px"]

    x_cut  = (arr["rec_track_pos_x_st1"] < 25) & (arr["rec_track_neg_x_st1"] < 25)
    py_cut = arr["rec_track_pos_py_st1"] * arr["rec_track_neg_py_st1"] < 0
    extra  = x_cut & py_cut
    in_mass = (mass > MASS_MIN) & (mass < MASS_MAX) & extra

    print(f"  spin {spin}  ({len(mass):,} events in flat tree)")
    print(f"    after x_st1 + py_st1 cuts : {int(extra.sum()):>6,}")
    print(f"    mass [{MASS_MIN}, {MASS_MAX}] GeV     : {int(in_mass.sum()):>6,}")
    print(f"      right  (px < 0)         : {int((in_mass & (px < 0)).sum()):>6,}")
    print(f"      left   (px > 0)         : {int((in_mass & (px > 0)).sum()):>6,}")
    print()
