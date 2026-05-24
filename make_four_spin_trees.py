#!/usr/bin/env python3
"""
make_four_spin_trees.py
  output_PM_{up,down}.root  →  data/flat_four_spin_states.root

Reads the raw nested trees (same as convert_to_flat_uproot.py) and applies
ALL cuts from that script plus the mode_final cuts, then splits into four
trees by spin × left/right so the result can be compared directly with
tssa_target_ITCuts_offset_x25_y3_ylikeSign_cut.root.

Cuts applied:
  1. FPGA bit-0 (MATRIX1) trigger
  2. Both track vertex z > -600 cm
  3. |y_st1| > 3 cm for both tracks at station 1
  4. chi2_tgt > 0, chi2_dump - chi2_tgt > 0, chi2_ups - chi2_tgt > 0 (pos & neg)
  5. x_st1 < 25 cm (both tracks)
  6. py_st1_pos * py_st1_neg < 0  (opposite y-sign at station 1)

No mass cut applied when filling the trees.

Left / right from dimuon px:
  right → px < 0
  left  → px > 0

Output trees:
  tree_up_right, tree_up_left, tree_down_right, tree_down_left

Output variables (all cut quantities + mass):
  rec_dimu_M,
  rec_dimu_px,
  z_vtx_pos,  z_vtx_neg,
  y_st1_pos,  y_st1_neg,
  x_st1_pos,  x_st1_neg,
  py_st1_pos, py_st1_neg,
  chi2_tgt_pos,  chi2_dump_pos,  chi2_ups_pos,
  chi2_tgt_neg,  chi2_dump_neg,  chi2_ups_neg

Usage:
    conda run -n root_env python3 make_four_spin_trees.py
"""

import os
import numpy as np
import uproot
import awkward as ak

FILES = [
    ("/Users/spin/SpinQuestAna/data/output_PM_up.root",   "up"),
    ("/Users/spin/SpinQuestAna/data/output_PM_down.root", "down"),
]
OUTPUT = "data/flat_four_spin_states.root"

_BRANCHES = [
    "event/fpga_bits",
    "dimuon_list.mom_target",
    "dimuon_list.pos_pos",
    "dimuon_list.pos_neg",
    "dimuon_list.pos_pos_st1",
    "dimuon_list.pos_neg_st1",
    "dimuon_list.mom_pos_st1",
    "dimuon_list.mom_neg_st1",
    "dimuon_list.chisq_target_pos",
    "dimuon_list.chisq_dump_pos",
    "dimuon_list.chisq_upstream_pos",
    "dimuon_list.chisq_target_neg",
    "dimuon_list.chisq_dump_neg",
    "dimuon_list.chisq_upstream_neg",
]


def flat(jagged):
    return ak.to_numpy(ak.flatten(jagged))


def flat_lv(lv):
    return (
        flat(lv["fP"]["fX"]),
        flat(lv["fP"]["fY"]),
        flat(lv["fP"]["fZ"]),
        flat(lv["fE"]),
    )


def flat_v3(v3):
    return flat(v3["fX"]), flat(v3["fY"]), flat(v3["fZ"])


def broadcast_event_to_dimuons(event_val, dimuon_jagged):
    n_per_event = ak.to_numpy(ak.num(dimuon_jagged))
    return np.repeat(ak.to_numpy(event_val), n_per_event)


def process(input_path, spin):
    print(f"\n── spin {spin}  ({input_path}) ───────────────────────────────")

    with uproot.open(input_path) as fin:
        tree   = fin["tree"]
        arrays = tree.arrays(_BRANCHES, library="ak")

    # ── flatten ───────────────────────────────────────────────────────────────
    fpga = broadcast_event_to_dimuons(
        arrays["event/fpga_bits"], arrays["dimuon_list.mom_target"]
    )

    px_d, py_d, pz_d, E_d = flat_lv(arrays["dimuon_list.mom_target"])
    _,    _,    z_vp       = flat_v3(arrays["dimuon_list.pos_pos"])
    _,    _,    z_vn       = flat_v3(arrays["dimuon_list.pos_neg"])
    x_st1_p, y_st1_p, _   = flat_v3(arrays["dimuon_list.pos_pos_st1"])
    x_st1_n, y_st1_n, _   = flat_v3(arrays["dimuon_list.pos_neg_st1"])
    _, py_st1_p, _, _      = flat_lv(arrays["dimuon_list.mom_pos_st1"])
    _, py_st1_n, _, _      = flat_lv(arrays["dimuon_list.mom_neg_st1"])

    chi2_tgt_p = flat(arrays["dimuon_list.chisq_target_pos"])
    chi2_dum_p = flat(arrays["dimuon_list.chisq_dump_pos"])
    chi2_ups_p = flat(arrays["dimuon_list.chisq_upstream_pos"])
    chi2_tgt_n = flat(arrays["dimuon_list.chisq_target_neg"])
    chi2_dum_n = flat(arrays["dimuon_list.chisq_dump_neg"])
    chi2_ups_n = flat(arrays["dimuon_list.chisq_upstream_neg"])

    # ── derived ───────────────────────────────────────────────────────────────
    p2 = px_d**2 + py_d**2 + pz_d**2
    M  = np.sqrt(np.maximum(E_d**2 - p2, 0.0))

    # ── cuts ──────────────────────────────────────────────────────────────────
    cut_fpga  = (fpga & 0x1) != 0
    cut_z     = (z_vp > -600.0) & (z_vn > -600.0)
    cut_y_st1 = (np.abs(y_st1_p) > 3.0) & (np.abs(y_st1_n) > 3.0)
    cut_chi2  = (
        (chi2_tgt_p > 0) & (chi2_dum_p - chi2_tgt_p > 0) & (chi2_ups_p - chi2_tgt_p > 0) &
        (chi2_tgt_n > 0) & (chi2_dum_n - chi2_tgt_n > 0) & (chi2_ups_n - chi2_tgt_n > 0)
    )
    cut_x_st1 = (x_st1_p < 25.9) & (x_st1_n < 25.9)
    cut_py    = py_st1_p * py_st1_n < 0

    sel = cut_fpga & cut_z & cut_y_st1 & cut_chi2 & cut_x_st1 & cut_py

    print(f"  raw dimuon candidates      : {len(M):>7,}")
    print(f"  after FPGA                 : {int(cut_fpga.sum()):>7,}")
    print(f"  after z_vtx > -600 cm      : {int((cut_fpga & cut_z).sum()):>7,}")
    print(f"  after |y_st1| > 3 cm       : {int((cut_fpga & cut_z & cut_y_st1).sum()):>7,}")
    print(f"  after chi2 cuts            : {int((cut_fpga & cut_z & cut_y_st1 & cut_chi2).sum()):>7,}")
    print(f"  after x_st1 < 25 cm        : {int((cut_fpga & cut_z & cut_y_st1 & cut_chi2 & cut_x_st1).sum()):>7,}")
    print(f"  after py sign cut (total)  : {int(sel.sum()):>7,}")
    in_mass = sel & (M > 2.0) & (M < 5.9)
    print(f"    mass [2.0, 5.9] GeV      : {int(in_mass.sum()):>7,}")

    # ── pack selected events ──────────────────────────────────────────────────
    out = {
        "rec_dimu_M":    M[sel],
        "rec_dimu_px":   px_d[sel],
        "z_vtx_pos":     z_vp[sel],
        "z_vtx_neg":     z_vn[sel],
        "y_st1_pos":     y_st1_p[sel],
        "y_st1_neg":     y_st1_n[sel],
        "x_st1_pos":     x_st1_p[sel],
        "x_st1_neg":     x_st1_n[sel],
        "py_st1_pos":    py_st1_p[sel],
        "py_st1_neg":    py_st1_n[sel],
        "chi2_tgt_pos":  chi2_tgt_p[sel],
        "chi2_dump_pos": chi2_dum_p[sel],
        "chi2_ups_pos":  chi2_ups_p[sel],
        "chi2_tgt_neg":  chi2_tgt_n[sel],
        "chi2_dump_neg": chi2_dum_n[sel],
        "chi2_ups_neg":  chi2_ups_n[sel],
    }

    # ── left / right split ────────────────────────────────────────────────────
    px_sel = out["rec_dimu_px"]
    right  = px_sel < 0
    left   = px_sel > 0

    mass_sel = out["rec_dimu_M"]
    print(f"    right (px < 0)           : {int(right.sum()):>7,}  | mass [2.0,5.9]: {int(((mass_sel<5.9)&(mass_sel>2.0)&right).sum()):>5,}")
    print(f"    left  (px > 0)           : {int(left.sum()):>7,}  | mass [2.0,5.9]: {int(((mass_sel<5.9)&(mass_sel>2.0)&left).sum()):>5,}")

    return (
        {k: v[right].astype(np.float64) for k, v in out.items()},
        {k: v[left].astype(np.float64)  for k, v in out.items()},
    )


def main():
    print(f"Output: {OUTPUT}")

    trees_out = {}
    for path, spin in FILES:
        right_data, left_data = process(path, spin)
        trees_out[f"tree_{spin}_right"] = right_data
        trees_out[f"tree_{spin}_left"]  = left_data

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with uproot.recreate(OUTPUT) as fout:
        for name, data in trees_out.items():
            fout[name] = data

    print(f"\n  Written → {OUTPUT}")
    print(f"  Trees  : {list(trees_out.keys())}")


if __name__ == "__main__":
    main()
