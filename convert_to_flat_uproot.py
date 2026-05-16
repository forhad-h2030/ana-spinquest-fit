#!/usr/bin/env python3
"""
convert_to_flat_uproot.py
  output_PM_{down,up}.root → flat_{down,up}.root  (uproot/awkward version)

Reads the nested DimuonData/EventData tree in batch via uproot + awkward arrays.
Physics quantities are computed with numpy formulas.

Cuts applied (matching AnaDimuon.cc logic):
  1. FPGA bit-0 (MATRIX1) trigger
  2. Both track vertex z > -600 cm
  3. |y_st1| > 3 cm for both tracks at station 1
  4. chi2_tgt > 0, chi2_dump - chi2_tgt > 0, chi2_ups - chi2_tgt > 0 (pos & neg)
  5. Dimuon invariant mass: 0.5 ≤ M ≤ 10.0 GeV

Usage:
    conda run -n root_env python3 convert_to_flat_uproot.py
"""

import os
import numpy as np
import uproot
import awkward as ak

FILES = [
    (
        "/Users/spin/SpinQuestAna/data/output_PM_down.root",
        "/Users/spin/ana-spinquest-fit/data/flat_PM_down.root",
    ),
    (
        "/Users/spin/SpinQuestAna/data/output_PM_up.root",
        "/Users/spin/ana-spinquest-fit/data/flat_PM_up.root",
    ),
]

_BRANCHES = [
    "event/fpga_bits",
    "dimuon_list.mom_target",
    "dimuon_list.mom_pos",
    "dimuon_list.mom_neg",
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
    """Repeat a per-event array to match the flat dimuon list."""
    n_per_event = ak.to_numpy(ak.num(dimuon_jagged))
    return np.repeat(ak.to_numpy(event_val), n_per_event)


def pseudorapidity(px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p - pz > 0, 0.5 * np.log((p + pz) / (p - pz)), 0.0)


def rapidity(E, pz):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(E - pz > 0, 0.5 * np.log((E + pz) / (E - pz)), 0.0)


def process_file(input_path: str, output_path: str) -> None:
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")

    with uproot.open(input_path) as fin:
        tree = fin["tree"]
        print(f"  Entries in tree: {tree.num_entries:,}")
        arrays = tree.arrays(_BRANCHES, library="ak")

    # ── Broadcast event-level FPGA bits to per-dimuon ─────────────────────
    fpga = broadcast_event_to_dimuons(
        arrays["event/fpga_bits"], arrays["dimuon_list.mom_target"]
    )

    # ── Flatten all dimuon-level branches ─────────────────────────────────
    px_d, py_d, pz_d, E_d = flat_lv(arrays["dimuon_list.mom_target"])
    px_p, py_p, pz_p, E_p = flat_lv(arrays["dimuon_list.mom_pos"])
    px_n, py_n, pz_n, E_n = flat_lv(arrays["dimuon_list.mom_neg"])
    _, _, z_vp             = flat_v3(arrays["dimuon_list.pos_pos"])
    _, _, z_vn             = flat_v3(arrays["dimuon_list.pos_neg"])
    x_st1_p, y_st1_p, _   = flat_v3(arrays["dimuon_list.pos_pos_st1"])
    x_st1_n, y_st1_n, _   = flat_v3(arrays["dimuon_list.pos_neg_st1"])
    px_st1_p, py_st1_p, _, _ = flat_lv(arrays["dimuon_list.mom_pos_st1"])
    px_st1_n, py_st1_n, _, _ = flat_lv(arrays["dimuon_list.mom_neg_st1"])

    chi2_tgt_p = flat(arrays["dimuon_list.chisq_target_pos"])
    chi2_dum_p = flat(arrays["dimuon_list.chisq_dump_pos"])
    chi2_ups_p = flat(arrays["dimuon_list.chisq_upstream_pos"])
    chi2_tgt_n = flat(arrays["dimuon_list.chisq_target_neg"])
    chi2_dum_n = flat(arrays["dimuon_list.chisq_dump_neg"])
    chi2_ups_n = flat(arrays["dimuon_list.chisq_upstream_neg"])

    n_raw = len(E_d)
    print(f"  Dimuon candidates (raw):          {n_raw:,}")

    # ── Cuts ──────────────────────────────────────────────────────────────
    # 1. FPGA bit-0 (MATRIX1) trigger
    cut_fpga = (fpga & 0x1) != 0

    # 2. Both track vertex z > -600 cm
    cut_z_trk = (z_vp > -600.0) & (z_vn > -600.0)

    # 3. |y_st1| > 3 cm for both tracks at station 1
    cut_y_st1 = (np.abs(y_st1_p) > 3.0) & (np.abs(y_st1_n) > 3.0)

    # 4. Chi-squared origin cuts (target is best vertex)
    cut_chi2_p = (
        (chi2_tgt_p > 0) &
        (chi2_dum_p - chi2_tgt_p > 0) &
        (chi2_ups_p - chi2_tgt_p > 0)
    )
    cut_chi2_n = (
        (chi2_tgt_n > 0) &
        (chi2_dum_n - chi2_tgt_n > 0) &
        (chi2_ups_n - chi2_tgt_n > 0)
    )

    # 5. Dimuon invariant mass
    M_MIN, M_MAX = 0.5, 10.0
    p2_d = px_d**2 + py_d**2 + pz_d**2
    M    = np.sqrt(np.maximum(E_d**2 - p2_d, 0.0))
    cut_mass = (M >= M_MIN) & (M <= M_MAX)

    cut_all = cut_fpga & cut_z_trk & cut_y_st1 & cut_chi2_p & cut_chi2_n
    sel     = cut_all & cut_mass

    print(f"  After FPGA trigger:               {int(cut_fpga.sum()):,}")
    print(f"  After Z_vertex > -600 cm:         {int((cut_fpga & cut_z_trk).sum()):,}")
    print(f"  After |y_st1| > 3 cm:             {int((cut_fpga & cut_z_trk & cut_y_st1).sum()):,}")
    print(f"  After chi2 cuts:                  {int(cut_all.sum()):,}")
    print(f"  After mass [{M_MIN}, {M_MAX}] GeV: {int(sel.sum()):,}")

    # ── Derived kinematics (computed on all, indexed with sel when writing) ─
    pT_d = np.sqrt(px_d**2 + py_d**2)
    rec_dimu_y   = rapidity(E_d, pz_d)
    rec_dimu_eta = pseudorapidity(px_d, py_d, pz_d)
    rec_dimu_mT  = np.sqrt(M**2 + pT_d**2)

    pT_p             = np.sqrt(px_p**2 + py_p**2)
    rec_mu_theta_pos = np.arctan(px_p / pz_p)   # horizontal bending angle
    eta_p            = pseudorapidity(px_p, py_p, pz_p)
    phi_p            = np.arctan2(py_p, px_p)

    pT_n             = np.sqrt(px_n**2 + py_n**2)
    rec_mu_theta_neg = np.arctan(px_n / pz_n)   # horizontal bending angle
    eta_n            = pseudorapidity(px_n, py_n, pz_n)
    phi_n            = np.arctan2(py_n, px_n)

    p3_p      = np.sqrt(px_p**2 + py_p**2 + pz_p**2)
    p3_n      = np.sqrt(px_n**2 + py_n**2 + pz_n**2)
    cos_alpha = (px_p*px_n + py_p*py_n + pz_p*pz_n) / (p3_p * p3_n + 1e-30)
    rec_mu_open_angle = np.arccos(np.clip(cos_alpha, -1.0, 1.0))

    rec_mu_dpt = pT_p - pT_n

    dphi = phi_p - phi_n
    dphi = dphi - 2 * np.pi * np.round(dphi / (2 * np.pi))
    rec_mu_deltaR = np.sqrt((eta_p - eta_n)**2 + dphi**2)

    rec_dz_vtx = z_vp - z_vn

    # ── Write output ──────────────────────────────────────────────────────
    out = {
        "rec_dimu_y":           rec_dimu_y[sel],
        "rec_dimu_eta":         rec_dimu_eta[sel],
        "rec_dimu_E":           E_d[sel],
        "rec_dimu_px":          px_d[sel],
        "rec_dimu_py":          py_d[sel],
        "rec_dimu_pz":          pz_d[sel],
        "rec_dimu_M":           M[sel],
        "rec_dimu_mT":          rec_dimu_mT[sel],
        "rec_mu_theta_pos":     rec_mu_theta_pos[sel],
        "rec_mu_theta_neg":     rec_mu_theta_neg[sel],
        "rec_mu_open_angle":    rec_mu_open_angle[sel],
        "rec_mu_dpt":           rec_mu_dpt[sel],
        "rec_mu_Epos":          E_p[sel],
        "rec_mu_Eneg":          E_n[sel],
        "rec_track_pos_x_st1":  x_st1_p[sel],
        "rec_track_neg_x_st1":  x_st1_n[sel],
        "rec_track_pos_px_st1": px_st1_p[sel],
        "rec_track_neg_px_st1": px_st1_n[sel],
        "rec_track_pos_py_st1": py_st1_p[sel],
        "rec_track_neg_py_st1": py_st1_n[sel],
        "rec_dz_vtx":           rec_dz_vtx[sel],
        "rec_mu_deltaR":        rec_mu_deltaR[sel],
    }

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with uproot.recreate(output_path) as fout:
        fout["tree"] = {k: v.astype(np.float64) for k, v in out.items()}

    print(f"  Written → {output_path}")


def main():
    for inp, out in FILES:
        print(f"\n── {os.path.basename(inp)} ──────────────────────────────")
        process_file(inp, out)


if __name__ == "__main__":
    main()
