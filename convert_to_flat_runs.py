#!/usr/bin/env python3
"""
convert_to_flat_runs.py
  Multi-run/spill reco data → flat_runs_{up,down}.root

  Pure uproot + awkward implementation — no ROOT or TreeDataDict.so needed.
  Branch-reading logic mirrors convert_to_flat_uproot.py exactly.

  Data layout on cluster:
    RECO_DIR/run_XXXXXX/spill_YYYYYYYYY/out/output_PM.root

  Spin assignment (reco-20260512):
    UP  : runs 6111–6118, 6156
    DOWN: runs 6135–6139, 6149–6155

  Cuts applied (same as convert_to_flat_uproot.py / AnaDimuon.cc):
    1. FPGA bit-0 (MATRIX1) trigger
    2. Both track vertex z > -600 cm
    3. |y_st1| > 3 cm for both tracks at station 1
    4. chi2_tgt > 0, chi2_dump − chi2_tgt > 0, chi2_ups − chi2_tgt > 0 (pos & neg)
    5. Dimuon invariant mass: 0.5 ≤ M ≤ 10.0 GeV

  Output (written to --outdir, default ./data/):
    flat_runs_up.root
    flat_runs_down.root

Usage (on cluster — only needs python3 + uproot + awkward + numpy):
    python3 convert_to_flat_runs.py
    python3 convert_to_flat_runs.py --outdir /path/to/output
    python3 convert_to_flat_runs.py --spin up        # only spin-up
    python3 convert_to_flat_runs.py --spin down      # only spin-down
    python3 convert_to_flat_runs.py --reco-dir /other/reco/dir

Then copy back to your analysis machine:
    scp spinquestgpvm01:~/ana-spinquest-fit/data/flat_runs_*.root data/
"""

import argparse
import glob
import os

import awkward as ak
import numpy as np
import uproot

# ── Default paths ──────────────────────────────────────────────────────────────
RECO_DIR = "/pnfs/e1039/persistent/users/kenichi/RecoData2024/reco-20260512"

# ── Spin assignment ────────────────────────────────────────────────────────────
SPIN_UP_RUNS   = {6111, 6112, 6113, 6114, 6115, 6116, 6117, 6118, 6156}
SPIN_DOWN_RUNS = {6135, 6136, 6137, 6138, 6139,
                  6149, 6150, 6151, 6152, 6153, 6154, 6155}

# ── Cut thresholds ─────────────────────────────────────────────────────────────
M_MIN, M_MAX = 0.5, 10.0

# ── Branches to read (same keys as convert_to_flat_uproot.py) ─────────────────
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


# ── Kinematic helpers (identical to convert_to_flat_uproot.py) ─────────────────

def flat(jagged):
    return ak.to_numpy(ak.flatten(jagged))


def flat_lv(lv):
    """Unpack a TLorentzVector branch → (px, py, pz, E) numpy arrays."""
    return (
        flat(lv["fP"]["fX"]),
        flat(lv["fP"]["fY"]),
        flat(lv["fP"]["fZ"]),
        flat(lv["fE"]),
    )


def flat_v3(v3):
    """Unpack a TVector3 branch → (x, y, z) numpy arrays."""
    return flat(v3["fX"]), flat(v3["fY"]), flat(v3["fZ"])


def broadcast_event_to_dimuons(event_val, dimuon_jagged):
    """Repeat a per-event scalar to match the flat (per-dimuon) array length."""
    n_per_event = ak.to_numpy(ak.num(dimuon_jagged))
    return np.repeat(ak.to_numpy(event_val), n_per_event)


def pseudorapidity(px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p - pz > 0, 0.5 * np.log((p + pz) / (p - pz)), 0.0)


def rapidity(E, pz):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(E - pz > 0, 0.5 * np.log((E + pz) / (E - pz)), 0.0)


# ── File discovery ─────────────────────────────────────────────────────────────

def get_spill_files(spin: str, reco_dir: str) -> list:
    """Return sorted list of output_PM.root paths for the given spin state."""
    run_nums = SPIN_UP_RUNS if spin == "up" else SPIN_DOWN_RUNS
    files = []
    for run_num in sorted(run_nums):
        run_dir = os.path.join(reco_dir, f"run_{run_num:06d}")
        pattern = os.path.join(run_dir, "spill_*", "out", "output_PM.root")
        matched = sorted(glob.glob(pattern))
        n = len(matched)
        if n == 0:
            print(f"  WARNING: run {run_num:06d} — no spill files found")
            print(f"           (looked for: {pattern})")
        else:
            print(f"  run {run_num:06d}: {n:3d} spill files")
        files.extend(matched)
    return files


# ── Per-spin processing ────────────────────────────────────────────────────────

def process_spin(spin: str, output_path: str, reco_dir: str) -> None:
    """
    Read all spill files for one spin state, apply cuts, write flat tree.
    Uses uproot.concatenate to load all files in one vectorised pass.
    """
    print(f"\n── Collecting spin-{spin} spill files ────────────────────────")
    files = get_spill_files(spin, reco_dir)
    if not files:
        raise RuntimeError(
            f"No spill files found for spin={spin!r} under {reco_dir!r}")
    print(f"  Total: {len(files)} spill files")

    # Build the list of "path:treename" sources for concatenate
    sources = [f"{p}:tree" for p in files]

    print(f"\n── Reading & concatenating {len(files)} files … (may take a while)")
    arrays = uproot.concatenate(sources, expressions=_BRANCHES, library="ak")

    # ── Broadcast FPGA bits (event-level) to per-dimuon ───────────────────
    fpga = broadcast_event_to_dimuons(
        arrays["event/fpga_bits"], arrays["dimuon_list.mom_target"])

    # ── Flatten all dimuon-level vector branches ───────────────────────────
    px_d, py_d, pz_d, E_d = flat_lv(arrays["dimuon_list.mom_target"])
    px_p, py_p, pz_p, E_p = flat_lv(arrays["dimuon_list.mom_pos"])
    px_n, py_n, pz_n, E_n = flat_lv(arrays["dimuon_list.mom_neg"])
    _,    _,    z_vp       = flat_v3(arrays["dimuon_list.pos_pos"])
    _,    _,    z_vn       = flat_v3(arrays["dimuon_list.pos_neg"])
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
    print(f"  Dimuon candidates (raw):             {n_raw:,}")

    # ── Cuts ──────────────────────────────────────────────────────────────
    cut_fpga  = (fpga & 0x1) != 0
    cut_z     = (z_vp > -600.0) & (z_vn > -600.0)
    cut_y_st1 = (np.abs(y_st1_p) > 3.0) & (np.abs(y_st1_n) > 3.0)
    cut_chi2_p = (chi2_tgt_p > 0) & (chi2_dum_p - chi2_tgt_p > 0) & (chi2_ups_p - chi2_tgt_p > 0)
    cut_chi2_n = (chi2_tgt_n > 0) & (chi2_dum_n - chi2_tgt_n > 0) & (chi2_ups_n - chi2_tgt_n > 0)

    M    = np.sqrt(np.maximum(E_d**2 - (px_d**2 + py_d**2 + pz_d**2), 0.0))
    cut_mass = (M >= M_MIN) & (M <= M_MAX)

    cut_all = cut_fpga & cut_z & cut_y_st1 & cut_chi2_p & cut_chi2_n
    sel     = cut_all & cut_mass

    print(f"  After FPGA trigger:                  {int(cut_fpga.sum()):,}")
    print(f"  After Z_vertex > -600 cm:            {int((cut_fpga & cut_z).sum()):,}")
    print(f"  After |y_st1| > 3 cm:               {int((cut_fpga & cut_z & cut_y_st1).sum()):,}")
    print(f"  After chi2 cuts:                     {int(cut_all.sum()):,}")
    print(f"  After mass [{M_MIN}, {M_MAX}] GeV:   {int(sel.sum()):,}")

    # ── Derived kinematics ─────────────────────────────────────────────────
    pT_d = np.sqrt(px_d**2 + py_d**2)
    pT_p = np.sqrt(px_p**2 + py_p**2)
    pT_n = np.sqrt(px_n**2 + py_n**2)

    p3_p = np.sqrt(px_p**2 + py_p**2 + pz_p**2)
    p3_n = np.sqrt(px_n**2 + py_n**2 + pz_n**2)
    cos_alpha = (px_p*px_n + py_p*py_n + pz_p*pz_n) / (p3_p * p3_n + 1e-30)

    phi_p = np.arctan2(py_p, px_p)
    phi_n = np.arctan2(py_n, px_n)
    dphi  = phi_p - phi_n
    dphi  = dphi - 2 * np.pi * np.round(dphi / (2 * np.pi))
    eta_p = pseudorapidity(px_p, py_p, pz_p)
    eta_n = pseudorapidity(px_n, py_n, pz_n)

    # ── Write output ──────────────────────────────────────────────────────
    out = {
        "rec_dimu_y":           rapidity(E_d, pz_d)[sel],
        "rec_dimu_eta":         pseudorapidity(px_d, py_d, pz_d)[sel],
        "rec_dimu_E":           E_d[sel],
        "rec_dimu_px":          px_d[sel],
        "rec_dimu_py":          py_d[sel],
        "rec_dimu_pz":          pz_d[sel],
        "rec_dimu_M":           M[sel],
        "rec_dimu_mT":          np.sqrt(M**2 + pT_d**2)[sel],
        "rec_mu_theta_pos":     np.arctan(px_p / pz_p)[sel],
        "rec_mu_theta_neg":     np.arctan(px_n / pz_n)[sel],
        "rec_mu_open_angle":    np.arccos(np.clip(cos_alpha, -1.0, 1.0))[sel],
        "rec_mu_dpt":           (pT_p - pT_n)[sel],
        "rec_mu_Epos":          E_p[sel],
        "rec_mu_Eneg":          E_n[sel],
        "rec_track_pos_x_st1":  x_st1_p[sel],
        "rec_track_neg_x_st1":  x_st1_n[sel],
        "rec_track_pos_px_st1": px_st1_p[sel],
        "rec_track_neg_px_st1": px_st1_n[sel],
        "rec_track_pos_py_st1": py_st1_p[sel],
        "rec_track_neg_py_st1": py_st1_n[sel],
        "rec_dz_vtx":           (z_vp - z_vn)[sel],
        "rec_mu_deltaR":        np.sqrt((eta_p - eta_n)**2 + dphi**2)[sel],
    }

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with uproot.recreate(output_path) as fout:
        fout["tree"] = {k: v.astype(np.float64) for k, v in out.items()}
    print(f"  Written → {output_path}  ({int(sel.sum()):,} events)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten multi-run/spill reco data → flat ROOT trees (uproot-only).")
    parser.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        help="Output directory for flat_runs_{up,down}.root  (default: ./data/)")
    parser.add_argument(
        "--spin", choices=["up", "down", "both"], default="both",
        help="Which spin state to process  (default: both)")
    parser.add_argument(
        "--reco-dir", default=RECO_DIR,
        help=f"Path to reco-YYYYMMDD directory  (default: {RECO_DIR})")
    args = parser.parse_args()

    reco_dir = args.reco_dir   # plain local variable — no global mutation needed
    spins    = ["up", "down"] if args.spin == "both" else [args.spin]

    for spin in spins:
        out = os.path.join(args.outdir, f"flat_runs_{spin}.root")
        process_spin(spin, out, reco_dir)

    print("\nDone.")
    if len(spins) == 2:
        print("\nNext steps — copy flat files to your analysis machine:")
        print(f"  scp spinquestgpvm01:{args.outdir}/flat_runs_up.root   data/")
        print(f"  scp spinquestgpvm01:{args.outdir}/flat_runs_down.root data/")


if __name__ == "__main__":
    main()
