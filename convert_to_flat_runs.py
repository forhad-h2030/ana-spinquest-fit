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

  Cuts applied — Cut #1 (DocDB #11359 / Liliet):
    1. z_track > -600 cm  (both track vertices)
    2. |y_st1| > 3 cm  — SKIPPED (station-1 branches absent in reco-20260512)
    3. chi2_tgt > 0, chi2_dump − chi2_tgt > 0, chi2_ups − chi2_tgt > 0 (pos & neg)
    4. py_st1_pos * py_st1_neg < 0  — SKIPPED (st1 absent)
    5. x_st1_pos < 25 cm, x_st1_neg < 25 cm  — SKIPPED (st1 absent)
    6. Dimuon invariant mass: M_MIN ≤ M ≤ 10.0 GeV

  Optional (--road-matching flag):
    Road matching: (pos_top & neg_bot) OR (pos_bot & neg_top)

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
M_MIN, M_MAX = 0.0, 10.0

# ── Branches to read ──────────────────────────────────────────────────────────
# NOTE: in the reco-20260512 files the tree layout differs from output_PM_up/down.root:
#   - fpga_bits is a flat top-level branch (not nested under "event/")
#   - station-1 branches (pos_pos_st1, pos_neg_st1, mom_pos_st1, mom_neg_st1)
#     are NOT present → the |y_st1|>3 cm cut is skipped during conversion,
#     and fit_mode_final's x_st1/py_st1 cuts are not available for this dataset.
_BRANCHES = [
    "fpga_bits",                          # read but not used in current selection
    "dimuon_list.mom_target",
    "dimuon_list.mom_pos",                # positive muon 4-momentum (→ pz_pos)
    "dimuon_list.mom_neg",                # negative muon 4-momentum (→ pz_neg)
    "dimuon_list.pos",                    # dimuon vertex position   (→ z_dimuon)
    "dimuon_list.pos_pos",                # positive track vertex    (→ z_track_pos)
    "dimuon_list.pos_neg",                # negative track vertex    (→ z_track_neg)
    "dimuon_list.chisq_target_pos",
    "dimuon_list.chisq_dump_pos",
    "dimuon_list.chisq_upstream_pos",
    "dimuon_list.chisq_target_neg",
    "dimuon_list.chisq_dump_neg",
    "dimuon_list.chisq_upstream_neg",
    # road-matching (top/bottom detector halves per track)
    "dimuon_list.pos_top",
    "dimuon_list.pos_bot",
    "dimuon_list.neg_top",
    "dimuon_list.neg_bot",
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

def process_spin(spin: str, output_path: str, reco_dir: str,
                 road_matching: bool = False) -> None:
    """
    Read all spill files for one spin state, apply cuts, write flat tree.
    Uses uproot.concatenate to load all files in one vectorised pass.

    road_matching : if True, require tracks in opposite detector halves
                    (pos_top & neg_bot) OR (pos_bot & neg_top).
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

    # ── Flatten all dimuon-level vector branches ───────────────────────────
    px_d, py_d, pz_d, E_d = flat_lv(arrays["dimuon_list.mom_target"])
    px_p, py_p, pz_p, E_p = flat_lv(arrays["dimuon_list.mom_pos"])
    px_n, py_n, pz_n, E_n = flat_lv(arrays["dimuon_list.mom_neg"])
    _,    _,    z_dimu     = flat_v3(arrays["dimuon_list.pos"])      # dimuon vertex z
    _,    _,    z_vp       = flat_v3(arrays["dimuon_list.pos_pos"])  # track vertex z (pos)
    _,    _,    z_vn       = flat_v3(arrays["dimuon_list.pos_neg"])  # track vertex z (neg)
    # NOTE: station-1 branches not present in reco-20260512 → |y_st1|>3 cm skipped

    chi2_tgt_p = flat(arrays["dimuon_list.chisq_target_pos"])
    chi2_dum_p = flat(arrays["dimuon_list.chisq_dump_pos"])
    chi2_ups_p = flat(arrays["dimuon_list.chisq_upstream_pos"])
    chi2_tgt_n = flat(arrays["dimuon_list.chisq_target_neg"])
    chi2_dum_n = flat(arrays["dimuon_list.chisq_dump_neg"])
    chi2_ups_n = flat(arrays["dimuon_list.chisq_upstream_neg"])

    # road-matching booleans
    pos_top = flat(arrays["dimuon_list.pos_top"]).astype(bool)
    pos_bot = flat(arrays["dimuon_list.pos_bot"]).astype(bool)
    neg_top = flat(arrays["dimuon_list.neg_top"]).astype(bool)
    neg_bot = flat(arrays["dimuon_list.neg_bot"]).astype(bool)

    # FPGA trigger bits — event-level branch, broadcast to per-dimuon
    fpga_ev  = ak.to_numpy(ak.flatten(arrays["fpga_bits"], axis=None))
    n_dimu   = ak.to_numpy(ak.num(arrays["dimuon_list.mom_target"], axis=1))
    fpga     = np.repeat(fpga_ev, n_dimu)

    n_raw = len(E_d)
    print(f"  Dimuon candidates (raw):             {n_raw:,}")

    # ── Cuts — Cut #1 (DocDB #11359 / Liliet) ─────────────────────────────
    # Road matching: optional (--road-matching flag)
    cut_road  = (pos_top & neg_bot) | (pos_bot & neg_top)

    # 0. FPGA bit-0 (MATRIX1) trigger
    cut_fpga  = (fpga & 0x1) != 0

    # 1. z_track > -600 cm  (both individual track vertices)
    cut_z_trk = (z_vp > -600.0) & (z_vn > -600.0)

    # 2. |y_st1| > 3 cm — SKIPPED (station-1 branches absent in reco-20260512)
    # 4. py_st1_pos * py_st1_neg < 0 — SKIPPED (st1 absent)
    # 5. x_st1_pos < 25 cm, x_st1_neg < 25 cm — SKIPPED (st1 absent)

    # 3. chi2: tgt > 0 + difference cuts  (pos & neg muons)
    cut_chi2_p = (
        (chi2_tgt_p > 0) &
        (chi2_dum_p - chi2_tgt_p > 0) & (chi2_ups_p - chi2_tgt_p > 0)
    )
    cut_chi2_n = (
        (chi2_tgt_n > 0) &
        (chi2_dum_n - chi2_tgt_n > 0) & (chi2_ups_n - chi2_tgt_n > 0)
    )

    M        = np.sqrt(np.maximum(E_d**2 - (px_d**2 + py_d**2 + pz_d**2), 0.0))
    cut_mass = (M >= M_MIN) & (M <= M_MAX)

    # base selection (always applied)
    cut_base = cut_fpga & cut_z_trk & cut_chi2_p & cut_chi2_n

    # road matching: optional for comparison studies
    cut_all = cut_base & cut_road if road_matching else cut_base
    sel     = cut_all & cut_mass

    # ── Detailed cut-flow (cumulative, one cut at a time) ─────────────────
    def cumshow(label, mask, c_in, applied=True):
        c_out = (c_in & mask) if applied else c_in
        tag   = "" if applied else "  [NOT APPLIED]"
        removed = int(c_in.sum()) - int(c_out.sum())
        print(f"  {label:<44s} {int(c_out.sum()):>7,}  (-{removed:,}){tag}")
        return c_out

    print(f"\n  {'cut':<44s} {'surviving':>7}  removed")
    print(f"  {'-'*65}")
    print(f"  {'(raw dimuon candidates)':<44s} {n_raw:>7,}")
    c = np.ones(n_raw, dtype=bool)
    c = cumshow("road matching",              cut_road,   c, applied=road_matching)
    c = cumshow("FPGA bit-0 (MATRIX1)",       cut_fpga,   c)
    c = cumshow("z_track > -600 cm",          cut_z_trk,  c)
    c = cumshow("|y_st1| > 3 cm",             c,          c, applied=False)
    c = cumshow("py_st1_pos*py_st1_neg < 0",  c,          c, applied=False)
    c = cumshow("x_st1 < 25 cm",              c,          c, applied=False)
    c = cumshow("chi2 cuts (pos muon)",        cut_chi2_p, c)
    c = cumshow("chi2 cuts (neg muon)",        cut_chi2_n, c)
    c = cumshow(f"mass in [{M_MIN}, {M_MAX}] GeV", cut_mass, c)
    print(f"  {'-'*65}")
    print(f"  {'FINAL selected':<44s} {int(sel.sum()):>7,}")

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
    # NOTE: rec_track_*_st1 columns are omitted — station-1 data not available
    # in reco-20260512. fit_mode_final's x_st1/py_st1 quality cuts are therefore
    # not applied when using --data runs; use fit_mode_bkg for base-cut fits.
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
        "rec_dimu_z_vtx":       z_dimu[sel],
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
    parser.add_argument(
        "--road-matching", action="store_true",
        help="Apply road-matching cut: require tracks in opposite detector halves "
             "(pos_top & neg_bot) OR (pos_bot & neg_top).  Off by default.")
    args = parser.parse_args()

    reco_dir = args.reco_dir
    spins    = ["up", "down"] if args.spin == "both" else [args.spin]

    # encode road-matching in the output filename so results don't collide
    suffix = "_rm" if args.road_matching else ""

    for spin in spins:
        out = os.path.join(args.outdir, f"flat_runs_{spin}{suffix}.root")
        process_spin(spin, out, reco_dir, road_matching=args.road_matching)

    print("\nDone.")
    if len(spins) == 2:
        print("\nNext steps — copy flat files to your analysis machine:")
        print(f"  scp spinquestgpvm01:{args.outdir}/flat_runs_up{suffix}.root   data/")
        print(f"  scp spinquestgpvm01:{args.outdir}/flat_runs_down{suffix}.root data/")


if __name__ == "__main__":
    main()
