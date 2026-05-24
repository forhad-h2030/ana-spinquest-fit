#!/usr/bin/env python3
"""
convert_to_flat_runs.py
  Multi-run/spill reco data → flat_runs_{up,down}.root

  Data layout on cluster:
    RECO_DIR/run_XXXXXX/spill_YYYYYYYYY/out/output_PM.root

  Spin assignment (reco-20260512):
    UP  : runs 6111-6118, 6156
    DOWN: runs 6135-6139, 6149-6155

  Applies exactly the same cuts as convert_to_flat.py:
    1. FPGA bit-0 (MATRIX1) trigger
    2. Both track vertex z > -600 cm
    3. |y_st1| > 3 cm for both tracks at station 1
    4. chi2_tgt > 0, chi2_dump - chi2_tgt > 0, chi2_ups - chi2_tgt > 0 (pos & neg)
    5. Dimuon invariant mass: 0.1 ≤ M ≤ 10.0 GeV

  Output (written to --outdir, default ./data/):
    flat_runs_up.root
    flat_runs_down.root

Usage (on cluster, inside the root_env conda environment):
    conda run -n root_env python3 convert_to_flat_runs.py
    conda run -n root_env python3 convert_to_flat_runs.py --outdir /path/to/output
    conda run -n root_env python3 convert_to_flat_runs.py --spin up   # only spin-up
    conda run -n root_env python3 convert_to_flat_runs.py --spin down # only spin-down

Then copy the output files back to your analysis machine:
    scp spinquestgpvm01:path/to/flat_runs_*.root data/
"""

import argparse
import glob
import math
import os

import numpy as np
import uproot
import ROOT

# ── Paths ─────────────────────────────────────────────────────────────────────
LIB_SO   = os.path.join(os.path.dirname(__file__), "lib", "TreeDataDict.so")
RECO_DIR = "/pnfs/e1039/persistent/users/kenichi/RecoData2024/reco-20260512"

ROOT.gROOT.SetBatch(True)
ROOT.gSystem.Load(LIB_SO)

# ── Spin assignment ────────────────────────────────────────────────────────────
SPIN_UP_RUNS   = {6111, 6112, 6113, 6114, 6115, 6116, 6117, 6118, 6156}
SPIN_DOWN_RUNS = {6135, 6136, 6137, 6138, 6139,
                  6149, 6150, 6151, 6152, 6153, 6154, 6155}

# ── Cut thresholds (match convert_to_flat.py) ─────────────────────────────────
MASS_MIN, MASS_MAX = 0.1, 10.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_spill_files(spin: str) -> list[str]:
    """Return sorted list of output_PM.root paths for the given spin."""
    run_nums = SPIN_UP_RUNS if spin == "up" else SPIN_DOWN_RUNS
    files = []
    for run_num in sorted(run_nums):
        run_dir = os.path.join(RECO_DIR, f"run_{run_num:06d}")
        pattern = os.path.join(run_dir, "spill_*", "out", "output_PM.root")
        matched = sorted(glob.glob(pattern))
        n = len(matched)
        if n == 0:
            print(f"  WARNING: run {run_num:06d} — no spill files found at {pattern}")
        else:
            print(f"  run {run_num:06d}: {n:3d} spill files")
        files.extend(matched)
    return files


def _empty_cols() -> dict:
    return {
        "rec_dimu_y":           [],
        "rec_dimu_eta":         [],
        "rec_dimu_E":           [],
        "rec_dimu_px":          [],
        "rec_dimu_py":          [],
        "rec_dimu_pz":          [],
        "rec_dimu_M":           [],
        "rec_dimu_mT":          [],
        "rec_mu_theta_pos":     [],
        "rec_mu_theta_neg":     [],
        "rec_mu_open_angle":    [],
        "rec_mu_dpt":           [],
        "rec_mu_Epos":          [],
        "rec_mu_Eneg":          [],
        "rec_track_pos_x_st1":  [],
        "rec_track_neg_x_st1":  [],
        "rec_track_pos_px_st1": [],
        "rec_track_neg_px_st1": [],
        "rec_track_pos_py_st1": [],
        "rec_track_neg_py_st1": [],
        "rec_dz_vtx":           [],
        "rec_mu_deltaR":        [],
    }


def _process_one_spill(input_path: str, cols: dict, ctrs: dict) -> None:
    """
    Open one spill file and accumulate passing dimuon candidates into cols.
    Mirrors the event loop in convert_to_flat.py exactly.
    """
    fin = ROOT.TFile.Open(input_path, "READ")
    if not fin or fin.IsZombie():
        print(f"  WARNING: cannot open {input_path!r} — skipping")
        return
    tree = fin.Get("tree")
    if not tree:
        fin.Close()
        print(f"  WARNING: no 'tree' object in {input_path!r} — skipping")
        return

    n_entries = int(tree.GetEntries())

    for i_ev in range(n_entries):
        tree.GetEntry(i_ev)
        ev    = tree.event
        dlist = tree.dimuon_list
        nd = dlist.size()
        if nd == 0:
            continue
        ctrs["raw"] += nd

        # 1. FPGA bit-0 (MATRIX1) — event-level
        if not (ev.fpga_bits & 1):
            continue
        ctrs["fpga"] += nd

        for j in range(nd):
            dd = dlist[j]

            # 2. Both track vertex z > -600 cm
            z_vp = dd.pos_pos.Z()
            z_vn = dd.pos_neg.Z()
            if z_vp <= -600.0 or z_vn <= -600.0:
                continue
            ctrs["zcut"] += 1

            # 3. |y_st1| > 3 cm for both tracks
            if abs(dd.pos_pos_st1.Y()) <= 3.0 or abs(dd.pos_neg_st1.Y()) <= 3.0:
                continue
            ctrs["yst1"] += 1

            # 4. Chi-squared origin cuts (target is best vertex)
            tp = dd.chisq_target_pos;  dp = dd.chisq_dump_pos;  up_ = dd.chisq_upstream_pos
            tn = dd.chisq_target_neg;  dn = dd.chisq_dump_neg;  un  = dd.chisq_upstream_neg
            if tp <= 0 or (dp - tp) <= 0 or (up_ - tp) <= 0:
                continue
            if tn <= 0 or (dn - tn) <= 0 or (un  - tn) <= 0:
                continue
            ctrs["chi2"] += 1

            # 5. Invariant mass
            dimu = dd.mom_target          # TLorentzVector
            M    = dimu.M()
            if M < MASS_MIN or M > MASS_MAX:
                continue
            ctrs["mass"] += 1

            # ── Derived kinematics (same as convert_to_flat.py) ──────────────
            mu_p = dd.mom_pos
            mu_n = dd.mom_neg

            cols["rec_dimu_y"].append(dimu.Rapidity())
            cols["rec_dimu_eta"].append(dimu.Eta())
            cols["rec_dimu_E"].append(dimu.E())
            cols["rec_dimu_px"].append(dimu.Px())
            cols["rec_dimu_py"].append(dimu.Py())
            cols["rec_dimu_pz"].append(dimu.Pz())
            cols["rec_dimu_M"].append(M)
            cols["rec_dimu_mT"].append(dimu.Mt())

            cols["rec_mu_theta_pos"].append(math.atan2(mu_p.Pt(), mu_p.Pz()))
            cols["rec_mu_theta_neg"].append(math.atan2(mu_n.Pt(), mu_n.Pz()))
            cols["rec_mu_Epos"].append(mu_p.E())
            cols["rec_mu_Eneg"].append(mu_n.E())
            cols["rec_mu_dpt"].append(mu_p.Pt() - mu_n.Pt())

            vp    = mu_p.Vect()
            vn    = mu_n.Vect()
            denom = vp.Mag() * vn.Mag()
            cos_a = vp.Dot(vn) / denom if denom > 0 else 1.0
            cols["rec_mu_open_angle"].append(math.acos(max(-1.0, min(1.0, cos_a))))

            dphi = mu_p.Phi() - mu_n.Phi()
            dphi = dphi - 2.0 * math.pi * round(dphi / (2.0 * math.pi))
            deta = mu_p.Eta() - mu_n.Eta()
            cols["rec_mu_deltaR"].append(math.sqrt(deta * deta + dphi * dphi))

            cols["rec_track_pos_x_st1"].append(dd.pos_pos_st1.X())
            cols["rec_track_neg_x_st1"].append(dd.pos_neg_st1.X())
            cols["rec_track_pos_px_st1"].append(dd.mom_pos_st1.Px())
            cols["rec_track_neg_px_st1"].append(dd.mom_neg_st1.Px())
            cols["rec_track_pos_py_st1"].append(dd.mom_pos_st1.Py())
            cols["rec_track_neg_py_st1"].append(dd.mom_neg_st1.Py())
            cols["rec_dz_vtx"].append(z_vp - z_vn)

    fin.Close()


def process_spin(spin: str, output_path: str) -> None:
    """Collect all spill files for one spin state, apply cuts, write flat tree."""
    print(f"\n── Collecting spin-{spin} spill files ────────────────────────")
    files = get_spill_files(spin)
    if not files:
        raise RuntimeError(
            f"No spill files found for spin={spin!r} under {RECO_DIR!r}")
    print(f"  Total: {len(files)} spill files")

    cols = _empty_cols()
    ctrs = {"raw": 0, "fpga": 0, "zcut": 0, "yst1": 0, "chi2": 0, "mass": 0}

    print(f"\n── Processing spills ─────────────────────────────────────────")
    n = len(files)
    for i, fpath in enumerate(files, 1):
        if i == 1 or i % 100 == 0 or i == n:
            # show run/spill in progress indicator
            parts = fpath.split(os.sep)
            label = "/".join(parts[-4:-1])  # run_XXXXXX/spill_YYY/out
            print(f"  [{i:4d}/{n}] {label}")
        _process_one_spill(fpath, cols, ctrs)

    print(f"\n── Cut-flow for spin-{spin} ───────────────────────────────────")
    print(f"  Dimuon candidates (raw):             {ctrs['raw']:,}")
    print(f"  After FPGA trigger:                  {ctrs['fpga']:,}")
    print(f"  After Z_vertex > -600 cm:            {ctrs['zcut']:,}")
    print(f"  After |y_st1| > 3 cm:               {ctrs['yst1']:,}")
    print(f"  After chi2 cuts:                     {ctrs['chi2']:,}")
    print(f"  After mass [{MASS_MIN}, {MASS_MAX}] GeV: {ctrs['mass']:,}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with uproot.recreate(output_path) as fout:
        fout["tree"] = {k: np.array(v, dtype=np.float64) for k, v in cols.items()}
    print(f"  Written → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten multi-run/spill reco data into flat ROOT trees.")
    parser.add_argument(
        "--outdir", default=os.path.join(os.path.dirname(__file__), "data"),
        help="Directory to write flat_runs_{up,down}.root (default: ./data/)")
    parser.add_argument(
        "--spin", choices=["up", "down", "both"], default="both",
        help="Which spin state to process (default: both)")
    parser.add_argument(
        "--reco-dir", default=RECO_DIR,
        help=f"Path to reco-YYYYMMDD directory (default: {RECO_DIR})")
    args = parser.parse_args()

    # allow override of RECO_DIR at runtime
    global RECO_DIR
    RECO_DIR = args.reco_dir

    spins = ["up", "down"] if args.spin == "both" else [args.spin]
    for spin in spins:
        out = os.path.join(args.outdir, f"flat_runs_{spin}.root")
        process_spin(spin, out)

    print("\nDone.")
    if len(spins) == 2:
        print("\nNext steps:")
        print(f"  scp spinquestgpvm01:{args.outdir}/flat_runs_up.root   data/")
        print(f"  scp spinquestgpvm01:{args.outdir}/flat_runs_down.root data/")


if __name__ == "__main__":
    main()
