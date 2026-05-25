#!/usr/bin/env python3
"""
fit_inclusive.py — rough J/ψ + ψ' yield estimate without spin or L/R splitting.

Applies the same track-quality cuts as fit_mode_final.py:
  - |x_st1| < 25 cm for both muon tracks
  - py_st1_pos × py_st1_neg < 0  (opposite-sign py at station 1)

Both spin-up and spin-down files are merged into one sample.
No px (left/right) split is applied.

After fitting, yields for J/ψ, ψ', and background (others) are reported
inside the ±3σ(J/ψ) window of the fitted mean.

Data source (--data):
  flat  (default) — data/flat_PM_{up,down}.root
  runs            — data/flat_runs_{up,down}.root  (convert_to_flat_runs.py)
"""

import argparse

import numpy as np
from scipy import stats as scipy_stats
import ROOT
from ROOT import (
    RooRealVar, RooFormulaVar, RooDataSet, RooGaussian, RooExponential,
    RooAddPdf, RooFit, RooArgSet, RooArgList,
    TCanvas, TLegend, TLatex,
)

from fit_common import MASS_MIN, MASS_MAX, N_BINS, RATIO, make_roo_dataset
from data_loader import load_spin_data, output_path, add_data_arg

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

_BASE_OUTPUT = "/Users/spin/ana-spinquest-fit/fit/fit_inclusive.png"

_BRANCHES_FULL = [
    "rec_dimu_M",
    "rec_track_pos_x_st1",
    "rec_track_neg_x_st1",
    "rec_track_pos_py_st1",
    "rec_track_neg_py_st1",
]
_BRANCHES_BASE = ["rec_dimu_M"]

_ST1_COLS = {"rec_track_pos_x_st1", "rec_track_neg_x_st1",
             "rec_track_pos_py_st1", "rec_track_neg_py_st1"}


def load_mass_array(mode: str) -> np.ndarray:
    """Load both spin states, apply track cuts (if available), return combined mass array."""
    try:
        raw     = load_spin_data(mode, _BRANCHES_FULL)
        has_st1 = True
    except Exception as exc:
        if any(col in str(exc) for col in _ST1_COLS):
            print(f"  WARNING: station-1 branches not found ({exc})")
            print("  → skipping x_st1 + py_st1 cuts (base cuts only)")
            raw     = load_spin_data(mode, _BRANCHES_BASE)
            has_st1 = False
        else:
            raise

    chunks = []
    for spin in ("up", "down"):
        arr = raw[spin]
        if has_st1:
            x_cut  = (arr["rec_track_pos_x_st1"] < 25) & (arr["rec_track_neg_x_st1"] < 25)
            py_cut = arr["rec_track_pos_py_st1"] * arr["rec_track_neg_py_st1"] < 0
            keep   = x_cut & py_cut
            n_in   = len(arr["rec_dimu_M"])
            print(f"  [{spin}] {n_in:,} → {int(keep.sum()):,} after x_st1 + py_st1 cuts")
            chunks.append(arr["rec_dimu_M"][keep])
        else:
            chunks.append(arr["rec_dimu_M"])

    combined = np.concatenate(chunks)
    mass_cut = (combined > MASS_MIN) & (combined < MASS_MAX)
    result   = combined[mass_cut]
    print(f"  [all]  {len(combined):,} → {len(result):,} after mass window [{MASS_MIN}, {MASS_MAX}] GeV")
    return result


def window_yields(mean_v, sigma_v, mean2_v, sigma2_v, tau_v,
                  nsig1_v, nsig2_v, nbkg_v):
    """
    Compute expected yields of J/ψ, ψ', and background inside ±3σ(J/ψ).

    Uses the analytic PDFs: Gaussian for signals, exponential for background.
    All three are normalised over [MASS_MIN, MASS_MAX] before integrating
    over the sub-range [lo, hi].
    """
    lo = mean_v  - 3.0 * sigma_v
    hi = mean_v  + 3.0 * sigma_v
    lo = max(lo, MASS_MIN)
    hi = min(hi, MASS_MAX)

    # Gaussian fraction in [lo, hi]
    def gauss_frac(mu, sig):
        return (scipy_stats.norm.cdf(hi, mu, sig)
                - scipy_stats.norm.cdf(lo, mu, sig))

    # Exponential fraction in [lo, hi] (normalised over full fit range)
    def exp_frac(t):
        # integral of exp(t*x) from a to b  =  (exp(t*b) - exp(t*a)) / t
        num   = (np.exp(t * hi)       - np.exp(t * lo))       / t
        denom = (np.exp(t * MASS_MAX) - np.exp(t * MASS_MIN)) / t
        return num / denom

    n_jpsi   = nsig1_v * gauss_frac(mean_v,  sigma_v)
    n_psip   = nsig2_v * gauss_frac(mean2_v, sigma2_v)
    n_others = nbkg_v  * exp_frac(tau_v)

    return lo, hi, n_jpsi, n_psip, n_others


def fit_and_save(mass_vals, output_path):
    if len(mass_vals) < 10:
        raise RuntimeError(f"Too few events ({len(mass_vals)}) for fit")

    mass = RooRealVar("rec_dimu_M", "Dimuon mass [GeV]", MASS_MIN, MASS_MAX)
    mass.setBins(N_BINS)

    mean1  = RooRealVar("mean1",  "J/#psi mean",   3.10, 2.90, 3.40)
    sigma1 = RooRealVar("sigma1", "J/#psi #sigma", 0.10, 0.04, 0.30)
    mean2  = RooFormulaVar("mean2",  "#psi' mean",   f"@0*{RATIO:.8f}", RooArgList(mean1))
    sigma2 = RooFormulaVar("sigma2", "#psi' #sigma", f"@0*{RATIO:.8f}", RooArgList(sigma1))

    tau   = RooRealVar("tau",   "Bkg slope",  -2.0,  -6.0,   0.0)
    nsig1 = RooRealVar("nsig1", "N J/#psi",   200,     0, 20000)
    nsig2 = RooRealVar("nsig2", "N #psi'",     80,     0,  8000)
    nbkg  = RooRealVar("nbkg",  "N bkg",      400,     0, 40000)

    sig1 = RooGaussian("sig1", "J/#psi PDF", mass, mean1, sigma1)
    sig2 = RooGaussian("sig2", "#psi' PDF",  mass, mean2, sigma2)
    bkg  = RooExponential("bkg", "Bkg PDF",  mass, tau)
    mdl  = RooAddPdf("mdl", "Total PDF",
                     RooArgList(sig1, sig2, bkg),
                     RooArgList(nsig1, nsig2, nbkg))

    ds = make_roo_dataset("ds_all", mass, mass_vals)
    print(f"  Events in fit: {len(mass_vals):,}")

    # ── Scan initial conditions ────────────────────────────────────────────
    print("\n── Scanning initial conditions ───────────────────────────────")
    best_chi2, best_state = float("inf"), None

    for m1 in [2.95, 3.00, 3.05, 3.10, 3.15, 3.20, 3.25]:
        for tau_init in [-4.0, -2.0, -1.0]:
            mean1.setVal(m1); sigma1.setVal(0.10)
            tau.setVal(tau_init); nsig1.setVal(200); nsig2.setVal(80); nbkg.setVal(400)

            res = mdl.fitTo(ds, RooFit.PrintLevel(-1), RooFit.Save())
            if res.status() != 0:
                continue

            fr = mass.frame()
            ds.plotOn(fr, RooFit.Invisible())
            mdl.plotOn(fr, RooFit.Invisible())
            chi2 = fr.chiSquare()

            if chi2 < best_chi2:
                best_chi2 = chi2
                best_state = {
                    "mean1": mean1.getVal(), "sigma1": sigma1.getVal(),
                    "tau": tau.getVal(), "nsig1": nsig1.getVal(),
                    "nsig2": nsig2.getVal(), "nbkg": nbkg.getVal(),
                }

    if best_state is None:
        print("  WARNING: no converged fit in scan; using last values")
    else:
        mean1.setVal(best_state["mean1"]); sigma1.setVal(best_state["sigma1"])
        tau.setVal(best_state["tau"]); nsig1.setVal(best_state["nsig1"])
        nsig2.setVal(best_state["nsig2"]); nbkg.setVal(best_state["nbkg"])

    # ── Final fit ──────────────────────────────────────────────────────────
    print("\n── Final fit ─────────────────────────────────────────────────")
    mdl.fitTo(ds, RooFit.PrintLevel(-1))

    m1_v   = mean1.getVal();  m1_e  = mean1.getError()
    s1_v   = sigma1.getVal(); s1_e  = sigma1.getError()
    m2_v   = mean2.getVal()
    s2_v   = sigma2.getVal()
    tau_v  = tau.getVal()
    ns1_v  = nsig1.getVal();  ns1_e = nsig1.getError()
    ns2_v  = nsig2.getVal();  ns2_e = nsig2.getError()
    nb_v   = nbkg.getVal();   nb_e  = nbkg.getError()

    print(f"  J/ψ  mean  = {m1_v:.4f} ± {m1_e:.4f} GeV")
    print(f"  J/ψ  sigma = {s1_v:.4f} ± {s1_e:.4f} GeV")
    print(f"  ψ'   mean  = {m2_v:.4f} GeV  (constrained)")
    print(f"  ψ'   sigma = {s2_v:.4f} GeV  (constrained)")
    print(f"  N(J/ψ)     = {ns1_v:.0f} ± {ns1_e:.0f}")
    print(f"  N(ψ')      = {ns2_v:.0f} ± {ns2_e:.0f}")
    print(f"  N(bkg)     = {nb_v:.0f}  ± {nb_e:.0f}")

    # chi2/ndf
    fr_chi2 = mass.frame()
    ds.plotOn(fr_chi2, RooFit.Invisible())
    mdl.plotOn(fr_chi2, RooFit.Invisible())
    chi2ndf = fr_chi2.chiSquare()
    print(f"  chi2/ndf   = {chi2ndf:.2f}")

    # ── 3σ window yields ───────────────────────────────────────────────────
    lo, hi, n_jpsi_w, n_psip_w, n_others_w = window_yields(
        m1_v, s1_v, m2_v, s2_v, tau_v, ns1_v, ns2_v, nb_v)

    print(f"\n── ±3σ(J/ψ) window  [{lo:.3f}, {hi:.3f}] GeV ────────────────")
    print(f"  J/ψ    = {n_jpsi_w:.0f}")
    print(f"  ψ'     = {n_psip_w:.0f}")
    print(f"  others = {n_others_w:.0f}")

    # ── Draw ──────────────────────────────────────────────────────────────
    bw = (MASS_MAX - MASS_MIN) / N_BINS
    canvas = TCanvas("c_incl", "Inclusive J/psi fit", 900, 700)
    canvas.SetLeftMargin(0.14); canvas.SetRightMargin(0.05)
    canvas.SetTopMargin(0.08);  canvas.SetBottomMargin(0.14)

    frame = mass.frame(RooFit.Title("Inclusive (spin up + down, no L/R cut)"))
    frame.GetYaxis().SetTitle(f"Events / ({bw:.3f} GeV)")
    frame.GetXaxis().SetTitle("Dimuon mass [GeV]")

    ds.plotOn(frame, RooFit.MarkerSize(0.8))
    mdl.plotOn(frame, RooFit.LineColor(ROOT.kBlue),      RooFit.LineWidth(2), RooFit.Name("fit_curve"))
    mdl.plotOn(frame, RooFit.Components("sig1"),
               RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen + 1),
               RooFit.LineWidth(2), RooFit.Name("jpsi_curve"))
    mdl.plotOn(frame, RooFit.Components("sig2"),
               RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kMagenta),
               RooFit.LineWidth(2), RooFit.Name("psi2_curve"))
    mdl.plotOn(frame, RooFit.Components("bkg"),
               RooFit.LineStyle(ROOT.kDotted), RooFit.LineColor(ROOT.kRed),
               RooFit.LineWidth(2), RooFit.Name("bkg_curve"))

    frame.Draw()

    leg = TLegend(0.62, 0.62, 0.93, 0.90)
    leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.036)
    leg.AddEntry(frame.findObject("fit_curve"),  "Fit",    "L")
    leg.AddEntry(frame.findObject("jpsi_curve"), "J/#psi", "L")
    leg.AddEntry(frame.findObject("psi2_curve"), "#psi'",  "L")
    leg.AddEntry(frame.findObject("bkg_curve"),  "Bkg",    "L")
    leg.Draw()

    lat = TLatex(); lat.SetNDC(); lat.SetTextSize(0.034); lat.SetTextAlign(13)

    # -- fit quality + shape block --
    y = 0.59
    lat.DrawLatex(0.63, y,       f"#chi^{{2}}/ndf = {chi2ndf:.2f}");          y -= 0.06
    lat.DrawLatex(0.63, y,       f"#mu(J/#psi) = {m1_v:.4f} #pm {m1_e:.4f} GeV"); y -= 0.06
    lat.DrawLatex(0.63, y,       f"#sigma(J/#psi) = {s1_v:.4f} #pm {s1_e:.4f} GeV"); y -= 0.08

    # -- ±3σ window yields block --
    lat.SetTextColor(ROOT.kGray + 2)
    lat.DrawLatex(0.63, y,       f"#pm3#sigma window [{lo:.3f},{hi:.3f}] GeV"); y -= 0.06
    lat.SetTextColor(ROOT.kBlack)
    lat.DrawLatex(0.63, y,       f"N(J/#psi)  = {n_jpsi_w:.0f}");   y -= 0.06
    lat.DrawLatex(0.63, y,       f"N(#psi')   = {n_psip_w:.0f}");    y -= 0.06
    lat.DrawLatex(0.63, y,       f"N(others) = {n_others_w:.0f}")

    canvas.Update()
    canvas.SaveAs(output_path)
    print(f"\nSaved → {output_path}")


def draw_mass_hist(mode: str, out_path: str) -> None:
    """
    Draw a simple combined (up + down) dimuon mass histogram, 0–6 GeV.
    All events that survived the conversion cuts are shown — no mass
    window, no spin split, no L/R split.
    """
    HIST_MIN, HIST_MAX, HIST_BINS = 0.0, 6.0, 120   # 0.05 GeV / bin

    # Load only the mass branch — no extra cuts here
    raw = load_spin_data(mode, ["rec_dimu_M"])
    mass = np.concatenate([raw["up"]["rec_dimu_M"],
                           raw["down"]["rec_dimu_M"]])
    mass = mass[(mass >= HIST_MIN) & (mass <= HIST_MAX)]
    print(f"  Events in histogram [{HIST_MIN}, {HIST_MAX}] GeV: {len(mass):,}")

    bw = (HIST_MAX - HIST_MIN) / HIST_BINS

    h = ROOT.TH1F("h_mass_all", "Dimuon mass (spin up + down, all events)",
                  HIST_BINS, HIST_MIN, HIST_MAX)
    h.FillN(len(mass), mass.astype(np.float64), np.ones(len(mass)))
    h.GetXaxis().SetTitle("Dimuon mass [GeV]")
    h.GetYaxis().SetTitle(f"Events / ({bw:.2f} GeV)")
    h.SetLineWidth(2)
    h.SetLineColor(ROOT.kBlue + 1)

    canvas = ROOT.TCanvas("c_hist_all", "Mass histogram", 900, 700)
    canvas.SetLeftMargin(0.13); canvas.SetRightMargin(0.05)
    canvas.SetTopMargin(0.08);  canvas.SetBottomMargin(0.13)

    ROOT.gStyle.SetOptStat("nemr")
    h.Draw("HIST")
    canvas.Update()

    st = h.FindObject("TPaveStats")
    if st:
        st.SetX1NDC(0.62); st.SetX2NDC(0.93)
        st.SetY1NDC(0.65); st.SetY2NDC(0.88)
        st.Draw()

    canvas.Update()
    canvas.SaveAs(out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inclusive J/ψ fit — spin up + down merged, no L/R split")
    add_data_arg(parser)
    parser.add_argument(
        "--hist-only", action="store_true",
        help="Draw a raw mass histogram (0–6 GeV, all events) instead of running the fit")
    args = parser.parse_args()

    out = output_path(_BASE_OUTPUT, args.data)
    print(f"\n── Data mode: {args.data!r}  →  {out}")

    if args.hist_only:
        hist_out = out.replace(".png", "_mass_hist.png")
        print("── Drawing mass histogram ────────────────────────────────────")
        draw_mass_hist(args.data, hist_out)
    else:
        print("── Loading data ──────────────────────────────────────────────")
        mass_vals = load_mass_array(args.data)
        fit_and_save(mass_vals, out)
