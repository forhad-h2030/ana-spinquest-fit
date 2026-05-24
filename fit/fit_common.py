"""
fit_common.py — shared constants, RooFit helpers, and the fit+draw engine.

Imported by fit_mode_bkg.py and fit_mode_final.py.
Both mode scripts produce {suffix: np.ndarray_of_mass_values} and call fit_and_save().
"""

import numpy as np
import ROOT
from ROOT import (
    RooRealVar, RooFormulaVar, RooDataSet, RooGaussian, RooExponential,
    RooAddPdf, RooSimultaneous, RooCategory,
    RooFit, RooArgSet, RooArgList, TCanvas, TH1F, TLatex, TLegend,
)

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

# ── Physics / binning constants ────────────────────────────────────────────────
MASS_MIN, MASS_MAX = 2.0, 5.9
N_BINS             = 60            # bin width = 0.065 GeV
RATIO              = 3.686 / 3.097  # M_ψ' / M_J/ψ  (PDG)

# Four fit categories: (spin, side, canvas-pad, label)
PANELS = [
    ("up",   "right", 1, "Spin Up, Right (p_{x} < 0)"),
    ("up",   "left",  2, "Spin Up, Left (p_{x} > 0)"),
    ("down", "right", 3, "Spin Down, Right (p_{x} < 0)"),
    ("down", "left",  4, "Spin Down, Left (p_{x} > 0)"),
]


def panel_suffix(spin, side):
    return f"{spin}_{side}"


# ── RooFit helpers ─────────────────────────────────────────────────────────────

def make_roo_dataset(name, mass_var, values):
    """Fill a RooDataSet from a numpy array of mass values."""
    ds = RooDataSet(name, name, RooArgSet(mass_var))
    for v in values:
        mass_var.setVal(float(v))
        ds.add(RooArgSet(mass_var))
    return ds


def build_panel_model(suffix, mass, mean1, sigma1, mean2, sigma2):
    """
    Per-panel extended PDF.  Signal shapes are passed in (shared across panels).
    Returns (model, (tau, nsig1, nsig2, nbkg), (sig1, sig2, bkg)).
    """
    tau   = RooRealVar(f"tau_{suffix}",   "Bkg slope", -2.0, -6.0,   0.0)
    nsig1 = RooRealVar(f"nsig1_{suffix}", "N J/#psi",   50,    0,  5000)
    nsig2 = RooRealVar(f"nsig2_{suffix}", "N #psi'",    20,    0,  2000)
    nbkg  = RooRealVar(f"nbkg_{suffix}",  "N bkg",     100,    0, 10000)

    sig1 = RooGaussian(f"sig1_{suffix}", "J/#psi PDF", mass, mean1, sigma1)
    sig2 = RooGaussian(f"sig2_{suffix}", "#psi' PDF",  mass, mean2, sigma2)
    bkg  = RooExponential(f"bkg_{suffix}", "Bkg PDF",  mass, tau)
    mdl  = RooAddPdf(f"mdl_{suffix}", "Total PDF",
                     RooArgList(sig1, sig2, bkg),
                     RooArgList(nsig1, nsig2, nbkg))

    return mdl, (tau, nsig1, nsig2, nbkg), (sig1, sig2, bkg)


def draw_panel(pad, mass, ds, mdl, suffix, nsig1, chi2ndf, label):
    """Draw fit frame with components, legend, and stat box on a pad."""
    bw = (MASS_MAX - MASS_MIN) / N_BINS
    frame = mass.frame(RooFit.Title(label))
    frame.GetYaxis().SetTitle(f"Events / ({bw:.3f} GeV)")
    frame.GetXaxis().SetTitle("Dimuon mass [GeV]")

    ds.plotOn(frame, RooFit.MarkerSize(0.8))
    mdl.plotOn(frame, RooFit.LineColor(ROOT.kBlue),      RooFit.LineWidth(2), RooFit.Name("fit_curve"))
    mdl.plotOn(frame, RooFit.Components(f"sig1_{suffix}"),
               RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen + 1),
               RooFit.LineWidth(2), RooFit.Name("jpsi_curve"))
    mdl.plotOn(frame, RooFit.Components(f"sig2_{suffix}"),
               RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kMagenta),
               RooFit.LineWidth(2), RooFit.Name("psi2_curve"))
    mdl.plotOn(frame, RooFit.Components(f"bkg_{suffix}"),
               RooFit.LineStyle(ROOT.kDotted), RooFit.LineColor(ROOT.kRed),
               RooFit.LineWidth(2), RooFit.Name("bkg_curve"))

    pad.cd()
    pad.SetLeftMargin(0.14); pad.SetRightMargin(0.05)
    pad.SetTopMargin(0.10);  pad.SetBottomMargin(0.14)
    frame.Draw()

    leg = TLegend(0.62, 0.55, 0.93, 0.88)
    leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.038)
    leg.AddEntry(frame.findObject("fit_curve"),  "Fit",    "L")
    leg.AddEntry(frame.findObject("jpsi_curve"), "J/#psi", "L")
    leg.AddEntry(frame.findObject("psi2_curve"), "#psi'",  "L")
    leg.AddEntry(frame.findObject("bkg_curve"),  "Bckg",   "L")
    leg.Draw()

    lat = TLatex(); lat.SetNDC(); lat.SetTextSize(0.038); lat.SetTextAlign(13)
    lat.DrawLatex(0.63, 0.53, f"#chi^{{2}}/ndf = {chi2ndf:.2f}")
    lat.DrawLatex(0.63, 0.47,
                  f"no. J/#psi = {int(nsig1.getVal())} #pm {int(nsig1.getError())}")

    pad.Update()
    return frame, leg, lat


# ── Histogram canvas ───────────────────────────────────────────────────────────

def draw_histograms(mass_arrays, output_path):
    """
    Draw a 2×2 canvas of TH1F mass histograms with ROOT stats boxes
    (entries, mean, std dev) — one pad per panel.
    """
    bw = (MASS_MAX - MASS_MIN) / N_BINS

    canvas = TCanvas("c_hist", "Mass histograms", 1400, 1100)
    canvas.Divide(2, 2)
    keep = []

    for spin, side, pad_idx, label in PANELS:
        suf  = panel_suffix(spin, side)
        vals = mass_arrays[suf].astype(np.float64)

        h = TH1F(f"h_{suf}", label, N_BINS, MASS_MIN, MASS_MAX)
        h.FillN(len(vals), vals, np.ones(len(vals)))
        h.GetXaxis().SetTitle("Dimuon mass [GeV]")
        h.GetYaxis().SetTitle(f"Events / ({bw:.3f} GeV)")
        h.SetLineWidth(2)

        pad = canvas.GetPad(pad_idx)
        pad.cd()
        pad.SetLeftMargin(0.14); pad.SetRightMargin(0.05)
        pad.SetTopMargin(0.10);  pad.SetBottomMargin(0.14)

        ROOT.gStyle.SetOptStat("nemr")   # entries, mean, std dev, RMS label
        h.Draw("E")
        pad.Update()

        # reposition the stats box to the upper-right corner
        st = h.FindObject("TPaveStats")
        if st:
            st.SetX1NDC(0.62); st.SetX2NDC(0.93)
            st.SetY1NDC(0.65); st.SetY2NDC(0.88)
            st.Draw()
        pad.Update()
        keep.append(h)

    canvas.Update()
    canvas.SaveAs(output_path)
    print(f"Saved → {output_path}")


# ── Main fit engine ────────────────────────────────────────────────────────────

def fit_and_save(mass_arrays, output_path):
    """
    Run simultaneous J/ψ fit and save a 2×2 canvas PDF.

    Parameters
    ----------
    mass_arrays : dict[str, np.ndarray]
        Keys are panel suffixes ("up_right", "up_left", "down_right", "down_left").
        Values are 1-D arrays of dimuon mass values already filtered to the
        desired sample (mass window + px direction + any extra cuts).
    output_path : str
        Destination PDF file.
    """
    mass = RooRealVar("rec_dimu_M", "Dimuon mass [GeV]", MASS_MIN, MASS_MAX)
    mass.setBins(N_BINS)

    mean1  = RooRealVar("mean1",  "J/#psi mean",   3.10, 2.90, 3.40)
    sigma1 = RooRealVar("sigma1", "J/#psi #sigma", 0.10, 0.04, 0.30)
    mean2  = RooFormulaVar("mean2",  "#psi' mean",   f"@0*{RATIO:.8f}", RooArgList(mean1))
    sigma2 = RooFormulaVar("sigma2", "#psi' #sigma", f"@0*{RATIO:.8f}", RooArgList(sigma1))

    # ── Datasets from numpy arrays ─────────────────────────────────────────
    datasets = {}
    for spin, side, _, _ in PANELS:
        suf  = panel_suffix(spin, side)
        vals = mass_arrays[suf]
        print(f"  [{suf}] {len(vals)} events in fit window")
        if len(vals) < 10:
            raise RuntimeError(f"Too few events in {suf}: {len(vals)}")
        datasets[suf] = make_roo_dataset(f"ds_{suf}", mass, vals)

    # ── Per-panel models (shared mean/sigma) ───────────────────────────────
    models      = {}
    panel_pars  = {}
    keep_shapes = []
    for spin, side, _, _ in PANELS:
        suf = panel_suffix(spin, side)
        mdl, pars, shapes = build_panel_model(suf, mass, mean1, sigma1, mean2, sigma2)
        models[suf]     = mdl
        panel_pars[suf] = pars
        keep_shapes.extend(shapes)

    # ── Simultaneous PDF ───────────────────────────────────────────────────
    sample = RooCategory("sample", "sample")
    for spin, side, _, _ in PANELS:
        sample.defineType(panel_suffix(spin, side))

    combData = RooDataSet(
        "combData", "Combined data", RooArgSet(mass, sample),
        RooFit.Index(sample),
        RooFit.Import("up_right",   datasets["up_right"]),
        RooFit.Import("up_left",    datasets["up_left"]),
        RooFit.Import("down_right", datasets["down_right"]),
        RooFit.Import("down_left",  datasets["down_left"]),
    )

    simPdf = RooSimultaneous("simPdf", "Simultaneous PDF", sample)
    for spin, side, _, _ in PANELS:
        suf = panel_suffix(spin, side)
        simPdf.addPdf(models[suf], suf)

    # ── Scan initial conditions ────────────────────────────────────────────
    print("\n── Scanning initial conditions ───────────────────────────────")
    best_chi2, best_state = float("inf"), None

    for m1 in [2.95, 3.00, 3.05, 3.10, 3.15, 3.20, 3.25]:
        for tau_init in [-4.0, -2.0, -1.0]:
            mean1.setVal(m1); sigma1.setVal(0.10)
            for spin, side, _, _ in PANELS:
                tau, ns1, ns2, nb = panel_pars[panel_suffix(spin, side)]
                tau.setVal(tau_init); ns1.setVal(50); ns2.setVal(20); nb.setVal(100)

            res = simPdf.fitTo(combData, RooFit.PrintLevel(-1), RooFit.Save())
            if res.status() != 0:
                continue

            total_chi2 = 0
            for spin, side, _, _ in PANELS:
                suf = panel_suffix(spin, side)
                fr = mass.frame()
                datasets[suf].plotOn(fr, RooFit.Invisible())
                models[suf].plotOn(fr, RooFit.Invisible())
                total_chi2 += fr.chiSquare()

            if total_chi2 < best_chi2:
                best_chi2 = total_chi2
                best_state = {
                    "mean1":  mean1.getVal(),
                    "sigma1": sigma1.getVal(),
                    "panels": {
                        panel_suffix(s, d): tuple(p.getVal() for p in panel_pars[panel_suffix(s, d)])
                        for s, d, _, _ in PANELS
                    },
                }

    if best_state is None:
        print("  WARNING: no converged fit in scan; using last values")
    else:
        mean1.setVal(best_state["mean1"])
        sigma1.setVal(best_state["sigma1"])
        for spin, side, _, _ in PANELS:
            suf = panel_suffix(spin, side)
            for p, v in zip(panel_pars[suf], best_state["panels"][suf]):
                p.setVal(v)

    # ── Final fit ──────────────────────────────────────────────────────────
    print("\n── Final simultaneous fit ────────────────────────────────────")
    simPdf.fitTo(combData, RooFit.PrintLevel(-1))
    print(f"  J/ψ  mean  = {mean1.getVal():.4f} ± {mean1.getError():.4f} GeV")
    print(f"  J/ψ  sigma = {sigma1.getVal():.4f} ± {sigma1.getError():.4f} GeV")
    print(f"  ψ'   mean  = {mean2.getVal():.4f} GeV  (constrained)")
    print(f"  ψ'   sigma = {sigma2.getVal():.4f} GeV  (constrained)")

    # ── Draw ──────────────────────────────────────────────────────────────
    canvas = TCanvas("c_jpsi", "J/psi fit", 1400, 1100)
    canvas.Divide(2, 2)
    keep_alive = [combData, simPdf, sample, mean1, sigma1, mean2, sigma2] + keep_shapes

    for spin, side, pad_idx, label in PANELS:
        suf = panel_suffix(spin, side)
        _, nsig1_p, *_ = panel_pars[suf]

        fr = mass.frame()
        datasets[suf].plotOn(fr, RooFit.Invisible())
        models[suf].plotOn(fr, RooFit.Invisible())
        chi2ndf = fr.chiSquare()
        print(f"  [{suf}]  chi2/ndf={chi2ndf:.2f}  "
              f"N(J/psi)={nsig1_p.getVal():.0f} ± {nsig1_p.getError():.0f}")

        frame, leg, lat = draw_panel(
            canvas.GetPad(pad_idx), mass, datasets[suf], models[suf],
            suf, nsig1_p, chi2ndf, label)
        keep_alive.extend([datasets[suf], frame, leg, lat])

    canvas.Update()
    canvas.SaveAs(output_path)
    print(f"\nSaved → {output_path}")

    hist_path = output_path.replace(".png", "_hist.png")
    draw_histograms(mass_arrays, hist_path)
