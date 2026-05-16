import ROOT
from ROOT import (
    RooRealVar, RooDataSet, RooGaussian, RooExponential, RooAddPdf,
    RooFit, RooArgSet, RooArgList, TCanvas, TLatex, TPad
)

import math
file_up = ROOT.TFile("/Users/spin/SpinQuestAna/data/up_dimuon.root")
file_down = ROOT.TFile("/Users/spin/SpinQuestAna/data/down_dimuon.root")
tree_up = file_up.Get("tree")
tree_down = file_down.Get("tree")

def fit_and_plot(tree, name, hist_filter, position_label):
    """Fit a model to a filtered tree with parameter scanning for best fit, and draw text with chi2/NDF and J/ψ count"""
    # Define the mass variable
    mass = RooRealVar("mass", "Dimuon mass", 2.0, 6.0)
    mass.setBins(64)

    # Fill RooDataSet
    dataset = RooDataSet("dataset_" + name, "Dataset with mass " + name, RooArgSet(mass))
    for event in tree:
        if 2.0 < event.rec_mass_dimuon < 6.0 and hist_filter(event):
            mass.setVal(event.rec_mass_dimuon)
            dataset.add(RooArgSet(mass))

    n_entries = dataset.numEntries()
    print(f"{name} - Number of entries in dataset: {n_entries}")
    if n_entries == 0:
        return None, None, None, None

    # Define scan grids for initial parameters (mean1 and tau)
    mean1_scan = [3.0 + i * 0.05 for i in range(5)]  # 3.0 to 3.2 step 0.05
    tau_scan = [-5.0 + i * 0.5 for i in range(9)]    # -5.0 to -1.0 step 0.5

    # Initialize best fit tracking
    best_chi2 = float('inf')
    best_params = {}
    best_fit_status = -1

    # Build base PDFs (will recreate model in loop)
    mean1 = RooRealVar("mean1_" + name, "Mean of J/ψ Gaussian", 3.1, 3.0, 3.4)
    sigma1 = RooRealVar("sigma1_" + name, "Width of J/ψ Gaussian", 0.1, 0.05, 0.3)
    mean2 = RooRealVar("mean2_" + name, "Mean of ψ' Gaussian", 3.7, 3.6, 4.0)
    sigma2 = RooRealVar("sigma2_" + name, "Width of ψ' Gaussian", 0.1, 0.05, 0.3)
    tau = RooRealVar("tau_" + name, "Exponential slope", -2.0, -5.0, 0.0)
    nsig1 = RooRealVar("nsig1_" + name, "Number of J/ψ signal events", 50, 0, 500)
    nsig2 = RooRealVar("nsig2_" + name, "Number of ψ' signal events", 50, 0, 500)
    nbkg = RooRealVar("nbkg_" + name, "Number of background events", 50, 0, 1000)

    param_objects = [mean1, sigma1, mean2, sigma2, tau, nsig1, nsig2, nbkg]
    param_names = ["mean1_" + name, "sigma1_" + name, "mean2_" + name, "sigma2_" + name,
                   "tau_" + name, "nsig1_" + name, "nsig2_" + name, "nbkg_" + name]

    print(f"{name} - Scanning {len(mean1_scan)} x {len(tau_scan)} = {len(mean1_scan)*len(tau_scan)} initial conditions...")

    for init_mean1 in mean1_scan:
        for init_tau in tau_scan:
            # Set initial values
            mean1.setVal(init_mean1)
            tau.setVal(init_tau)

            # Build model for this iteration
            signal1 = RooGaussian("signal1_" + name, "J/ψ Signal PDF", mass, mean1, sigma1)
            signal2 = RooGaussian("signal2_" + name, "ψ' Signal PDF", mass, mean2, sigma2)
            background = RooExponential("background_" + name, "Background PDF", mass, tau)
            model = RooAddPdf(
                "model_" + name, "Total PDF " + name,
                RooArgList(signal1, signal2, background),
                RooArgList(nsig1, nsig2, nbkg)
            )

            # Fit
            fit_result = model.fitTo(dataset, RooFit.PrintLevel(-1), RooFit.Save())
            fit_status = fit_result.status()

            if fit_status == 0:  # Only consider converged fits
                # Compute chi2
                temp_frame = mass.frame(RooFit.Title("Temp"))
                dataset.plotOn(temp_frame, RooFit.Silence())
                model.plotOn(temp_frame, RooFit.Silence())
                chi2 = temp_frame.chiSquare()

                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_fit_status = fit_status
                    best_params = {param_names[i]: param_objects[i].getVal() for i in range(len(param_objects))}

            # Clean up temp objects only if they exist
            if 'model' in locals(): del model
            if 'signal1' in locals(): del signal1
            if 'signal2' in locals(): del signal2
            if 'background' in locals(): del background
            if 'fit_result' in locals(): del fit_result
            if 'temp_frame' in locals(): del temp_frame

    if best_params is None:
        print(f"{name} - No valid fit found in scan!")
        return None, None, None, None

    print(f"{name} - Best chi2/NDF from scan: {best_chi2:.2f}")

    # Restore best parameters
    for i, (pname, val) in enumerate(best_params.items()):
        param_objects[i].setVal(val)

    # Rebuild the best model with the best parameters
    signal1 = RooGaussian("signal1_" + name, "J/ψ Signal PDF", mass, mean1, sigma1)
    signal2 = RooGaussian("signal2_" + name, "ψ' Signal PDF", mass, mean2, sigma2)
    background = RooExponential("background_" + name, "Background PDF", mass, tau)
    model = RooAddPdf(
        "model_" + name, "Total PDF " + name,
        RooArgList(signal1, signal2, background),
        RooArgList(nsig1, nsig2, nbkg)
    )

    # Extract values
    jpsi_counts = nsig1.getVal()
    jpsi_mean = mean1.getVal()
    jpsi_sigma = sigma1.getVal()
    total_counts = dataset.sumEntries()

    # Make frame and plot
    frame = mass.frame(RooFit.Title(""))
    dataset.plotOn(frame)
    model.plotOn(frame, RooFit.LineColor(ROOT.kBlue))
    model.plotOn(frame, RooFit.Components("signal1_" + name),
                 RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen))
    model.plotOn(frame, RooFit.Components("signal2_" + name),
                 RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kMagenta))
    model.plotOn(frame, RooFit.Components("background_" + name),
                 RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))

    # Use the best chi2 from scan
    chi2ndf = best_chi2

    print(f"{name} - J/ψ Signal Counts: {jpsi_counts:.0f}, Mean: {jpsi_mean:.3f}, Sigma: {jpsi_sigma:.3f}, Total: {total_counts:.0f}, Chi²/NDF: {chi2ndf:.2f}")

    # Draw frame
    frame.Draw()

    # Overlay text
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.03)
    latex.SetTextAlign(13)  # top-left
    y = 0.88
    dy = 0.05
    latex.DrawLatex(0.15, y, position_label)
    y -= dy
    latex.DrawLatex(0.15, y, f"#chi^{{2}}/ndf = {chi2ndf:.2f}")
    y -= dy
    latex.DrawLatex(0.15, y, f"no. J/#psi = {int(jpsi_counts)}")

    # Clean up
    del model, signal1, signal2, background

    return frame, latex, dataset, mass  # Return mass for raw plot

# Canvas with two pads side by side
canvas = TCanvas("fitCanvas", "Fit and Raw Data for up_right", 1200, 600)
canvas.Divide(2, 1)

latexs = []

# Fit and plot for up_right only
canvas.cd(1)
frame, latex, dataset, mass = fit_and_plot(
    tree_up, "up_right",
    lambda event: event.rec_px_dimuon < 0 and event.rec_pz_dimuon > 0,
    "up_right"
)
if latex: latexs.append(latex)

canvas.cd(2)
raw_frame = mass.frame(RooFit.Title("Raw Data"))
dataset.plotOn(raw_frame)
raw_frame.Draw()

canvas.Draw()
