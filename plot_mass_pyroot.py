#!/usr/bin/env python3
"""
plot_mass_pyroot.py — independent cross-check using PyROOT TTree::Draw.

Reads the raw output_PM_{up,down}.root files directly and applies all cuts
via TTree::Draw selection strings — no dependence on the uproot pipeline.

Cuts:
  1. FPGA bit-0
  2. z_vtx > -600 cm (both tracks)
  3. |y_st1| > 3 cm (both tracks)
  4. chi2 target/dump/upstream (pos & neg)
  5. x_st1 < 25 cm (both tracks)
  6. py_st1_pos * py_st1_neg < 0

Output: plots/mass_pyroot.png

Usage:
    conda run -n root_env python3 plot_mass_pyroot.py
"""

import os
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(1111)
ROOT.gStyle.SetStatFormat("6.4g")
ROOT.gStyle.SetOptTitle(1)
ROOT.gStyle.SetStatX(0.98)
ROOT.gStyle.SetStatY(0.98)
ROOT.gStyle.SetStatW(0.22)
ROOT.gStyle.SetStatH(0.12)
ROOT.gStyle.SetStatBorderSize(1)

FILE_UP   = "/Users/spin/SpinQuestAna/data/output_PM_up.root"
FILE_DOWN = "/Users/spin/SpinQuestAna/data/output_PM_down.root"
OUTPUT    = "plots/mass_pyroot.png"

N_BINS        = 152
M_MIN, M_MAX  = 0.1, 10.0

# ── selection strings ──────────────────────────────────────────────────────────
BASE_SEL = (
    "(event.fpga_bits&1)"
    " && dimuon_list.pos_pos.fZ>-600 && dimuon_list.pos_neg.fZ>-600"
    " && abs(dimuon_list.pos_pos_st1.fY)>3 && abs(dimuon_list.pos_neg_st1.fY)>3"
    " && dimuon_list.chisq_target_pos>0"
    " && (dimuon_list.chisq_dump_pos-dimuon_list.chisq_target_pos)>0"
    " && (dimuon_list.chisq_upstream_pos-dimuon_list.chisq_target_pos)>0"
    " && dimuon_list.chisq_target_neg>0"
    " && (dimuon_list.chisq_dump_neg-dimuon_list.chisq_target_neg)>0"
    " && (dimuon_list.chisq_upstream_neg-dimuon_list.chisq_target_neg)>0"
    " && dimuon_list.pos_pos_st1.fX<25 && dimuon_list.pos_neg_st1.fX<25"
    " && dimuon_list.mom_pos_st1.fP.fY*dimuon_list.mom_neg_st1.fP.fY<0"
)
RIGHT_SEL = BASE_SEL + " && dimuon_list.mom_target.fP.fX<0"
LEFT_SEL  = BASE_SEL + " && dimuon_list.mom_target.fP.fX>0"

PANELS = [
    (1, "up",   RIGHT_SEL, "Spin Up, Right (p_{x} < 0)",   "h_up_right"),
    (2, "up",   LEFT_SEL,  "Spin Up, Left (p_{x} > 0)",    "h_up_left"),
    (3, "down", RIGHT_SEL, "Spin Down, Right (p_{x} < 0)", "h_down_right"),
    (4, "down", LEFT_SEL,  "Spin Down, Left (p_{x} > 0)",  "h_down_left"),
]

DRAW_VAR = "dimuon_list.mom_target.M()"


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    fin = {
        "up":   ROOT.TFile.Open(FILE_UP),
        "down": ROOT.TFile.Open(FILE_DOWN),
    }

    canvas = ROOT.TCanvas("canvas", "Dimuon Mass", 1200, 900)
    canvas.Divide(2, 2)

    hists = []

    for pad, file_key, sel, title, hist_name in PANELS:
        canvas.cd(pad)
        ROOT.gPad.SetLeftMargin(0.12)
        ROOT.gPad.SetBottomMargin(0.12)
        ROOT.gPad.SetTopMargin(0.10)

        tree = fin[file_key].Get("tree")
        draw_expr = f"{DRAW_VAR}>>{hist_name}({N_BINS},{M_MIN},{M_MAX})"
        tree.Draw(draw_expr, sel, "HIST")

        h = ROOT.gDirectory.Get(hist_name)
        h.SetTitle(title)
        h.GetXaxis().SetTitle("Dimuon mass [GeV]")
        h.GetYaxis().SetTitle("Events")
        h.GetXaxis().SetTitleSize(0.05)
        h.GetYaxis().SetTitleSize(0.05)
        h.GetXaxis().SetLabelSize(0.04)
        h.GetYaxis().SetLabelSize(0.04)
        h.SetLineColor(ROOT.kBlue + 1)
        h.SetLineWidth(1)
        h.Draw("HIST")
        hists.append(h)

        print(f"  {hist_name:<16s}: {int(h.GetEntries()):>5,}  mean={h.GetMean():.3f}")

    canvas.SaveAs(OUTPUT)
    for f in fin.values():
        f.Close()
    print(f"\nSaved → {OUTPUT}")


if __name__ == "__main__":
    main()
