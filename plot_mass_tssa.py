#!/usr/bin/env python3
"""
plot_mass_tssa.py — 2×2 dimuon mass histograms from the pre-split TSSA file.

Input:  data/tssa_target_ITCuts_offset_x25_y3_ylikeSign_cut.root
        Trees: tree_up_left, tree_up_right, tree_down_left, tree_down_right
Output: plots/mass_tssa.png

No additional cuts applied.

Usage:
    conda run -n root_env python3 plot_mass_tssa.py
"""

import os
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT  = "data/tssa_target_ITCuts_offset_x25_y3_ylikeSign_cut.root"
OUTPUT = "plots/mass_tssa.png"

M_MIN, M_MAX = 0.1, 10.0
N_BINS = 152                              # bin width ≈ 0.065 GeV
BIN_EDGES = np.linspace(M_MIN, M_MAX, N_BINS + 1)
BIN_WIDTH = BIN_EDGES[1] - BIN_EDGES[0]

HIST_COLOR = (0.33, 0.33, 0.78)          # ROOT-like blue-purple

PANELS = [
    # (row, col, tree_name, title, hist_name)
    (0, 0, "tree_up_right",   r"Spin Up, Right ($p_x < 0$)",   "h_up_right"),
    (0, 1, "tree_up_left",    r"Spin Up, Left ($p_x > 0$)",    "h_up_left"),
    (1, 0, "tree_down_right", r"Spin Down, Right ($p_x < 0$)", "h_down_right"),
    (1, 1, "tree_down_left",  r"Spin Down, Left ($p_x > 0$)",  "h_down_left"),
]


def style_axes(ax):
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def stat_box(ax, hist_name, mass):
    entries = len(mass)
    mean    = float(np.mean(mass))
    std_dev = float(np.std(mass))
    lines = [
        (hist_name, ""),
        ("Entries", f"{entries}"),
        ("Mean",    f"{mean:.4g}"),
        ("Std Dev", f"{std_dev:.4g}"),
    ]
    text = "\n".join(f"{k:<10}{v}" for k, v in lines)
    ax.text(0.97, 0.97, text, transform=ax.transAxes,
            fontsize=8, verticalalignment="top", horizontalalignment="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="square,pad=0.5", facecolor="white",
                      edgecolor="black", linewidth=0.8))


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.38, wspace=0.3)

    with uproot.open(INPUT) as fin:
        for row, col, tree_name, title, hist_name in PANELS:
            ax   = axes[row][col]
            mass = ak.to_numpy(fin[tree_name].arrays(["m_tgt"], library="ak")["m_tgt"])

            ax.hist(mass, bins=BIN_EDGES, histtype="step",
                    color=HIST_COLOR, linewidth=1.0)

            ax.set_title(hist_name, fontsize=11)
            ax.set_xlabel("Dimuon mass [GeV]", fontsize=10)
            ax.set_ylabel("Events", fontsize=10)
            ax.set_xlim(M_MIN, M_MAX)
            ax.set_ylim(bottom=0)
            ax.tick_params(axis="both", labelsize=9)
            style_axes(ax)

            # panel label above plot
            ax.text(0.5, 1.06, title, transform=ax.transAxes,
                    fontsize=10, ha="center", va="bottom")

            stat_box(ax, hist_name, mass)

    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()
