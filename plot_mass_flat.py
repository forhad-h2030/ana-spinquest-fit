#!/usr/bin/env python3
"""
plot_mass_flat.py — 2×2 dimuon mass histograms from the flat up/down files.

Input:  data/flat_PM_up.root   (single tree "tree", rec_dimu_M + rec_dimu_px)
        data/flat_PM_down.root
Output: plots/mass_flat.png

Left/right split from rec_dimu_px:
    right → px < 0
    left  → px > 0

No additional cuts applied.

Usage:
    conda run -n root_env python3 plot_mass_flat.py
"""

import os
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILE_UP   = "data/flat_PM_up.root"
FILE_DOWN = "data/flat_PM_down.root"
OUTPUT    = "plots/mass_flat.png"

M_MIN, M_MAX = 0.1, 10.0
N_BINS = 152                              # bin width ≈ 0.065 GeV
BIN_EDGES = np.linspace(M_MIN, M_MAX, N_BINS + 1)
BIN_WIDTH = BIN_EDGES[1] - BIN_EDGES[0]

HIST_COLOR = (0.33, 0.33, 0.78)          # ROOT-like blue-purple

BRANCHES = ["rec_dimu_M", "rec_dimu_px"]

PANELS = [
    # (row, col, spin_file, side,    title,                           hist_name)
    (0, 0, "up",   "right", r"Spin Up, Right ($p_x < 0$)",   "h_up_right"),
    (0, 1, "up",   "left",  r"Spin Up, Left ($p_x > 0$)",    "h_up_left"),
    (1, 0, "down", "right", r"Spin Down, Right ($p_x < 0$)", "h_down_right"),
    (1, 1, "down", "left",  r"Spin Down, Left ($p_x > 0$)",  "h_down_left"),
]


def style_axes(ax):
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def stat_box(ax, hist_name, mass):
    entries = len(mass)
    mean    = float(np.mean(mass)) if entries > 0 else 0.0
    std_dev = float(np.std(mass))  if entries > 0 else 0.0
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

    # Load both spin files once
    spin_data = {}
    for spin, path in [("up", FILE_UP), ("down", FILE_DOWN)]:
        with uproot.open(path) as fin:
            arrays = fin["tree"].arrays(BRANCHES, library="ak")
            spin_data[spin] = {
                "mass": ak.to_numpy(arrays["rec_dimu_M"]),
                "px":   ak.to_numpy(arrays["rec_dimu_px"]),
            }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.38, wspace=0.3)

    for row, col, spin, side, title, hist_name in PANELS:
        ax   = axes[row][col]
        mass = spin_data[spin]["mass"]
        px   = spin_data[spin]["px"]

        sel  = (px < 0) if side == "right" else (px > 0)
        mass = mass[sel]

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
