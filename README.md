# SpinQuest (E1039) J/ψ TSSA Analysis

Transverse single-spin asymmetry (TSSA) analysis for J/ψ production using SpinQuest/E1039 data.

## Repository structure

```
ana-spinquest-fit/
├── convert_to_flat_uproot.py   # Convert raw data to flat trees
├── fit/
│   ├── fit_common.py           # Shared RooFit helpers and fit engine
│   ├── fit_mode_bkg.py         # Fit: base cuts only
│   └── fit_mode_final.py       # Fit: + station-1 track quality cuts
└── data/
    ├── flat_PM_up.root          # Flat tree, spin-up target
    └── flat_PM_down.root        # Flat tree, spin-down target
```

## Setup

Requires a conda environment with ROOT 6.34 (with PyROOT) and uproot 5:

```bash
conda activate root_env
```

Key packages: `root` (PyROOT enabled), `uproot`, `awkward`, `numpy`.

## Usage

### Step 1 — Convert raw data to flat trees

Copy the raw input files from the SeaQuest cluster before running:

* from the `/pnfs` area:
  `/seaquest/users/lcalerod/e1039-analysis/RecoData2024/work_ana/scratch/noHodoInTime/x-offset/`
  — copy `output_PM_up.root` and `output_PM_down.root` to `SpinQuestAna/data/`

Then run:

```bash
conda run -n root_env python3 convert_to_flat_uproot.py
```

Applies quality cuts and writes flat `flat_PM_{up,down}.root` to `data/`.

Cuts applied:
1. FPGA bit-0 (MATRIX1 trigger)
2. Both track vertex z > −600 cm
3. |y\_st1| > 3 cm for both muon tracks at station 1
4. χ² target vertex: χ²\_tgt > 0, χ²\_dump − χ²\_tgt > 0, χ²\_ups − χ²\_tgt > 0 (pos & neg)
5. Dimuon invariant mass: 0.5 ≤ M ≤ 10.0 GeV

### Step 2 — Run the mass fit

```bash
conda run -n root_env python3 fit/fit_mode_bkg.py    # base cuts only
conda run -n root_env python3 fit/fit_mode_final.py  # + station-1 cuts
```

Each script produces two PDFs:
- `fit_mode_bkg.pdf` / `fit_mode_final.pdf` — 2×2 RooFit canvas
- `fit_mode_bkg_hist.pdf` / `fit_mode_final_hist.pdf` — 2×2 raw mass histograms

## Fit model

Simultaneous unbinned extended maximum-likelihood fit across four categories:

| Category         | Target spin | Dimuon p\_x |
|------------------|-------------|-------------|
| Spin Up, Right   | ↑           | p\_x < 0   |
| Spin Up, Left    | ↑           | p\_x > 0   |
| Spin Down, Right | ↓           | p\_x < 0   |
| Spin Down, Left  | ↓           | p\_x > 0   |

Signal shape: two Gaussians (J/ψ and ψ') with **shared mean and sigma** across all four panels.
The ψ' parameters are constrained via the PDG mass ratio: M\_ψ'/M\_J/ψ = 3.686/3.097.
Background: per-panel exponential.

Fit range: 2.0–5.9 GeV, 60 bins (bin width = 0.065 GeV).

### Additional cuts in `fit_mode_final.py`

Applied with numpy before any RooFit call:
- x\_st1 < 25 cm for both tracks
- py\_st1\_pos × py\_st1\_neg < 0 (opposite-sign py at station 1)

## Output branches (flat trees)

| Branch | Description |
|--------|-------------|
| `rec_dimu_M` | Dimuon invariant mass [GeV] |
| `rec_dimu_px/py/pz` | Dimuon 3-momentum [GeV] |
| `rec_dimu_y/eta` | Dimuon rapidity / pseudorapidity |
| `rec_dimu_mT` | Dimuon transverse mass √(M²+pT²) [GeV] |
| `rec_mu_theta_pos/neg` | Muon horizontal bending angle arctan(px/pz) at vertex [rad] |
| `rec_mu_open_angle` | Opening angle between the two muons [rad] |
| `rec_mu_dpt` | pT(μ⁺) − pT(μ⁻) [GeV] |
| `rec_mu_deltaR` | ΔR(μ⁺, μ⁻) in (η, φ) |
| `rec_track_pos/neg_x_st1` | Track x position at station 1 [cm] |
| `rec_track_pos/neg_px_st1` | Track px at station 1 [GeV] |
| `rec_track_pos/neg_py_st1` | Track py at station 1 [GeV] |
| `rec_dz_vtx` | z\_vtx(μ⁺) − z\_vtx(μ⁻) [cm] |
