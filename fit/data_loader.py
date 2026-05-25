"""
data_loader.py — unified data-source configuration for all fit modes.

Two source modes
----------------
  "flat"  (default) — the two pre-existing flat files produced by
                       convert_to_flat.py.  These are the single-file
                       spin-up / spin-down datasets used in all original
                       analysis.

  "runs"             — flat files produced by convert_to_flat_runs.py,
                       which aggregate many run directories from:
                         RECO_DIR/run_XXXXXX/spill_*/out/output_PM.root
                       Spin assignment (reco-20260512):
                         UP  : runs 6111-6118, 6156
                         DOWN: runs 6135-6139, 6149-6155

Usage in a fit script
---------------------
    from data_loader import load_spin_data, output_path

    raw = load_spin_data(mode, branches)
    # raw["up"]   → dict of branch → np.ndarray
    # raw["down"] → dict of branch → np.ndarray

    out = output_path(base_path, mode)
    # "flat" → base_path unchanged
    # "runs" → inserts "_runs" before the extension
"""

import os
import uproot

# ── File paths ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root

_PATHS = {
    "flat": {
        "up":   os.path.join(_HERE, "data", "flat_PM_up.root"),
        "down": os.path.join(_HERE, "data", "flat_PM_down.root"),
    },
    "runs": {
        "up":   os.path.join(_HERE, "data", "flat_runs_up.root"),
        "down": os.path.join(_HERE, "data", "flat_runs_down.root"),
    },
    "runs_rm": {
        "up":   os.path.join(_HERE, "data", "flat_runs_up_rm.root"),
        "down": os.path.join(_HERE, "data", "flat_runs_down_rm.root"),
    },
}

# ── Spin assignment metadata (for reference / documentation) ──────────────────
SPIN_UP_RUNS   = frozenset({6111, 6112, 6113, 6114, 6115, 6116, 6117, 6118, 6156})
SPIN_DOWN_RUNS = frozenset({6135, 6136, 6137, 6138, 6139,
                             6149, 6150, 6151, 6152, 6153, 6154, 6155})


# ── Public API ─────────────────────────────────────────────────────────────────

def load_spin_data(mode: str, branches: list) -> dict:
    """
    Load spin-up and spin-down arrays from the chosen data source.

    Parameters
    ----------
    mode     : "flat" or "runs"
    branches : list of branch names to read from the flat tree

    Returns
    -------
    dict with keys "up" and "down".
    Each value is a dict of {branch_name: np.ndarray}.
    """
    if mode not in _PATHS:
        raise ValueError(
            f"Unknown data mode {mode!r}. Choose 'flat', 'runs', or 'runs_rm'.")

    raw = {}
    for spin in ("up", "down"):
        path = _PATHS[mode][spin]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[mode={mode!r}, spin={spin!r}] File not found: {path}\n"
                + ("  → Run convert_to_flat_runs.py first, then copy "
                   "flat_runs_*.root to data/."
                   if mode == "runs" else
                   "  → Run convert_to_flat.py to create the flat files."))
        with uproot.open(path) as f:
            raw[spin] = f["tree"].arrays(branches, library="np")
        n = len(next(iter(raw[spin].values())))
        print(f"  [{mode}/{spin}] {n:,} events from {os.path.basename(path)}")

    return raw


def output_path(base: str, mode: str) -> str:
    """
    Derive an output file path that encodes the data mode.

    "flat"    → base unchanged          (e.g. fit_mode_final.png)
    "runs"    → _runs inserted          (e.g. fit_mode_final_runs.png)
    "runs_rm" → _runs_rm inserted       (e.g. fit_mode_final_runs_rm.png)
    """
    if mode == "flat":
        return base
    root, ext = os.path.splitext(base)
    suffix = "_runs_rm" if mode == "runs_rm" else "_runs"
    return f"{root}{suffix}{ext}"


def add_data_arg(parser) -> None:
    """Add --data argument to an argparse.ArgumentParser."""
    parser.add_argument(
        "--data", choices=["flat", "runs", "runs_rm"], default="flat",
        help="Data source: 'flat' = single flat files (default); "
             "'runs' = multi-run reco data; "
             "'runs_rm' = multi-run with road-matching cut")
