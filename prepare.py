"""Fixed infrastructure for J1-J2 2D Heisenberg recipe-search experiments.

This file is READ-ONLY for the agent.
It defines the lattice geometry, bond lists, sector logic,
device selection, and constants.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
LOGS_DIR = PROJECT_ROOT / "logs"
CONTROLLER_DIR = PROJECT_ROOT / "controller"

# ── Lattice parameters ──
LX = int(os.environ.get("TV_LX", "4"))
LY = int(os.environ.get("TV_LY", "4"))
N_SITES = LX * LY
BOUNDARY = "periodic"

# ── Family parameters ──
J1 = 1.0  # nearest-neighbor coupling (fixed)
J2_MIN = 0.0
J2_MAX = 0.7
PANEL_SIZE = 3  # random J2 values per trial

# ── Campaign defaults ──
DEFAULT_SEARCH_ITERATIONS = 20
DEFAULT_MAX_STEPS = 3000
DEFAULT_BATCH_SIZE = 256

# ── Device ──
def detect_device() -> str:
    requested = os.environ.get("TV_DEVICE", "auto").strip().lower()
    if requested in {"cpu", "mps", "cuda"}:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = detect_device()


# ── Lattice geometry ──
def _site_index(x: int, y: int, ly: int) -> int:
    """Convert (x, y) grid coordinates to a flat site index (row-major)."""
    return x * ly + y


def build_nn_bonds(lx: int, ly: int) -> list[tuple[int, int]]:
    """Nearest-neighbor bonds on a periodic square lattice.
    Each bond (i, j) with i < j appears exactly once."""
    bonds: list[tuple[int, int]] = []
    for x in range(lx):
        for y in range(ly):
            site = _site_index(x, y, ly)
            # right neighbor
            right = _site_index((x + 1) % lx, y, ly)
            if site < right:
                bonds.append((site, right))
            elif right < site:
                bonds.append((right, site))
            # up neighbor
            up = _site_index(x, (y + 1) % ly, ly)
            if site < up:
                bonds.append((site, up))
            elif up < site:
                bonds.append((up, site))
    # Deduplicate (periodic BC can create duplicates for small lattices)
    return sorted(set(bonds))


def build_nnn_bonds(lx: int, ly: int) -> list[tuple[int, int]]:
    """Next-nearest-neighbor bonds (diagonals) on a periodic square lattice.
    Each bond (i, j) with i < j appears exactly once."""
    bonds: list[tuple[int, int]] = []
    for x in range(lx):
        for y in range(ly):
            site = _site_index(x, y, ly)
            # upper-right diagonal
            ur = _site_index((x + 1) % lx, (y + 1) % ly, ly)
            pair = (min(site, ur), max(site, ur))
            bonds.append(pair)
            # upper-left diagonal
            ul = _site_index((x - 1) % lx, (y + 1) % ly, ly)
            pair = (min(site, ul), max(site, ul))
            bonds.append(pair)
    return sorted(set(bonds))


NN_BONDS = build_nn_bonds(LX, LY)
NNN_BONDS = build_nnn_bonds(LX, LY)
ALL_BONDS: list[tuple[int, int, float]] = []
for i, j in NN_BONDS:
    ALL_BONDS.append((i, j, J1))
# NNN bonds will have coupling J2 added at runtime


# ── Sector logic ──
def sector_n_up(sector: str) -> int:
    s = sector.lower().strip()
    if s == "sz0":
        return N_SITES // 2
    if s == "sz1":
        return N_SITES // 2 + 1
    raise ValueError(f"Unknown sector: {sector}")


# ── Randomness ──
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_random_j2(seed: int, count: int = PANEL_SIZE) -> tuple[float, ...]:
    """Sample random J2/J1 values for one trial panel."""
    rng = random.Random(seed)
    values: list[float] = []
    seen: set[float] = set()
    while len(values) < count:
        v = round(rng.uniform(J2_MIN, J2_MAX), 3)
        if v not in seen:
            seen.add(v)
            values.append(v)
    return tuple(values)


def build_curve_j2(n_points: int = 8) -> tuple[float, ...]:
    """Fixed J2 grid for final curve evaluation."""
    return tuple(float(x) for x in np.linspace(J2_MIN, J2_MAX, n_points))


# ── Directory helpers ──
def ensure_project_dirs() -> None:
    for p in (RESULTS_DIR, ANALYSIS_DIR, LOGS_DIR, CONTROLLER_DIR):
        p.mkdir(parents=True, exist_ok=True)


# ── Manifest ──
def build_manifest() -> dict[str, object]:
    return {
        "lx": LX,
        "ly": LY,
        "n_sites": N_SITES,
        "boundary": BOUNDARY,
        "device": DEVICE,
        "n_nn_bonds": len(NN_BONDS),
        "n_nnn_bonds": len(NNN_BONDS),
        "j1": J1,
        "j2_min": J2_MIN,
        "j2_max": J2_MAX,
        "panel_size": PANEL_SIZE,
    }


if __name__ == "__main__":
    ensure_project_dirs()
    m = build_manifest()
    print(json.dumps(m, indent=2, sort_keys=True))
    print(f"\nNN bonds ({len(NN_BONDS)}):")
    for b in NN_BONDS:
        print(f"  {b}")
    print(f"\nNNN bonds ({len(NNN_BONDS)}):")
    for b in NNN_BONDS:
        print(f"  {b}")
