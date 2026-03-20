"""Exact diagonalization for the J1-J2 Heisenberg model on a periodic square lattice.

Uses scipy sparse matrices restricted to a fixed Sz sector.
Supports lattices up to ~24 sites (dim ≈ 2.7 M for 4×6 Sz=0).
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare import LX, LY, N_SITES, NN_BONDS, NNN_BONDS, J1, sector_n_up


def _enumerate_basis(n_sites: int, n_up: int) -> list[int]:
    """Return sorted list of basis states (as ints) with exactly n_up set bits."""
    sites = list(range(n_sites))
    basis: list[int] = []
    for up_positions in combinations(sites, n_up):
        state = 0
        for pos in up_positions:
            state |= 1 << pos
        basis.append(state)
    basis.sort()
    return basis


def _basis_index_map(basis: list[int]) -> dict[int, int]:
    """Map from state-as-int to index in the basis list."""
    return {state: idx for idx, state in enumerate(basis)}


def _bit(state: int, pos: int) -> int:
    return (state >> pos) & 1


def _flip(state: int, pos: int) -> int:
    return state ^ (1 << pos)


def build_j1j2_hamiltonian_sparse(
    n_sites: int,
    nn_bonds: list[tuple[int, int]],
    nnn_bonds: list[tuple[int, int]],
    j1: float,
    j2: float,
    n_up: int,
) -> csr_matrix:
    """Build the sparse Hamiltonian for J1-J2 Heisenberg in fixed Sz sector.

    H = J1 Σ_{nn} Si·Sj + J2 Σ_{nnn} Si·Sj

    Each Si·Sj = Sz_i Sz_j + (1/2)(S+_i S-_j + S-_i S+_j)
    With Sz = σ - 1/2 where σ ∈ {0, 1}.
    """
    basis = _enumerate_basis(n_sites, n_up)
    dim = len(basis)
    idx_map = _basis_index_map(basis)

    H = lil_matrix((dim, dim), dtype=np.float64)

    # Combine all bonds with their couplings
    all_bonds: list[tuple[int, int, float]] = []
    for i, j in nn_bonds:
        all_bonds.append((i, j, j1))
    for i, j in nnn_bonds:
        all_bonds.append((i, j, j2))

    for state_idx, state in enumerate(basis):
        diag = 0.0
        for site_i, site_j, coupling in all_bonds:
            if coupling == 0.0:
                continue
            si = _bit(state, site_i)
            sj = _bit(state, site_j)

            # Diagonal: Sz_i Sz_j = (si - 0.5)(sj - 0.5)
            sz_i = si - 0.5
            sz_j = sj - 0.5
            diag += coupling * sz_i * sz_j

            # Off-diagonal: (1/2)(S+_i S-_j + S-_i S+_j)
            # Only acts when si != sj
            if si != sj:
                flipped = _flip(_flip(state, site_i), site_j)
                flipped_idx = idx_map[flipped]
                H[state_idx, flipped_idx] += 0.5 * coupling

        H[state_idx, state_idx] += diag

    return H.tocsr()


def exact_ground_state(
    lx: int,
    ly: int,
    j2: float,
    sector: str = "sz0",
    nn_bonds: list[tuple[int, int]] | None = None,
    nnn_bonds: list[tuple[int, int]] | None = None,
    k: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ground state energy and wavefunction.

    Returns:
        (energies, wavefunctions) where energies has shape (k,)
        and wavefunctions has shape (dim, k).
    """
    n_sites = lx * ly
    n_up = sector_n_up(sector)

    if nn_bonds is None:
        from prepare import build_nn_bonds
        nn_bonds = build_nn_bonds(lx, ly)
    if nnn_bonds is None:
        from prepare import build_nnn_bonds
        nnn_bonds = build_nnn_bonds(lx, ly)

    H = build_j1j2_hamiltonian_sparse(n_sites, nn_bonds, nnn_bonds, J1, j2, n_up)
    energies, psis = eigsh(H, k=k, which="SA")
    order = np.argsort(energies)
    return energies[order], psis[:, order]


def compute_reference_table(
    lx: int = 4,
    ly: int = 4,
    j2_values: list[float] | None = None,
) -> list[dict[str, object]]:
    """Compute ED reference energies for a grid of J2 values."""
    from prepare import build_nn_bonds, build_nnn_bonds

    if j2_values is None:
        j2_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    nn = build_nn_bonds(lx, ly)
    nnn = build_nnn_bonds(lx, ly)
    n_sites = lx * ly

    rows: list[dict[str, object]] = []
    for j2 in j2_values:
        for sector in ("sz0", "sz1"):
            n_up = sector_n_up(sector)
            dim = comb(n_sites, n_up)
            print(f"  ED: {lx}x{ly}, J2={j2:.2f}, {sector}, dim={dim}...", end=" ", flush=True)
            energies, _ = exact_ground_state(lx, ly, j2, sector, nn, nnn, k=2)
            e0 = float(energies[0])
            e1 = float(energies[1]) if len(energies) > 1 else None
            row = {
                "lx": lx,
                "ly": ly,
                "n_sites": n_sites,
                "j2": j2,
                "sector": sector,
                "n_up": n_up,
                "hilbert_dim": dim,
                "E0": e0,
                "E0_per_site": e0 / n_sites,
                "E1": e1,
                "E1_per_site": e1 / n_sites if e1 is not None else None,
            }
            rows.append(row)
            print(f"E0/N = {e0/n_sites:.6f}")
    return rows


if __name__ == "__main__":
    print(f"Computing ED reference table for {LX}x{LY} lattice...\n")
    table = compute_reference_table(LX, LY)
    print("\n" + "=" * 70)
    print(f"{'J2':>5} {'Sector':>6} {'Dim':>10} {'E0/N':>12} {'E1/N':>12} {'Gap/N':>12}")
    print("=" * 70)
    for row in table:
        gap_str = ""
        if row["E1_per_site"] is not None:
            gap = row["E1_per_site"] - row["E0_per_site"]
            gap_str = f"{gap:12.6f}"
        print(
            f"{row['j2']:5.2f} {row['sector']:>6} {row['hilbert_dim']:10d} "
            f"{row['E0_per_site']:12.6f} "
            f"{row['E1_per_site']:12.6f}" if row['E1_per_site'] is not None else "",
            gap_str,
        )

    out_path = ROOT / "analysis" / "ed_reference_table.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=2))
    print(f"\nSaved to {out_path}")
