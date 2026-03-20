# Methodology

This repository is the first worked example of a broader idea: using
autoresearch-style automated recipe search for quantum many-body physics.
The workflow is adapted to a scientific setting where the physics definition and
evaluation harness remain fixed while only the mutable training recipe is
searched.

## Physical Problem

The repository targets the spin-1/2 J1-J2 Heisenberg antiferromagnet on a
periodic square lattice:

```text
H = J1 Σ_nn S_i · S_j + J2 Σ_nnn S_i · S_j
```

Current public experiments focus on the 4x4 lattice, where exact
diagonalization is still available for validation.

## Variational Perspective

The central object is a variational wavefunction `psi_theta(sigma)` represented
by a neural quantum state. The aim is to approximate the ground state by
minimizing the variational energy

```text
E(theta) = <psi_theta|H|psi_theta> / <psi_theta|psi_theta>.
```

By the variational principle, this quantity is always an upper bound on the true
ground-state energy. That is why energy is the primary target metric: if the
ansatz and optimization improve, the estimated energy should move downward
toward the exact reference.

The project keeps the Hamiltonian and lattice fixed and only searches over the
recipe used to represent and optimize `psi_theta`.

## Fixed vs Mutable Components

Fixed components:

- `prepare.py`: lattice geometry, bond construction, Hamiltonian helpers, CLI defaults
- `exact_diag.py`: exact diagonalization reference data
- `controller/run_panel_eval.py`: short evaluation panel

Mutable component:

- `train.py`: neural quantum state, optimizer, estimators, scheduling, and training logic

This separation lets us search training recipes without changing the underlying
physics or the evaluation target.

## Why The Metrics Matter

Different metrics are included because low energy alone is not enough to judge a
variational state.

- Energy error versus exact diagonalization:
  the clearest measure of how close the model is to the exact 4x4 ground state.
- Local-energy variance `Var(E_loc) / N^2`:
  an exact eigenstate has zero local-energy variance, so this is a useful
  quality check even when energies are close.
- Sector gap between `sz0` and `sz1`:
  a compact way to test whether the model reproduces low-lying structure beyond
  a single sector energy.
- Order diagnostics `m_Neel^2`, `m_stripe_x^2`, `m_stripe_y^2`:
  these show whether the wavefunction captures the expected qualitative regime
  across low- and high-frustration anchor points.

## Search Protocols

### Short-budget panel search

The short-budget search evaluates `train.py` on a fixed panel of `J2` values and
keeps a modification only if the mean energy per site improves. This is useful
for fast recipe search, but it is not a substitute for a full post-training
benchmark.

In practice, this stage asks a narrow question:

- under a small computational budget, which recipe changes improve optimization
  behavior most reliably?

### Post-training evaluation

After a promising recipe is identified, it is compared against a baseline on a
separate benchmark with exact references.

Current metrics:

- absolute energy error per site against exact diagonalization,
- local-energy variance `Var(E_loc) / N^2`,
- sector-gap error between `sz0` and `sz1`,
- sampled order diagnostics `m_Neel^2`, `m_stripe_x^2`, `m_stripe_y^2`.

This stage asks a broader question:

- does the short-budget search winner remain good when trained longer and judged
  against physically meaningful reference quantities?

## Procedure

The current workflow has three layers.

1. Define the fixed problem.
   `prepare.py` and `exact_diag.py` specify the lattice, couplings, bond
   structure, and exact 4x4 references.

2. Search over the mutable recipe.
   `train.py` is modified either by the API-driven loop or by the direct-edit
   keep/discard workflow. These searches target short-budget improvement on a
   stable panel.

3. Evaluate the winning recipe more seriously.
   `controller/post_training_eval.py` compares the baseline and champion recipes
   against exact diagonalization and order diagnostics.

This split is deliberate: it avoids conflating "optimizes faster under a small
budget" with "is a better final variational model."

## Evaluation Anchors

The current anchor points are chosen to cover different regimes:

- `J2 = 0.0`: unfrustrated antiferromagnet
- `J2 = 0.5`: strongly frustrated regime
- `J2 = 1.0`: large-`J2` stripe-dominated regime

The post-training evaluator can also run denser grids such as
`0.0, 0.4, 0.5, 0.6, 1.0`.

## Why This Benchmark Matters

The 4x4 setting is intentionally modest:

- it is small enough for exact validation,
- it is still nontrivial because frustration and sign structure matter,
- it is a good controlled testbed for automated recipe search before moving to
  larger systems without exact references.

In that sense, the J1-J2 example is not just a target problem; it is also the
methodological prototype used to test how well autoresearch-style iteration
transfers to quantum many-body benchmarks.

## Current Limitations

- finite-size effects are large on 4x4,
- short-budget panel winners may not remain winners under long training,
- the current repository is best viewed as a controlled methodology benchmark,
  not yet a state-of-the-art large-scale solver.
