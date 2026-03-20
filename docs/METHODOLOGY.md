# Methodology

This repository is the first worked example of a broader idea: using
autoresearch-style automated recipe search for quantum many-body physics.
The workflow is inspired by Andrej Karpathy's
[`autoresearch`](https://github.com/karpathy/autoresearch), but adapted to a
scientific setting where the physics definition and evaluation harness remain
fixed while only the mutable training recipe is searched.

## Physical Problem

The repository targets the spin-1/2 J1-J2 Heisenberg antiferromagnet on a
periodic square lattice:

```text
H = J1 Σ_nn S_i · S_j + J2 Σ_nnn S_i · S_j
```

Current public experiments focus on the 4x4 lattice, where exact
diagonalization is still available for validation.

## Fixed vs Mutable Components

Fixed components:

- `prepare.py`: lattice geometry, bond construction, Hamiltonian helpers, CLI defaults
- `exact_diag.py`: exact diagonalization reference data
- `controller/run_panel_eval.py`: short evaluation panel

Mutable component:

- `train.py`: neural quantum state, optimizer, estimators, scheduling, and training logic

This separation lets us search training recipes without changing the underlying
physics or the evaluation target.

## Search Protocols

### Short-budget panel search

The short-budget search evaluates `train.py` on a fixed panel of `J2` values and
keeps a modification only if the mean energy per site improves. This is useful
for fast recipe search, but it is not a substitute for a full post-training
benchmark.

### Post-training evaluation

After a promising recipe is identified, it is compared against a baseline on a
separate benchmark with exact references.

Current metrics:

- absolute energy error per site against exact diagonalization,
- local-energy variance `Var(E_loc) / N^2`,
- sector-gap error between `sz0` and `sz1`,
- sampled order diagnostics `m_Neel^2`, `m_stripe_x^2`, `m_stripe_y^2`.

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
