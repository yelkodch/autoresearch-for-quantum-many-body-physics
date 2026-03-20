# J1-J2 Heisenberg 2D Research Prompt

You are improving a neural quantum state for the frustrated J1-J2 Heisenberg
model on a 2D square lattice (currently 4×4, periodic boundary conditions).

## The physics

H = J1 Σ_{nn} Si·Sj + J2 Σ_{nnn} Si·Sj

- J1 = 1 (nearest-neighbor coupling, fixed)
- J2 ∈ [0, 0.7] (next-nearest-neighbor = diagonals, the family parameter)
- Lattice: square LX × LY with periodic boundary conditions (torus)
- Spin-1/2, Sz=0 sector (equal number of up and down spins)

### Phase diagram
- J2/J1 < 0.4 → Néel antiferromagnet (Marshall sign convention works)
- J2/J1 ≈ 0.4–0.6 → Frustrated regime (complex sign structure)
- J2/J1 > 0.6 → Columnar/stripe order

### Reference energies (4×4 lattice, exact diagonalization)
- J2=0.0: E/N = -0.7018
- J2=0.1: E/N = -0.6598
- J2=0.2: E/N = -0.6199
- J2=0.3: E/N = -0.5830
- J2=0.4: E/N = -0.5511
- J2=0.5: E/N = -0.5286
- J2=0.6: E/N = -0.5259
- J2=0.7: E/N = -0.5639

## Mutable Scope (train.py — modify freely)

You can change ANYTHING in train.py:
- Model architecture: transformer, CNN, RNN, MLP, or any combination
- Optimizer: Adam, SGD, natural gradient, stochastic reconfiguration, anything
- Loss function: REINFORCE, PPO, contrastive, anything
- Phase/sign representation: Marshall sign, learned phase network, complex amplitudes
- Batch size, number of training steps, learning rate schedule
- Variance reduction techniques (baselines, control variates)
- ANY code changes that make the energy lower

## Fixed Scope (do NOT change)

- `prepare.py` — lattice geometry, bond lists, CLI infrastructure
- The Hamiltonian physics: local_energies must compute the correct J1-J2 
  Heisenberg energy using NN_BONDS and NNN_BONDS from prepare.py
- Output format: `RunSummary` dataclass and summary.json must match the schema
- `parse_args()` function signature

## Score

score = mean(E / N_SITES) averaged over 3 random J2 values in [0, 0.7]

**Lower is better** (more negative energy per site).

## Hints for good results

1. 256-1024 samples per step is standard for 4×4
2. 1000+ steps is minimum for convergence
3. Marshall sign only works for J2 ≈ 0. For J2 > 0.3, you NEED a phase network
4. Stochastic Reconfiguration converges 5-10× faster than REINFORCE
5. The off-diagonal local energy requires one forward pass per flippable bond.
   For 4×4 with ~64 bonds, this is manageable. Batch all flips for efficiency.
6. Start simple, iterate. Don't propose giant rewrites that break everything.
7. The model currently has ~18K parameters. That's reasonable for 4×4.

## Common failure modes

- Batch too small → gradient noise → no convergence
- No phase network → wrong sign structure → energy plateaus above -0.50 for J2 > 0.3
- Model too large for the training budget → underfitting
- Complex code that introduces bugs → smoke test failure
- Forgetting periodic boundary conditions (bonds wrap around)
