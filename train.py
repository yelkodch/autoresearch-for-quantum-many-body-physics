"""Mutable training recipe for J1-J2 2D Heisenberg experiments.

The agent CAN modify ANYTHING in this file EXCEPT:
  - parse_args() — CLI interface is fixed
  - RunSummary dataclass — output schema is fixed
  - The physics: local_energies must compute the correct J1-J2 Heisenberg
    energy using NN_BONDS and NNN_BONDS from prepare.py

Everything else — model architecture, optimizer, loss, sampling,
phase representation, batch size, etc — is free to change.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    DEVICE,
    J1,
    LX,
    LY,
    NN_BONDS,
    NNN_BONDS,
    N_SITES,
    PROJECT_ROOT,
    sector_n_up,
    seed_everything,
)

# ════════════════════════════════════════════════════════════════
# EVERYTHING BELOW IS MODIFIABLE BY THE AGENT
# ════════════════════════════════════════════════════════════════

# ── Model hyperparameters ──
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
FF_HIDDEN_DIM = 64
DROPOUT = 0.0

# ── Training hyperparameters ──
MAX_STEPS = 3000
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
BETAS = (0.9, 0.999)
GRAD_CLIP_NORM = 0.5

# ── LR schedule ──
LR_WARMUP_FRACTION = 0.1
LR_FINAL_MULTIPLIER = 0.2

# ── Evaluation ──
EVAL_SAMPLES = 512
EVAL_REPEATS = 5

# ── Sign/phase ──
SIGN_STRUCTURE_MODE = "checkerboard_marshall"

# ── Loss ──
BASELINE_TYPE = "median"
ADVANTAGE_TYPE = "centered"


# ── Model ──
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class NQSModel(nn.Module):
    """Autoregressive transformer for sampling spin configurations
    in a fixed Sz sector on a 2D lattice."""

    def __init__(self, n_sites: int, n_up: int) -> None:
        super().__init__()
        self.n_sites = n_sites
        self.n_up = n_up
        self.bos_token = 2
        self.token_embed = nn.Embedding(3, D_MODEL)  # 0, 1, BOS
        self.pos_embed = nn.Parameter(torch.zeros(1, n_sites, D_MODEL))
        self.blocks = nn.ModuleList(
            [TransformerBlock(D_MODEL, N_HEADS, FF_HIDDEN_DIM, DROPOUT) for _ in range(N_LAYERS)]
        )
        self.out_norm = nn.LayerNorm(D_MODEL)
        self.out_proj = nn.Linear(D_MODEL, 2)  # logits for spin 0 or 1

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def _build_input(self, configs: torch.Tensor) -> torch.Tensor:
        bos = torch.full((configs.shape[0], 1), self.bos_token, dtype=torch.long, device=configs.device)
        return torch.cat([bos, configs[:, :-1]], dim=1)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        seq_len = input_tokens.shape[1]
        x = self.token_embed(input_tokens) + self.pos_embed[:, :seq_len, :]
        mask = self._causal_mask(seq_len, input_tokens.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.out_proj(self.out_norm(x))

    def _step_valid_choices(self, ones_used: torch.Tensor, site: int) -> torch.Tensor:
        remaining = self.n_up - ones_used
        sites_left_after = self.n_sites - site - 1
        valid_zero = sites_left_after >= remaining
        valid_one = remaining > 0
        return torch.stack([valid_zero, valid_one], dim=-1)

    def _valid_mask(self, configs: torch.Tensor) -> torch.Tensor:
        batch, seq_len = configs.shape
        positions = torch.arange(seq_len, device=configs.device).unsqueeze(0).expand(batch, -1)
        ones_before = torch.cumsum(configs, dim=1) - configs
        sites_left_after = seq_len - positions - 1
        valid_zero = sites_left_after >= (self.n_up - ones_before)
        valid_one = (self.n_up - ones_before) > 0
        return torch.stack([valid_zero, valid_one], dim=-1)

    def log_prob(self, configs: torch.Tensor) -> torch.Tensor:
        input_tokens = self._build_input(configs)
        logits = self.forward(input_tokens)
        valid = self._valid_mask(configs)
        masked_logits = logits.masked_fill(~valid, float("-inf"))
        log_probs = F.log_softmax(masked_logits, dim=-1)
        chosen = torch.gather(log_probs, 2, configs.unsqueeze(-1)).squeeze(-1)
        return chosen.sum(dim=1)

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        configs = torch.zeros((batch_size, self.n_sites), dtype=torch.long, device=device)
        ones_used = torch.zeros(batch_size, dtype=torch.long, device=device)
        for site in range(self.n_sites):
            input_tokens = self._build_input(configs)
            logits = self.forward(input_tokens)[:, site, :]
            valid = self._step_valid_choices(ones_used, site)
            masked = logits.masked_fill(~valid, float("-inf"))
            probs = F.softmax(masked, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            configs[:, site] = sampled
            ones_used += sampled
        return configs


# ── Local energy ──
def _site_xy(site: int) -> tuple[int, int]:
    return site // LY, site % LY


def _checkerboard_label(site: int) -> int:
    x, y = _site_xy(site)
    return (x + y) & 1


def _stripe_x_label(site: int) -> int:
    x, _ = _site_xy(site)
    return x & 1


def _stripe_y_label(site: int) -> int:
    _, y = _site_xy(site)
    return y & 1


def fixed_gauge_sign_ratio(
    configs: torch.Tensor,
    bond_i: int,
    bond_j: int,
    mode: str = SIGN_STRUCTURE_MODE,
) -> torch.Tensor:
    """Return the sign ratio contributed by a fixed gauge in the wavefunction ansatz.

    This is part of the variational ansatz, not a modification of the Hamiltonian.
    The current baseline supports a few fixed gauges that the agent can later replace
    with a learned sign or phase network.
    """
    if mode == "none":
        value = 1.0
    elif mode == "checkerboard_marshall":
        value = -1.0 if _checkerboard_label(bond_i) != _checkerboard_label(bond_j) else 1.0
    elif mode == "stripe_x":
        value = -1.0 if _stripe_x_label(bond_i) != _stripe_x_label(bond_j) else 1.0
    elif mode == "stripe_y":
        value = -1.0 if _stripe_y_label(bond_i) != _stripe_y_label(bond_j) else 1.0
    else:
        raise ValueError(f"Unknown SIGN_STRUCTURE_MODE: {mode}")
    return torch.full((configs.shape[0],), value, dtype=torch.float32, device=configs.device)


def sign_ratio(model: NQSModel, configs: torch.Tensor, bond_i: int, bond_j: int) -> torch.Tensor:
    """Hook for the sign/phase part of the ansatz.

    Today this is a fixed gauge. The agent can later replace this helper with a learned
    sign or phase model without touching the Hamiltonian implementation.
    """
    del model
    return fixed_gauge_sign_ratio(configs, bond_i, bond_j)


def log_amplitude(model: NQSModel, configs: torch.Tensor) -> torch.Tensor:
    """log|psi(sigma)| = 0.5 * log p(sigma)."""
    return 0.5 * model.log_prob(configs)


def local_energies(
    model: NQSModel, configs: torch.Tensor, j2: float
) -> torch.Tensor:
    """Compute local energies for the J1-J2 Heisenberg model.

    H = J1 Σ_{nn} Si·Sj + J2 Σ_{nnn} Si·Sj

    Each bond contributes:
      diagonal: J * Sz_i Sz_j = J * (±1/4) depending on whether spins are same
      off-diagonal: J/2 * psi(flipped)/psi(original) * sign_ratio [if spins differ]
    """
    device = configs.device
    log_amp = log_amplitude(model, configs)
    local_e = torch.zeros(configs.shape[0], dtype=torch.float32, device=device)

    # Build combined bond list: (site_i, site_j, coupling)
    bonds: list[tuple[int, int, float]] = []
    for i, j in NN_BONDS:
        bonds.append((i, j, J1))
    if j2 != 0.0:
        for i, j in NNN_BONDS:
            bonds.append((i, j, j2))

    for site_i, site_j, coupling in bonds:
        si = configs[:, site_i]
        sj = configs[:, site_j]
        same = si == sj

        # Diagonal: Sz_i Sz_j = (s - 0.5)(s - 0.5) = ±0.25
        local_e += torch.where(same, 0.25 * coupling, -0.25 * coupling)

        # Off-diagonal: only when spins differ
        flippable = ~same
        if not torch.any(flippable):
            continue

        flipped = configs[flippable].clone()
        flipped[:, site_i] = 1 - flipped[:, site_i]
        flipped[:, site_j] = 1 - flipped[:, site_j]

        flipped_log_amp = log_amplitude(model, flipped)
        sign = sign_ratio(model, configs[flippable], site_i, site_j)
        ratio = torch.exp(flipped_log_amp - log_amp[flippable])
        offdiag = 0.5 * coupling * sign * ratio
        local_e[flippable] = local_e[flippable] + offdiag

    return local_e


# ── Training ──
def lr_schedule(step: int, total_steps: int) -> float:
    progress = step / max(1, total_steps)
    if progress < LR_WARMUP_FRACTION and LR_WARMUP_FRACTION > 0:
        return LEARNING_RATE * (progress / LR_WARMUP_FRACTION)
    tail = (progress - LR_WARMUP_FRACTION) / max(1e-8, 1.0 - LR_WARMUP_FRACTION)
    cosine = 0.5 * (1.0 + math.cos(math.pi * tail))
    return LEARNING_RATE * (LR_FINAL_MULTIPLIER + (1.0 - LR_FINAL_MULTIPLIER) * cosine)


def training_step(
    model: NQSModel, optimizer: torch.optim.Optimizer, j2: float, device: torch.device
) -> float:
    configs = model.sample(BATCH_SIZE, device)
    log_p = model.log_prob(configs)
    energies = local_energies(model, configs, j2)

    if BASELINE_TYPE == "mean":
        baseline = energies.mean()
    elif BASELINE_TYPE == "median":
        baseline = energies.median()
    else:
        baseline = energies.mean()

    advantages = (energies - baseline).detach()
    if ADVANTAGE_TYPE == "zscore":
        advantages = advantages / (advantages.std(unbiased=False) + 1e-6)

    loss = torch.mean(2.0 * advantages * log_p)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    return float(energies.mean().item())


@torch.no_grad()
def evaluate_energy(
    model: NQSModel, j2: float, device: torch.device
) -> tuple[float, float, list[float]]:
    estimates: list[float] = []
    for _ in range(EVAL_REPEATS):
        configs = model.sample(EVAL_SAMPLES, device)
        energies = local_energies(model, configs, j2)
        estimates.append(float(energies.mean().item()))
    mean_e = sum(estimates) / len(estimates)
    std_e = float(statistics.pstdev(estimates)) if len(estimates) > 1 else 0.0
    return mean_e, std_e, estimates


# ── CLI (FIXED — do not modify) ──
@dataclass
class RunSummary:
    timestamp: str
    device: str
    j2: float
    sector: str
    n_sites: int
    lx: int
    ly: int
    n_up: int
    seed: int
    max_steps: int
    steps_completed: int
    train_wall_time_s: float
    predicted_energy: float
    energy_per_site: float
    eval_energy_std: float
    eval_energy_std_per_site: float
    model_params: int
    batch_size: int
    learning_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="J1-J2 NQS training")
    parser.add_argument("--j2", type=float, required=True)
    parser.add_argument("--sector", choices=("sz0", "sz1"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--time-budget-s", type=float, default=None,
                        help="Optional wall-clock time cap in seconds")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="manual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(DEVICE)
    n_up = sector_n_up(args.sector)
    model = NQSModel(N_SITES, n_up).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=BETAS, weight_decay=WEIGHT_DECAY)

    max_steps = args.max_steps
    start = time.perf_counter()
    steps_done = 0
    for step in range(max_steps):
        if args.time_budget_s and (time.perf_counter() - start) >= args.time_budget_s:
            break
        lr = lr_schedule(step, max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr
        training_step(model, optimizer, args.j2, device)
        steps_done += 1

    predicted_energy, eval_std, _ = evaluate_energy(model, args.j2, device)
    wall_time = time.perf_counter() - start

    summary = RunSummary(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        device=str(device),
        j2=float(args.j2),
        sector=args.sector,
        n_sites=N_SITES,
        lx=LX,
        ly=LY,
        n_up=n_up,
        seed=args.seed,
        max_steps=max_steps,
        steps_completed=steps_done,
        train_wall_time_s=wall_time,
        predicted_energy=predicted_energy,
        energy_per_site=predicted_energy / N_SITES,
        eval_energy_std=eval_std,
        eval_energy_std_per_site=eval_std / N_SITES,
        model_params=n_params,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )

    output_dir = args.output_dir or (PROJECT_ROOT / "results" / f"{args.tag}_{args.sector}_{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": asdict(summary)}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
