"""Post-training evaluation for J1-J2 train.py recipes.

This script compares a baseline recipe against the current winning recipe after
longer training budgets, using metrics that are meaningful for the square-lattice
J1-J2 Heisenberg model on 4x4:

- energy error against exact diagonalization in fixed Sz sectors
- local energy variance (variational quality)
- gap between sz1 and sz0 sectors
- sampled order parameters for Néel and stripe tendencies

It does not mutate train.py. Instead it loads recipe snapshots from disk and
trains them in-process for reproducible evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("TV_DEVICE", "cpu")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exact_diag import exact_ground_state
from prepare import DEVICE, N_SITES, RESULTS_DIR, sector_n_up, seed_everything

DEFAULT_J2_VALUES = (0.0, 0.4, 0.5, 0.6, 1.0)
DEFAULT_SEEDS = (11, 22, 33)
DEFAULT_STEPS = (50, 200, 500)
ANCHOR_J2_FOR_ORDER = {0.0, 1.0}


@dataclass
class ConfigSpec:
    label: str
    path: str


@dataclass
class RunMetrics:
    config_label: str
    config_path: str
    max_steps: int
    j2: float
    seed: int
    sector: str
    steps_completed: int
    train_wall_time_s: float
    energy_per_site: float
    eval_energy_std_per_site: float
    local_energy_var: float
    local_energy_var_per_site2: float
    m_neel_sq: float | None
    m_stripe_x_sq: float | None
    m_stripe_y_sq: float | None
    exact_energy_per_site: float
    abs_energy_error_per_site: float


def load_recipe_module(path: Path):
    module_name = f"train_recipe_{hashlib.sha1(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def train_recipe(
    module: Any,
    *,
    j2: float,
    sector: str,
    seed: int,
    max_steps: int,
    time_budget_s: float | None,
) -> tuple[torch.nn.Module, int, float]:
    seed_everything(seed)
    device = torch.device(DEVICE)
    n_up = sector_n_up(sector)
    model = module.NQSModel(N_SITES, n_up).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=module.LEARNING_RATE,
        betas=module.BETAS,
        weight_decay=module.WEIGHT_DECAY,
    )

    start = time.perf_counter()
    steps_done = 0
    for step in range(max_steps):
        if time_budget_s is not None and (time.perf_counter() - start) >= time_budget_s:
            break
        lr = module.lr_schedule(step, max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        module.training_step(model, optimizer, j2, device)
        steps_done += 1
    wall = time.perf_counter() - start
    return model, steps_done, wall


def checkerboard_weights(n_sites: int) -> torch.Tensor:
    side = int(round(math.sqrt(n_sites)))
    if side * side != n_sites:
        raise ValueError(f"Expected square lattice size, got n_sites={n_sites}")
    weights = []
    for site in range(n_sites):
        x, y = divmod(site, side)
        weights.append(1.0 if ((x + y) % 2 == 0) else -1.0)
    return torch.tensor(weights, dtype=torch.float32)


def stripe_x_weights(n_sites: int) -> torch.Tensor:
    side = int(round(math.sqrt(n_sites)))
    weights = []
    for site in range(n_sites):
        x, _ = divmod(site, side)
        weights.append(1.0 if (x % 2 == 0) else -1.0)
    return torch.tensor(weights, dtype=torch.float32)


def stripe_y_weights(n_sites: int) -> torch.Tensor:
    side = int(round(math.sqrt(n_sites)))
    weights = []
    for site in range(n_sites):
        _, y = divmod(site, side)
        weights.append(1.0 if (y % 2 == 0) else -1.0)
    return torch.tensor(weights, dtype=torch.float32)


def sampled_order_parameters(configs: torch.Tensor) -> tuple[float, float, float]:
    sz = configs.to(torch.float32) - 0.5
    denom = float(configs.shape[1] ** 2)
    neel = (sz * checkerboard_weights(configs.shape[1]).to(configs.device)).sum(dim=1)
    stripe_x = (sz * stripe_x_weights(configs.shape[1]).to(configs.device)).sum(dim=1)
    stripe_y = (sz * stripe_y_weights(configs.shape[1]).to(configs.device)).sum(dim=1)
    return (
        float((neel.square().mean() / denom).item()),
        float((stripe_x.square().mean() / denom).item()),
        float((stripe_y.square().mean() / denom).item()),
    )


def evaluate_trained_model(
    module: Any,
    model: torch.nn.Module,
    *,
    j2: float,
    sector: str,
    metric_samples: int,
    metric_repeats: int,
) -> dict[str, float | None]:
    device = torch.device(DEVICE)
    mean_e, std_e, _ = module.evaluate_energy(model, j2, device)

    variances: list[float] = []
    neel_vals: list[float] = []
    stripe_x_vals: list[float] = []
    stripe_y_vals: list[float] = []
    for _ in range(metric_repeats):
        configs = model.sample(metric_samples, device)
        energies = module.local_energies(model, configs, j2)
        variances.append(float(energies.var(unbiased=False).item()))
        if sector == "sz0":
            mn, msx, msy = sampled_order_parameters(configs)
            neel_vals.append(mn)
            stripe_x_vals.append(msx)
            stripe_y_vals.append(msy)

    return {
        "energy_per_site": mean_e / N_SITES,
        "eval_energy_std_per_site": std_e / N_SITES,
        "local_energy_var": float(sum(variances) / len(variances)),
        "local_energy_var_per_site2": float(sum(variances) / len(variances) / (N_SITES ** 2)),
        "m_neel_sq": float(sum(neel_vals) / len(neel_vals)) if neel_vals else None,
        "m_stripe_x_sq": float(sum(stripe_x_vals) / len(stripe_x_vals)) if stripe_x_vals else None,
        "m_stripe_y_sq": float(sum(stripe_y_vals) / len(stripe_y_vals)) if stripe_y_vals else None,
    }


def enumerate_basis(n_sites: int, n_up: int) -> list[int]:
    basis: list[int] = []
    for up_positions in combinations(range(n_sites), n_up):
        state = 0
        for pos in up_positions:
            state |= 1 << pos
        basis.append(state)
    basis.sort()
    return basis


def bit(state: int, pos: int) -> int:
    return (state >> pos) & 1


def exact_order_parameters(j2: float) -> dict[str, float]:
    energies, psi = exact_ground_state(4, 4, j2, sector="sz0", k=1)
    basis = enumerate_basis(16, sector_n_up("sz0"))
    prob = np.abs(psi[:, 0]) ** 2
    neel_w = checkerboard_weights(16).numpy()
    stripe_x_w = stripe_x_weights(16).numpy()
    stripe_y_w = stripe_y_weights(16).numpy()

    m_neel: list[float] = []
    m_stripe_x: list[float] = []
    m_stripe_y: list[float] = []
    for state in basis:
        sz = np.array([bit(state, i) - 0.5 for i in range(16)], dtype=np.float64)
        m_neel.append(float(np.dot(sz, neel_w)))
        m_stripe_x.append(float(np.dot(sz, stripe_x_w)))
        m_stripe_y.append(float(np.dot(sz, stripe_y_w)))

    denom = 16.0 * 16.0
    return {
        "energy_per_site": float(energies[0] / 16.0),
        "m_neel_sq": float(np.sum(prob * np.square(m_neel)) / denom),
        "m_stripe_x_sq": float(np.sum(prob * np.square(m_stripe_x)) / denom),
        "m_stripe_y_sq": float(np.sum(prob * np.square(m_stripe_y)) / denom),
    }


def compute_exact_cache(j2_values: list[float]) -> dict[str, dict[float, float]]:
    exact_energy_sz0: dict[float, float] = {}
    exact_energy_sz1: dict[float, float] = {}
    exact_gap: dict[float, float] = {}
    order_metrics: dict[float, dict[str, float]] = {}
    for j2 in j2_values:
        e0_sz0, _ = exact_ground_state(4, 4, j2, sector="sz0", k=1)
        e0_sz1, _ = exact_ground_state(4, 4, j2, sector="sz1", k=1)
        exact_energy_sz0[j2] = float(e0_sz0[0] / 16.0)
        exact_energy_sz1[j2] = float(e0_sz1[0] / 16.0)
        exact_gap[j2] = float(e0_sz1[0] - e0_sz0[0])
        order_metrics[j2] = exact_order_parameters(j2)
    return {
        "sz0": exact_energy_sz0,
        "sz1": exact_energy_sz1,
        "gap": exact_gap,
        "order": order_metrics,
    }


def run_evaluation(
    config: ConfigSpec,
    *,
    j2_values: list[float],
    seeds: list[int],
    steps: list[int],
    time_budget_s: float | None,
    metric_samples: int,
    metric_repeats: int,
    exact_cache: dict[str, dict[float, float] | dict[float, dict[str, float]]],
) -> list[RunMetrics]:
    recipe_path = Path(config.path)
    if not recipe_path.is_absolute():
        recipe_path = ROOT / recipe_path
    module = load_recipe_module(recipe_path)
    all_runs: list[RunMetrics] = []
    for max_steps in steps:
        for j2 in j2_values:
            for seed in seeds:
                for sector in ("sz0", "sz1"):
                    model, steps_done, wall = train_recipe(
                        module,
                        j2=j2,
                        sector=sector,
                        seed=seed,
                        max_steps=max_steps,
                        time_budget_s=time_budget_s,
                    )
                    metrics = evaluate_trained_model(
                        module,
                        model,
                        j2=j2,
                        sector=sector,
                        metric_samples=metric_samples,
                        metric_repeats=metric_repeats,
                    )
                    exact_energy = exact_cache[sector][j2]  # type: ignore[index]
                    all_runs.append(
                        RunMetrics(
                            config_label=config.label,
                            config_path=str(recipe_path),
                            max_steps=max_steps,
                            j2=j2,
                            seed=seed,
                            sector=sector,
                            steps_completed=steps_done,
                            train_wall_time_s=wall,
                            energy_per_site=float(metrics["energy_per_site"]),
                            eval_energy_std_per_site=float(metrics["eval_energy_std_per_site"]),
                            local_energy_var=float(metrics["local_energy_var"]),
                            local_energy_var_per_site2=float(metrics["local_energy_var_per_site2"]),
                            m_neel_sq=metrics["m_neel_sq"],  # type: ignore[arg-type]
                            m_stripe_x_sq=metrics["m_stripe_x_sq"],  # type: ignore[arg-type]
                            m_stripe_y_sq=metrics["m_stripe_y_sq"],  # type: ignore[arg-type]
                            exact_energy_per_site=float(exact_energy),
                            abs_energy_error_per_site=abs(float(metrics["energy_per_site"]) - float(exact_energy)),
                        )
                    )
                    del model
    return all_runs


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def summarize_runs(
    runs: list[RunMetrics],
    *,
    exact_cache: dict[str, dict[float, float] | dict[float, dict[str, float]]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[RunMetrics]] = {}
    for run in runs:
        grouped.setdefault((run.config_label, run.max_steps), []).append(run)

    summary: dict[str, Any] = {"overall": [], "per_j2": [], "per_seed_gap": []}
    for (config_label, max_steps), group in sorted(grouped.items()):
        sz0_runs = [r for r in group if r.sector == "sz0"]
        sz1_runs = [r for r in group if r.sector == "sz1"]

        gaps: list[dict[str, Any]] = []
        for seed in sorted({r.seed for r in group}):
            for j2 in sorted({r.j2 for r in group}):
                run0 = next(r for r in sz0_runs if r.seed == seed and r.j2 == j2)
                run1 = next(r for r in sz1_runs if r.seed == seed and r.j2 == j2)
                gap = (run1.energy_per_site - run0.energy_per_site) * N_SITES
                exact_gap = float(exact_cache["gap"][j2])  # type: ignore[index]
                gap_err = abs(gap - exact_gap)
                gaps.append(
                    {
                        "config_label": config_label,
                        "max_steps": max_steps,
                        "seed": seed,
                        "j2": j2,
                        "gap": gap,
                        "exact_gap": exact_gap,
                        "abs_gap_error": gap_err,
                    }
                )

        summary["per_seed_gap"].extend(gaps)
        summary["overall"].append(
            {
                "config_label": config_label,
                "max_steps": max_steps,
                "mae_energy_per_site_sz0": mean([r.abs_energy_error_per_site for r in sz0_runs]),
                "max_energy_error_per_site_sz0": max(r.abs_energy_error_per_site for r in sz0_runs),
                "mean_local_energy_var_per_site2_sz0": mean([r.local_energy_var_per_site2 for r in sz0_runs]),
                "mae_gap": mean([g["abs_gap_error"] for g in gaps]),
                "mean_train_wall_time_s": mean([r.train_wall_time_s for r in group]),
            }
        )

        for j2 in sorted({r.j2 for r in group}):
            sz0_j = [r for r in sz0_runs if r.j2 == j2]
            exact_order = exact_cache["order"][j2]  # type: ignore[index]
            summary["per_j2"].append(
                {
                    "config_label": config_label,
                    "max_steps": max_steps,
                    "j2": j2,
                    "exact_energy_per_site_sz0": float(exact_cache["sz0"][j2]),  # type: ignore[index]
                    "mean_energy_per_site_sz0": mean([r.energy_per_site for r in sz0_j]),
                    "std_energy_per_site_sz0": float(statistics.pstdev([r.energy_per_site for r in sz0_j])) if len(sz0_j) > 1 else 0.0,
                    "mae_energy_per_site_sz0": mean([r.abs_energy_error_per_site for r in sz0_j]),
                    "mean_local_energy_var_per_site2_sz0": mean([r.local_energy_var_per_site2 for r in sz0_j]),
                    "mean_m_neel_sq": mean([r.m_neel_sq for r in sz0_j if r.m_neel_sq is not None]),
                    "mean_m_stripe_x_sq": mean([r.m_stripe_x_sq for r in sz0_j if r.m_stripe_x_sq is not None]),
                    "mean_m_stripe_y_sq": mean([r.m_stripe_y_sq for r in sz0_j if r.m_stripe_y_sq is not None]),
                    "exact_m_neel_sq": exact_order["m_neel_sq"],
                    "exact_m_stripe_x_sq": exact_order["m_stripe_x_sq"],
                    "exact_m_stripe_y_sq": exact_order["m_stripe_y_sq"],
                    "regime_anchor": j2 in ANCHOR_J2_FOR_ORDER,
                }
            )
    return summary


def write_summary_markdown(
    output_dir: Path,
    summary: dict[str, Any],
    configs: list[ConfigSpec],
    j2_values: list[float],
    seeds: list[int],
    steps: list[int],
) -> None:
    def display_path(path_text: str) -> str:
        path = Path(path_text)
        try:
            return str(path.resolve().relative_to(ROOT))
        except Exception:
            return path.name

    lines = [
        "# Post-Training Evaluation",
        "",
        "## Setup",
        f"- configs: {', '.join(f'{c.label}={display_path(c.path)}' for c in configs)}",
        f"- j2_values: {j2_values}",
        f"- seeds: {seeds}",
        f"- step_budgets: {steps}",
        "",
        "## Overall Metrics",
        "",
        "| config | steps | MAE E/N (sz0) | Max | MAE gap | mean Var(E_loc)/N^2 | mean wall time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["overall"]:
        lines.append(
            f"| {row['config_label']} | {row['max_steps']} | {row['mae_energy_per_site_sz0']:.6f} | "
            f"{row['max_energy_error_per_site_sz0']:.6f} | {row['mae_gap']:.6f} | "
            f"{row['mean_local_energy_var_per_site2_sz0']:.6f} | {row['mean_train_wall_time_s']:.1f} |"
        )

    lines += [
        "",
        "## Per-J2 Metrics",
        "",
        "| config | steps | J2 | mean E/N | exact E/N | MAE | mean Var(E_loc)/N^2 | mean m_Neel^2 | mean m_stripe^2 | exact m_Neel^2 | exact m_stripe^2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_j2"]:
        model_stripe = max(row["mean_m_stripe_x_sq"], row["mean_m_stripe_y_sq"])
        exact_stripe = max(row["exact_m_stripe_x_sq"], row["exact_m_stripe_y_sq"])
        lines.append(
            f"| {row['config_label']} | {row['max_steps']} | {row['j2']:.2f} | "
            f"{row['mean_energy_per_site_sz0']:.6f} | {row['exact_energy_per_site_sz0']:.6f} | "
            f"{row['mae_energy_per_site_sz0']:.6f} | {row['mean_local_energy_var_per_site2_sz0']:.6f} | "
            f"{row['mean_m_neel_sq']:.6f} | {model_stripe:.6f} | {row['exact_m_neel_sq']:.6f} | {exact_stripe:.6f} |"
        )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def maybe_make_plot(output_dir: Path, summary: dict[str, Any]) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    per_j2 = summary["per_j2"]
    if not per_j2:
        return None
    combos = sorted({(row["config_label"], row["max_steps"]) for row in per_j2})
    j2_values = sorted({row["j2"] for row in per_j2})

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    for config_label, max_steps in combos:
        rows = [row for row in per_j2 if row["config_label"] == config_label and row["max_steps"] == max_steps]
        rows.sort(key=lambda x: x["j2"])
        axes[0].plot(
            [r["j2"] for r in rows],
            [r["mae_energy_per_site_sz0"] for r in rows],
            marker="o",
            label=f"{config_label} ({max_steps} steps)",
        )
        axes[1].plot(
            [r["j2"] for r in rows],
            [r["mean_local_energy_var_per_site2_sz0"] for r in rows],
            marker="o",
            label=f"{config_label} ({max_steps} steps)",
        )

    axes[0].set_title("Energy Error vs ED")
    axes[0].set_xlabel("J2/J1")
    axes[0].set_ylabel("MAE(E/N)")
    axes[0].legend()

    axes[1].set_title("Variational Quality")
    axes[1].set_xlabel("J2/J1")
    axes[1].set_ylabel("Var(E_loc) / N^2")
    axes[1].legend()

    plot_path = output_dir / "post_training_summary.png"
    fig.savefig(plot_path, dpi=180)
    return str(plot_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training evaluation for baseline vs champion J1-J2 recipes")
    parser.add_argument("--baseline-path", type=Path, default=None)
    parser.add_argument("--candidate-path", type=Path, default=None)
    parser.add_argument("--baseline-label", type=str, default="baseline")
    parser.add_argument("--candidate-label", type=str, default="champion")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / f"post_training_eval_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--j2-values", type=float, nargs="+", default=list(DEFAULT_J2_VALUES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--time-budget-s", type=float, default=None)
    parser.add_argument("--metric-samples", type=int, default=2048)
    parser.add_argument("--metric-repeats", type=int, default=2)
    args = parser.parse_args()

    if args.baseline_path is None or args.candidate_path is None:
        raise SystemExit(
            "Provide --baseline-path and --candidate-path, for example the "
            "initial_train.py and best_train.py snapshots from a completed direct-edit campaign."
        )

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ConfigSpec(args.baseline_label, str(args.baseline_path)),
        ConfigSpec(args.candidate_label, str(args.candidate_path)),
    ]
    exact_cache = compute_exact_cache(list(args.j2_values))

    runs: list[RunMetrics] = []
    for config in configs:
        print(f"\n=== Evaluating {config.label} from {config.path} ===", flush=True)
        runs.extend(
            run_evaluation(
                config,
                j2_values=list(args.j2_values),
                seeds=list(args.seeds),
                steps=list(args.steps),
                time_budget_s=args.time_budget_s,
                metric_samples=args.metric_samples,
                metric_repeats=args.metric_repeats,
                exact_cache=exact_cache,
            )
        )

    runs_path = output_dir / "runs.jsonl"
    with runs_path.open("w") as handle:
        for run in runs:
            handle.write(json.dumps(asdict(run), sort_keys=True) + "\n")

    summary = summarize_runs(runs, exact_cache=exact_cache)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (output_dir / "exact_cache.json").write_text(json.dumps(exact_cache, indent=2, sort_keys=True))
    write_summary_markdown(
        output_dir,
        summary,
        configs=configs,
        j2_values=list(args.j2_values),
        seeds=list(args.seeds),
        steps=list(args.steps),
    )
    plot_path = maybe_make_plot(output_dir, summary)

    payload = {
        "output_dir": str(output_dir),
        "n_runs": len(runs),
        "configs": [asdict(c) for c in configs],
        "j2_values": list(args.j2_values),
        "seeds": list(args.seeds),
        "steps": list(args.steps),
        "plot_path": plot_path,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
