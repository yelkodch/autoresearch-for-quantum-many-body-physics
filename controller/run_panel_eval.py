"""Run a single evaluation panel: execute train.py for multiple J2 values."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "train.py"

# Import prepare from project root
sys.path.insert(0, str(ROOT))
from prepare import N_SITES, sample_random_j2


def run_one_delta(
    j2: float,
    sector: str,
    seed: int,
    output_dir: Path,
    max_steps: int = 1000,
    time_budget_s: float | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict | None:
    """Execute train.py for a single (j2, sector) and return the summary."""
    cmd = [
        sys.executable, str(TRAIN_PY),
        "--j2", str(j2),
        "--sector", sector,
        "--seed", str(seed),
        "--max-steps", str(max_steps),
        "--output-dir", str(output_dir),
    ]
    if time_budget_s is not None:
        cmd += ["--time-budget-s", str(time_budget_s)]

    import os
    env = {**os.environ, "TV_DEVICE": "cpu"}
    if extra_env:
        env.update(extra_env)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=int((time_budget_s or 1800) + 60),
            env=env,
        )
        if result.returncode != 0:
            print(f"  [ERROR] train.py failed for j2={j2}: {result.stderr[:500]}")
            return None
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text())
        return None
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] j2={j2}")
        return None
    except Exception as e:
        print(f"  [ERROR] j2={j2}: {e}")
        return None


def run_panel(
    campaign_dir: Path,
    iteration: int,
    panel_seed: int,
    max_steps: int = 1000,
    time_budget_s: float | None = None,
) -> dict:
    """Run a full panel evaluation and return the panel summary."""
    j2_values = sample_random_j2(panel_seed)
    iter_dir = campaign_dir / f"iteration_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, j2 in enumerate(j2_values):
        output_dir = iter_dir / f"delta_{idx}_j2_{j2:.3f}_sz0"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Panel [{idx+1}/{len(j2_values)}] j2={j2:.3f}...", end=" ", flush=True)
        t0 = time.time()
        summary = run_one_delta(j2, "sz0", panel_seed + idx, output_dir, max_steps, time_budget_s)
        dt = time.time() - t0
        if summary and "summary" in summary:
            s = summary["summary"]
            results.append({
                "j2": j2,
                "energy_per_site": s["energy_per_site"],
                "eval_energy_std_per_site": s["eval_energy_std_per_site"],
                "steps_completed": s["steps_completed"],
                "wall_time_s": dt,
            })
            print(f"E/N = {s['energy_per_site']:.4f} ({dt:.0f}s)")
        else:
            results.append({"j2": j2, "energy_per_site": 0.0, "error": True})
            print(f"FAILED ({dt:.0f}s)")

    valid = [r for r in results if "error" not in r]
    if valid:
        panel_score = sum(r["energy_per_site"] for r in valid) / len(valid)
    else:
        panel_score = 0.0

    panel_summary = {
        "iteration": iteration,
        "panel_seed": panel_seed,
        "j2_values": list(j2_values),
        "panel_score": panel_score,
        "results": results,
        "n_valid": len(valid),
        "n_total": len(results),
    }

    (iter_dir / "panel_summary.json").write_text(json.dumps(panel_summary, indent=2))
    return panel_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--panel-seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--time-budget-s", type=float, default=None)
    args = parser.parse_args()
    summary = run_panel(args.campaign_dir, args.iteration, args.panel_seed,
                        args.max_steps, args.time_budget_s)
    print(f"\nPanel score: {summary['panel_score']:.4f}")
