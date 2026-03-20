"""Sequential batch runner for direct-edit J1-J2 campaigns.

This script applies a queue of conservative hyperparameter edits to train.py,
runs one keep/discard evaluation per edit, and always branches from the current
best snapshot stored in the campaign directory.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "train.py"
RUN_ONE = ROOT / "controller" / "run_direct_experiment.py"


VARIANTS: list[tuple[str, dict[str, object]]] = [
    ("lower learning rate to 7e-4", {"LEARNING_RATE": 7e-4}),
    ("lower learning rate to 5e-4", {"LEARNING_RATE": 5e-4}),
    ("raise learning rate to 8e-4", {"LEARNING_RATE": 8e-4}),
    ("increase batch size to 384", {"BATCH_SIZE": 384}),
    ("increase batch size to 512", {"BATCH_SIZE": 512}),
    ("disable weight decay", {"WEIGHT_DECAY": 0.0}),
    ("raise weight decay to 5e-5", {"WEIGHT_DECAY": 5e-5}),
    ("reduce grad clip to 0.5", {"GRAD_CLIP_NORM": 0.5}),
    ("raise warmup fraction to 0.1", {"LR_WARMUP_FRACTION": 0.1}),
    ("raise final lr multiplier to 0.2", {"LR_FINAL_MULTIPLIER": 0.2}),
    ("lr 7e-4 with batch size 512", {"LEARNING_RATE": 7e-4, "BATCH_SIZE": 512}),
    ("lr 7e-4 with no weight decay", {"LEARNING_RATE": 7e-4, "WEIGHT_DECAY": 0.0}),
    ("batch size 512 with grad clip 0.5", {"BATCH_SIZE": 512, "GRAD_CLIP_NORM": 0.5}),
    ("lr 7e-4 with warmup 0.1", {"LEARNING_RATE": 7e-4, "LR_WARMUP_FRACTION": 0.1}),
    ("lr 8e-4 with batch size 384", {"LEARNING_RATE": 8e-4, "BATCH_SIZE": 384}),
    ("batch size 384 with no weight decay", {"BATCH_SIZE": 384, "WEIGHT_DECAY": 0.0}),
    ("grad clip 0.5 with no weight decay", {"GRAD_CLIP_NORM": 0.5, "WEIGHT_DECAY": 0.0}),
    ("warmup 0.1 and final lr multiplier 0.2", {"LR_WARMUP_FRACTION": 0.1, "LR_FINAL_MULTIPLIER": 0.2}),
]


def format_value(value: object) -> str:
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def apply_overrides(source: str, overrides: dict[str, object]) -> str:
    updated = source
    for name, value in overrides.items():
        pattern = re.compile(rf"(?m)^({re.escape(name)}\s*=\s*).*$")
        rendered = format_value(value)
        updated, count = pattern.subn(lambda match: f"{match.group(1)}{rendered}", updated, count=1)
        if count != 1:
            raise ValueError(f"Could not find top-level assignment for {name!r} in train.py")
    return updated


def best_source_for_campaign(campaign_dir: Path) -> str:
    best_path = campaign_dir / "best_train.py"
    if best_path.exists():
        return best_path.read_text()
    return TRAIN_PY.read_text()


def load_attempted_descriptions(campaign_dir: Path) -> set[str]:
    ledger_path = campaign_dir / "ledger.jsonl"
    if not ledger_path.exists():
        return set()
    attempted: set[str] = set()
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        attempted.add(json.loads(line)["description"])
    return attempted


def summarize_overrides(overrides: dict[str, object]) -> str:
    return ", ".join(f"{name}={value!r}" for name, value in overrides.items())


def run_variant(campaign_dir: Path, description: str, max_steps: int, time_budget_s: float | None) -> int:
    cmd = [
        sys.executable,
        str(RUN_ONE),
        "--campaign-dir",
        str(campaign_dir),
        "--description",
        description,
        "--max-steps",
        str(max_steps),
    ]
    if time_budget_s is not None:
        cmd += ["--time-budget-s", str(time_budget_s)]

    result = subprocess.run(cmd, cwd=ROOT, text=True)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of conservative direct-edit train.py changes")
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--time-budget-s", type=float, default=None)
    parser.add_argument("--time-limit-s", type=float, default=1800.0)
    parser.add_argument("--max-variants", type=int, default=None)
    args = parser.parse_args()

    campaign_dir = args.campaign_dir if args.campaign_dir.is_absolute() else ROOT / args.campaign_dir
    start = time.monotonic()
    attempted = 0
    seen_descriptions = load_attempted_descriptions(campaign_dir)

    try:
        for idx, (label, overrides) in enumerate(VARIANTS, start=1):
            if args.max_variants is not None and attempted >= args.max_variants:
                break
            if (time.monotonic() - start) >= args.time_limit_s:
                break

            desc = f"batch {idx:02d}: {label} [{summarize_overrides(overrides)}]"
            if desc in seen_descriptions:
                print(f"\n=== skipping already attempted variant: {desc} ===", flush=True)
                continue

            base_source = best_source_for_campaign(campaign_dir)
            candidate_source = apply_overrides(base_source, overrides)
            TRAIN_PY.write_text(candidate_source)

            print(f"\n=== {desc} ===", flush=True)
            rc = run_variant(campaign_dir, desc, args.max_steps, args.time_budget_s)
            attempted += 1
            seen_descriptions.add(desc)
            if rc != 0:
                raise SystemExit(rc)
    finally:
        best_path = campaign_dir / "best_train.py"
        if best_path.exists():
            TRAIN_PY.write_text(best_path.read_text())

    print(f"\nCompleted {attempted} batch variants.", flush=True)


if __name__ == "__main__":
    main()
