"""Direct-edit keep/discard experiment loop for J1-J2.

This script is designed for an agent that edits train.py directly in the workspace.
Each invocation evaluates the *current* train.py, logs the result, and either keeps
the change or restores the previous best snapshot.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controller.run_panel_eval import run_panel
from prepare import RESULTS_DIR, ensure_project_dirs

TRAIN_PY = ROOT / "train.py"
RESULTS_HEADER = "snapshot\tpanel_score\tstatus\tdescription\n"
KEY_KNOBS = (
    "D_MODEL",
    "N_HEADS",
    "N_LAYERS",
    "FF_HIDDEN_DIM",
    "DROPOUT",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "BETAS",
    "GRAD_CLIP_NORM",
    "LR_WARMUP_FRACTION",
    "LR_FINAL_MULTIPLIER",
    "EVAL_SAMPLES",
    "EVAL_REPEATS",
    "SIGN_STRUCTURE_MODE",
    "BASELINE_TYPE",
    "ADVANTAGE_TYPE",
)


@dataclass
class ExperimentState:
    campaign_dir: str
    created_at: str
    completed_iterations: int
    best_score: float | None
    best_iteration: int | None
    best_snapshot: str | None
    fixed_max_steps: int | None
    fixed_time_budget_s: float | None
    fixed_panel_seed: int | None = None


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def snapshot_id_for_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:7]


def load_state(state_path: Path) -> ExperimentState | None:
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text())
    return ExperimentState(**payload)


def save_state(state_path: Path, state: ExperimentState) -> None:
    state_path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True))


def history_path_for(campaign_dir: Path) -> Path:
    search_path = campaign_dir / "search_history.md"
    legacy_path = campaign_dir / "campaign_history.md"
    if search_path.exists() or not legacy_path.exists():
        return search_path
    return legacy_path


def best_snapshot_marker_path_for(campaign_dir: Path) -> Path:
    search_path = campaign_dir / "current_best_snapshot.txt"
    legacy_path = campaign_dir / "CURRENT_BEST_SNAPSHOT.txt"
    if search_path.exists() or not legacy_path.exists():
        return search_path
    return legacy_path


def ensure_campaign(
    campaign_dir: Path,
    max_steps: int | None,
    time_budget_s: float | None,
    panel_seed: int | None,
) -> ExperimentState:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    state_path = campaign_dir / "experiment_state.json"
    state = load_state(state_path)
    if state is None:
        baseline_text = TRAIN_PY.read_text()
        baseline_snapshot = snapshot_id_for_text(baseline_text)
        (campaign_dir / "results.tsv").write_text(RESULTS_HEADER)
        (campaign_dir / "initial_train.py").write_text(baseline_text)
        (campaign_dir / "best_train.py").write_text(baseline_text)
        best_snapshot_marker_path_for(campaign_dir).write_text(baseline_snapshot + "\n")
        try:
            campaign_dir_str = str(campaign_dir.relative_to(ROOT))
        except ValueError:
            campaign_dir_str = str(campaign_dir)
        state = ExperimentState(
            campaign_dir=campaign_dir_str,
            created_at=iso_now(),
            completed_iterations=0,
            best_score=None,
            best_iteration=None,
            best_snapshot=baseline_snapshot,
            fixed_max_steps=max_steps,
            fixed_time_budget_s=time_budget_s,
            fixed_panel_seed=panel_seed if panel_seed is not None else 10000,
        )
        save_state(state_path, state)
        return state

    if state.fixed_max_steps != max_steps or state.fixed_time_budget_s != time_budget_s:
        raise SystemExit(
            "Fixed budget mismatch for existing direct-edit campaign. "
            f"Expected max_steps={state.fixed_max_steps}, time_budget_s={state.fixed_time_budget_s}; "
            f"got max_steps={max_steps}, time_budget_s={time_budget_s}."
        )
    if state.fixed_panel_seed is None:
        state.fixed_panel_seed = panel_seed if panel_seed is not None else 10000
        save_state(state_path, state)
    elif panel_seed is not None and state.fixed_panel_seed != panel_seed:
        raise SystemExit(
            "Fixed panel seed mismatch for existing direct-edit campaign. "
            f"Expected panel_seed={state.fixed_panel_seed}; got panel_seed={panel_seed}."
        )
    return state


def append_results_tsv(campaign_dir: Path, snapshot: str, panel_score: float, status: str, description: str) -> None:
    with (campaign_dir / "results.tsv").open("a") as handle:
        handle.write(f"{snapshot}\t{panel_score:.6f}\t{status}\t{description}\n")


def append_history(campaign_dir: Path, entry: dict[str, object]) -> None:
    history_path = history_path_for(campaign_dir)
    lines = [
        f"## Iteration {int(entry['iteration']):03d}",
        f"- snapshot: {entry['snapshot']}",
        f"- score: {float(entry['panel_score']):.4f}",
        f"- status: {entry['status']}",
        f"- description: {entry['description']}",
        f"- panel: {entry['panel_metrics_summary']}",
    ]
    if entry.get("delta_vs_best_before") is not None:
        lines.append(f"- delta_vs_best_before: {float(entry['delta_vs_best_before']):+.4f}")
    if entry.get("changed_constants_summary"):
        lines.append(f"- changes: {entry['changed_constants_summary']}")
    if entry.get("restored_best_snapshot"):
        lines.append(f"- restored_best_snapshot: {entry['restored_best_snapshot']}")
    with history_path.open("a") as handle:
        handle.write("\n".join(lines) + "\n\n")


def summarize_panel(panel: dict[str, object]) -> str:
    parts: list[str] = []
    for item in panel.get("results", []):
        if item.get("error"):
            parts.append(f"J2={item['j2']:.3f}: ERROR")
        else:
            parts.append(
                f"J2={item['j2']:.3f}: E/N={item['energy_per_site']:.4f}, "
                f"std/N={item['eval_energy_std_per_site']:.4f}, steps={item['steps_completed']}"
            )
    return " | ".join(parts)


def load_ledger_entries(campaign_dir: Path) -> list[dict[str, object]]:
    ledger_path = campaign_dir / "ledger.jsonl"
    if not ledger_path.exists():
        return []
    entries: list[dict[str, object]] = []
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


def extract_literal_constants(source: str) -> dict[str, object]:
    module = ast.parse(source)
    constants: dict[str, object] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.isupper():
            continue
        try:
            constants[target.id] = ast.literal_eval(node.value)
        except Exception:
            continue
    return constants


def diff_constants(reference_source: str, candidate_source: str) -> dict[str, dict[str, object]]:
    reference = extract_literal_constants(reference_source)
    candidate = extract_literal_constants(candidate_source)
    changes: dict[str, dict[str, object]] = {}
    for key in sorted(set(reference) | set(candidate)):
        before = reference.get(key)
        after = candidate.get(key)
        if before != after:
            changes[key] = {"from": before, "to": after}
    return changes


def summarize_changed_constants(changes: dict[str, dict[str, object]], limit: int = 8) -> str:
    if not changes:
        return "none"
    items = list(changes.items())
    parts = [f"{name}: {payload['from']!r} -> {payload['to']!r}" for name, payload in items[:limit]]
    extra = len(items) - limit
    if extra > 0:
        parts.append(f"+{extra} more")
    return "; ".join(parts)


def write_search_memory(campaign_dir: Path, state: ExperimentState) -> None:
    entries = load_ledger_entries(campaign_dir)
    best_train_path = campaign_dir / "best_train.py"
    best_constants = extract_literal_constants(best_train_path.read_text()) if best_train_path.exists() else {}

    best_improvements: list[dict[str, object]] = []
    running_best: float | None = None
    for entry in entries:
        score = float(entry["panel_score"])
        if running_best is None or score < running_best:
            best_improvements.append(entry)
            running_best = score

    recent_mistakes = [
        entry
        for entry in reversed(entries)
        if entry.get("status") in {"discard", "crash"}
    ][:10]

    lines = [
        "# Search Memory",
        "",
        "Read this file before editing train.py. It records the ideas already tried,",
        "their scores on the fixed panel, and the current best recipe.",
        "",
        "## Current Best",
        f"- best_iteration: {state.best_iteration}",
        f"- best_snapshot: {state.best_snapshot}",
        f"- best_score: {state.best_score}",
        f"- fixed_panel_seed: {state.fixed_panel_seed}",
        f"- completed_iterations: {state.completed_iterations}",
        "",
        "## Best Key Knobs",
    ]

    knob_lines = 0
    for name in KEY_KNOBS:
        if name in best_constants:
            lines.append(f"- `{name} = {best_constants[name]!r}`")
            knob_lines += 1
    if knob_lines == 0:
        lines.append("- No literal key knobs extracted from best_train.py")

    lines += ["", "## Improvements So Far"]
    if best_improvements:
        for entry in best_improvements:
            delta = entry.get("delta_vs_best_before")
            delta_text = ""
            if delta is not None:
                delta_text = f", delta_vs_best_before={float(delta):+.4f}"
            lines.append(
                f"- Iteration {int(entry['iteration']):03d}: {entry['description']} "
                f"-> score {float(entry['panel_score']):.4f}{delta_text}. "
                f"Changes: {entry.get('changed_constants_summary', 'none')}"
            )
    else:
        lines.append("- No successful iterations recorded yet.")

    lines += ["", "## Mistakes And Regressions"]
    if recent_mistakes:
        for entry in recent_mistakes:
            delta = entry.get("delta_vs_best_before")
            delta_text = f"{float(delta):+.4f}" if delta is not None else "n/a"
            lines.append(
                f"- Iteration {int(entry['iteration']):03d}: {entry['description']} "
                f"-> status={entry['status']}, score={float(entry['panel_score']):.4f}, "
                f"delta_vs_best_before={delta_text}. Changes: {entry.get('changed_constants_summary', 'none')}"
            )
    else:
        lines.append("- No discarded or crashing iterations yet.")

    lines += ["", "## Recent Attempt Log"]
    for entry in entries[-12:]:
        delta = entry.get("delta_vs_best_before")
        delta_text = f"{float(delta):+.4f}" if delta is not None else "n/a"
        lines.append(
            f"- Iteration {int(entry['iteration']):03d}: status={entry['status']}, "
            f"score={float(entry['panel_score']):.4f}, delta_vs_best_before={delta_text}, "
            f"description={entry['description']}"
        )

    lines += [
        "",
        "## Editing Guidance",
        "- Start from best_train.py, not from a discarded branch.",
        "- Change only a few knobs at a time so the effect is attributable.",
        "- Prefer ideas not already listed in the recent attempt log unless you have a concrete reason to retry them.",
    ]

    (campaign_dir / "search_memory.md").write_text("\n".join(lines) + "\n")
    (campaign_dir / "search_memory.json").write_text(
        json.dumps(
            {
                "best_iteration": state.best_iteration,
                "best_snapshot": state.best_snapshot,
                "best_score": state.best_score,
                "fixed_panel_seed": state.fixed_panel_seed,
                "completed_iterations": state.completed_iterations,
                "best_key_knobs": {name: best_constants.get(name) for name in KEY_KNOBS if name in best_constants},
                "improvements": best_improvements,
                "recent_mistakes": recent_mistakes,
                "recent_attempts": entries[-12:],
            },
            indent=2,
            sort_keys=True,
        )
    )


def record_iteration(
    campaign_dir: Path,
    iteration: int,
    snapshot: str,
    description: str,
    panel: dict[str, object],
    status: str,
    restored_best_snapshot: str | None,
    delta_vs_best_before: float | None,
    changed_constants: dict[str, dict[str, object]],
) -> None:
    panel_score = float(panel["panel_score"])
    entry = {
        "iteration": iteration,
        "timestamp": iso_now(),
        "snapshot": snapshot,
        "description": description,
        "panel_score": panel_score,
        "panel_summary": panel,
        "panel_metrics_summary": summarize_panel(panel),
        "status": status,
        "restored_best_snapshot": restored_best_snapshot,
        "delta_vs_best_before": delta_vs_best_before,
        "changed_constants": changed_constants,
        "changed_constants_summary": summarize_changed_constants(changed_constants),
    }
    with (campaign_dir / "ledger.jsonl").open("a") as handle:
        handle.write(json.dumps(entry) + "\n")
    append_history(campaign_dir, entry)
    append_results_tsv(campaign_dir, snapshot, panel_score, status, description)


def restore_best_snapshot(campaign_dir: Path) -> str:
    best_train_path = campaign_dir / "best_train.py"
    best_text = best_train_path.read_text()
    best_snapshot = snapshot_id_for_text(best_text)
    TRAIN_PY.write_text(best_text)
    best_snapshot_marker_path_for(campaign_dir).write_text(best_snapshot + "\n")
    return best_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct-edit keep/discard experiment runner for J1-J2")
    parser.add_argument("--campaign-dir", type=Path, default=None)
    parser.add_argument("--description", type=str, default="experiment")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--time-budget-s", type=float, default=None)
    parser.add_argument("--panel-seed", type=int, default=None)
    args = parser.parse_args()

    ensure_project_dirs()
    if args.campaign_dir is None:
        campaign_dir = RESULTS_DIR / f"recipe_search_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    elif args.campaign_dir.is_absolute():
        campaign_dir = args.campaign_dir
    else:
        campaign_dir = ROOT / args.campaign_dir
    state = ensure_campaign(campaign_dir, args.max_steps, args.time_budget_s, args.panel_seed)
    state_path = campaign_dir / "experiment_state.json"

    iteration = state.completed_iterations + 1
    current_text = TRAIN_PY.read_text()
    current_snapshot = snapshot_id_for_text(current_text)
    panel_seed = state.fixed_panel_seed if state.fixed_panel_seed is not None else 10000

    iter_dir = campaign_dir / f"iteration_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "train_before_eval.py").write_text(current_text)

    best_before_text = (campaign_dir / "best_train.py").read_text()
    changed_constants = diff_constants(best_before_text, current_text)

    panel = run_panel(
        campaign_dir=campaign_dir,
        iteration=iteration,
        panel_seed=panel_seed,
        max_steps=args.max_steps,
        time_budget_s=args.time_budget_s,
    )

    panel_score = float(panel["panel_score"])
    delta_vs_best_before = None if state.best_score is None else panel_score - state.best_score
    valid_runs = int(panel["n_valid"])
    restored_best = None

    if valid_runs == 0:
        status = "crash"
        restored_best = restore_best_snapshot(campaign_dir)
    elif state.best_score is None or panel_score < state.best_score:
        status = "keep"
        state.best_score = panel_score
        state.best_iteration = iteration
        state.best_snapshot = current_snapshot
        (campaign_dir / "best_train.py").write_text(current_text)
        best_snapshot_marker_path_for(campaign_dir).write_text(current_snapshot + "\n")
    else:
        status = "discard"
        restored_best = restore_best_snapshot(campaign_dir)

    record_iteration(
        campaign_dir=campaign_dir,
        iteration=iteration,
        snapshot=current_snapshot,
        description=args.description,
        panel=panel,
        status=status,
        restored_best_snapshot=restored_best,
        delta_vs_best_before=delta_vs_best_before,
        changed_constants=changed_constants,
    )

    state.completed_iterations = iteration
    save_state(state_path, state)
    write_search_memory(campaign_dir, state)

    try:
        search_dir_display = str(campaign_dir.relative_to(ROOT))
    except ValueError:
        search_dir_display = str(campaign_dir)

    print(
        json.dumps(
            {
                "search_dir": search_dir_display,
                "iteration": iteration,
                "snapshot": current_snapshot,
                "panel_score": panel_score,
                "status": status,
                "best_score": state.best_score,
                "best_iteration": state.best_iteration,
                "restored_best_snapshot": restored_best,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
