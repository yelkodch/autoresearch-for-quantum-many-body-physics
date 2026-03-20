"""Agentic loop: LLM modifies train.py to find the best NQS recipe.

Supports:
  --resume     Continue from the last completed iteration
  --max-wall-time  Stop after N hours and checkpoint
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "controller"))

from openai_compatible_client import chat_completion
from run_panel_eval import run_panel


# ── Constants ──
TRAIN_PY = ROOT / "train.py"
PROGRAM_MD = ROOT / "program.md"
PREPARE_PY = ROOT / "prepare.py"

REQUIRED_IMPORTS = ["from prepare import"]
REQUIRED_FUNCTIONS = ["parse_args", "RunSummary", "main"]
PROMPT_FROM_BEST_DEGRADATION_TOL = 0.05

# ── Guardrails ──
def passes_syntax_check(code: str) -> tuple[bool, str]:
    """Check that the code is valid Python."""
    tmp = ROOT / "_tmp_syntax_check.py"
    try:
        tmp.write_text(code)
        py_compile.compile(str(tmp), doraise=True)
        return True, ""
    except py_compile.PyCompileError as e:
        return False, str(e)
    finally:
        tmp.unlink(missing_ok=True)


def preserves_fixed_elements(code: str) -> tuple[bool, str]:
    """Check that required elements are present."""
    for req in REQUIRED_IMPORTS:
        if req not in code:
            return False, f"Missing required import: {req}"
    for req in REQUIRED_FUNCTIONS:
        if req not in code:
            return False, f"Missing required function/class: {req}"
    return True, ""


def passes_smoke_test(code: str, timeout: int = 30) -> tuple[bool, str]:
    """Run train.py with minimal config to check it doesn't crash."""
    tmp = ROOT / "_tmp_smoke_train.py"
    try:
        tmp.write_text(code)
        env = {**os.environ, "TV_DEVICE": "cpu"}
        result = subprocess.run(
            [sys.executable, str(tmp), "--j2", "0.0", "--sector", "sz0",
             "--seed", "999", "--max-steps", "3", "--output-dir", str(ROOT / "_tmp_smoke_out")],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        if result.returncode != 0:
            return False, f"Exit code {result.returncode}: {result.stderr[:300]}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Smoke test timed out"
    except Exception as e:
        return False, str(e)
    finally:
        tmp.unlink(missing_ok=True)
        shutil.rmtree(ROOT / "_tmp_smoke_out", ignore_errors=True)


def extract_code_from_response(response: str) -> str | None:
    """Extract the first ```python ... ``` block from the agent's response."""
    pattern = r'```python\s*\n(.*?)```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try without language tag
    pattern = r'```\s*\n(.*?)```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_search_replace_blocks(response: str) -> list[tuple[str, str]]:
    """Extract Aider-style SEARCH/REPLACE blocks from the response."""
    pattern = re.compile(
        r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
        re.DOTALL,
    )
    return [(match.group(1), match.group(2)) for match in pattern.finditer(response)]


def apply_search_replace_blocks(code: str, blocks: list[tuple[str, str]]) -> tuple[str | None, str]:
    """Apply SEARCH/REPLACE edits to code, rejecting missing or ambiguous matches."""
    if not blocks:
        return None, "No SEARCH/REPLACE blocks found"

    updated = code
    for index, (search_text, replace_text) in enumerate(blocks, start=1):
        if not search_text:
            return None, f"Edit block {index} has empty SEARCH section"
        matches = updated.count(search_text)
        if matches == 0:
            return None, f"SEARCH block {index} not found in current train.py"
        if matches > 1:
            return None, f"SEARCH block {index} is ambiguous ({matches} matches)"
        updated = updated.replace(search_text, replace_text, 1)
    return updated, ""


def build_candidate_code(current_code: str, response: str) -> tuple[str | None, str]:
    """Turn an agent response into concrete code, preferring direct edit blocks."""
    blocks = extract_search_replace_blocks(response)
    if blocks:
        patched, err = apply_search_replace_blocks(current_code, blocks)
        if patched is not None:
            return patched, ""

    extracted = extract_code_from_response(response)
    if extracted is not None:
        return extracted, ""

    return None, "No valid SEARCH/REPLACE blocks or complete python file found"


# ── History ──
def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparseable>"


def _top_level_structure(code: str) -> tuple[dict[str, str], dict[str, str]]:
    tree = ast.parse(code)
    lines = code.splitlines()
    constants: dict[str, str] = {}
    definitions: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.isupper():
                constants[name] = _safe_unparse(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name.isupper() and node.value is not None:
                constants[name] = _safe_unparse(node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = max(0, node.lineno - 1)
            end = getattr(node, "end_lineno", node.lineno)
            definitions[node.name] = "\n".join(lines[start:end]).strip()
    return constants, definitions


def summarize_code_changes(old_code: str, new_code: str) -> tuple[str, dict[str, object]]:
    """Build a semantic-ish summary of the top-level changes between two code versions."""
    old_constants, old_defs = _top_level_structure(old_code)
    new_constants, new_defs = _top_level_structure(new_code)

    changed_constants = []
    for name in sorted(set(old_constants) | set(new_constants)):
        if old_constants.get(name) != new_constants.get(name):
            changed_constants.append({
                "name": name,
                "old": old_constants.get(name),
                "new": new_constants.get(name),
            })

    added_defs = sorted(name for name in new_defs if name not in old_defs)
    removed_defs = sorted(name for name in old_defs if name not in new_defs)
    modified_defs = sorted(
        name for name in new_defs
        if name in old_defs and new_defs[name] != old_defs[name]
    )

    details: list[str] = []
    if changed_constants:
        preview = ", ".join(
            f"{item['name']}={item['new']}" for item in changed_constants[:4]
        )
        if len(changed_constants) > 4:
            preview += f", +{len(changed_constants) - 4} more"
        details.append(f"constants: {preview}")
    if modified_defs:
        preview = ", ".join(modified_defs[:4])
        if len(modified_defs) > 4:
            preview += f", +{len(modified_defs) - 4} more"
        details.append(f"modified defs: {preview}")
    if added_defs:
        details.append(f"added defs: {', '.join(added_defs[:4])}")
    if removed_defs:
        details.append(f"removed defs: {', '.join(removed_defs[:4])}")

    old_lines = set(old_code.strip().split("\n"))
    new_lines = set(new_code.strip().split("\n"))
    diff_stats = f"+{len(new_lines - old_lines)} -{len(old_lines - new_lines)} lines"
    summary = "; ".join(details) if details else "no top-level semantic summary"
    return (
        f"{summary} ({diff_stats})",
        {
            "changed_constants": changed_constants,
            "modified_definitions": modified_defs,
            "added_definitions": added_defs,
            "removed_definitions": removed_defs,
            "diff_stats": diff_stats,
        },
    )


def summarize_panel_results(panel: dict) -> tuple[str, list[dict[str, object]]]:
    results = []
    parts: list[str] = []
    for item in panel.get("results", []):
        compact = {
            "j2": item.get("j2"),
            "energy_per_site": item.get("energy_per_site"),
            "eval_energy_std_per_site": item.get("eval_energy_std_per_site"),
            "steps_completed": item.get("steps_completed"),
            "wall_time_s": item.get("wall_time_s"),
            "error": item.get("error", False),
        }
        results.append(compact)
        if compact["error"]:
            parts.append(f"J2={compact['j2']:.3f}: ERROR")
        else:
            parts.append(
                f"J2={compact['j2']:.3f}: E/N={compact['energy_per_site']:.4f}, "
                f"std/N={compact['eval_energy_std_per_site']:.4f}, "
                f"steps={compact['steps_completed']}"
            )
    return " | ".join(parts), results


def compact_history(ledger: list[dict]) -> str:
    """Build a full history string for the agent prompt."""
    lines = []
    for entry in ledger:
        panel_metrics = entry.get("panel_metrics_summary", "n/a")
        lines.append(
            f"Iteration {entry['iteration']:03d}: "
            f"score={entry['panel_score']:.4f}, "
            f"accepted={'✓' if entry.get('accepted') else '✗'}, "
            f"panel=[{panel_metrics}], "
            f"changes={entry.get('change_summary', 'n/a')}"
        )
        if entry.get("rejection_reason"):
            lines.append(f"  rejection_reason={entry['rejection_reason']}")
    return "\n".join(lines)


def append_campaign_history(campaign_dir: Path, entry: dict) -> None:
    history_path = campaign_dir / "campaign_history.md"
    lines = [
        f"## Iteration {entry['iteration']:03d}",
        f"- score: {entry['panel_score']:.4f}",
        f"- accepted: {'yes' if entry.get('accepted') else 'no'}",
        f"- panel: {entry.get('panel_metrics_summary', 'n/a')}",
        f"- changes: {entry.get('change_summary', 'n/a')}",
    ]
    if entry.get("rejection_reason"):
        lines.append(f"- rejection_reason: {entry['rejection_reason']}")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a") as f:
        f.write("\n".join(lines) + "\n\n")


def should_branch_from_best(current_score: float, best_score: float, tolerance: float = PROMPT_FROM_BEST_DEGRADATION_TOL) -> bool:
    """Return True when the current candidate is clearly worse than the best known one."""
    if best_score == 0.0:
        return False
    return current_score > (best_score + tolerance)


# ── Campaign state ──
def load_campaign_state(campaign_dir: Path) -> dict:
    """Load or initialize campaign state."""
    state_path = campaign_dir / "campaign_state.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {
        "completed_iterations": 0,
        "total_iterations": 0,
        "best_score": 0.0,
        "best_iteration": -1,
        "status": "new",
        "start_time": datetime.now(timezone.utc).isoformat(),
    }


def save_campaign_state(campaign_dir: Path, state: dict) -> None:
    (campaign_dir / "campaign_state.json").write_text(
        json.dumps(state, indent=2)
    )


def load_ledger(campaign_dir: Path) -> list[dict]:
    ledger_path = campaign_dir / "ledger.jsonl"
    if not ledger_path.exists():
        return []
    entries = []
    for line in ledger_path.read_text().strip().split('\n'):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def append_ledger(campaign_dir: Path, entry: dict) -> None:
    ledger_path = campaign_dir / "ledger.jsonl"
    with open(ledger_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main loop ──
def run_agentic_loop(
    campaign_dir: Path,
    n_iterations: int = 20,
    max_steps: int = 1000,
    time_budget_s: float | None = None,
    max_wall_time_s: float | None = None,
    resume: bool = False,
) -> None:
    campaign_dir.mkdir(parents=True, exist_ok=True)

    # Load program.md
    program_text = PROGRAM_MD.read_text() if PROGRAM_MD.exists() else "No program.md found."

    # Load or init state
    state = load_campaign_state(campaign_dir)
    ledger = load_ledger(campaign_dir)

    if resume and state["status"] == "paused":
        start_iter = state["completed_iterations"] + 1
        best_score = state["best_score"]
        best_code = (campaign_dir / "best_train.py").read_text()
        print(f"Resuming from iteration {start_iter} (best score: {best_score:.4f})")
    else:
        start_iter = 1
        best_score = 0.0
        best_code = TRAIN_PY.read_text()
        # Save initial train.py
        (campaign_dir / "initial_train.py").write_text(best_code)
        state["total_iterations"] = n_iterations
        state["status"] = "running"

    current_code = TRAIN_PY.read_text()
    prepare_checksum = hashlib.md5(PREPARE_PY.read_bytes()).hexdigest()

    campaign_start = time.time()

    for iteration in range(start_iter, n_iterations + 1):
        # Check wall time
        if max_wall_time_s and (time.time() - campaign_start) >= max_wall_time_s:
            print(f"\n⏱ Max wall time reached. Pausing at iteration {iteration - 1}.")
            state["status"] = "paused"
            save_campaign_state(campaign_dir, state)
            break

        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{n_iterations}")
        print(f"{'='*60}")

        # 1. Run panel eval
        panel_seed = 10000 + iteration * 137
        panel = run_panel(campaign_dir, iteration, panel_seed, max_steps, time_budget_s)
        score = panel["panel_score"]
        panel_metrics_summary, compact_panel_results = summarize_panel_results(panel)

        print(f"\n  Panel score: {score:.4f}")

        # Track best
        is_best = score < best_score or best_score == 0.0
        if is_best:
            best_score = score
            best_code = current_code
            state["best_score"] = best_score
            state["best_iteration"] = iteration
            (campaign_dir / "best_train.py").write_text(best_code)
            print(f"  ★ NEW BEST: {best_score:.4f}")

        # 2. Choose prompt base
        prompt_base = "current"
        if should_branch_from_best(score, best_score) and best_code.strip() != current_code.strip():
            prompt_base = "best"
            current_code = best_code
            TRAIN_PY.write_text(best_code)
            print(
                f"  ↺ Current score is worse than best by more than "
                f"{PROMPT_FROM_BEST_DEGRADATION_TOL:.3f}; branching from best_train.py"
            )

        # 3. Build agent prompt
        history_str = compact_history(ledger)
        best_code_block = ""
        if best_code.strip() != current_code.strip():
            best_code_block = f"""

## Best train.py so far
```python
{best_code}
```
"""
        prompt = f"""
{program_text}

## Current train.py
```python
{current_code}
```
{best_code_block}

## Results
- Current score (this iteration): {score:.4f}
- Best score so far: {best_score:.4f}
- Iteration: {iteration}/{n_iterations}
- Current panel details: {panel_metrics_summary or 'n/a'}
- Prompt base: {prompt_base}

## History
{history_str if history_str else 'No history yet (first iteration).'}

## Your task
Analyze the current results and modify train.py to achieve a lower (more negative) energy score.
Think about what changes would most improve convergence and accuracy.
Edit the file directly by returning one or more SEARCH/REPLACE blocks in this exact format:
<<<<<<< SEARCH
<exact text copied from the current train.py>
=======
<replacement text>
>>>>>>> REPLACE

Rules:
- Return ONLY SEARCH/REPLACE blocks, with no explanation.
- Use exact text in SEARCH blocks so the controller can patch the real file.
- Use as few blocks as possible.
- If a very large rewrite is truly necessary, you may instead return the full train.py in one ```python ... ``` block as a fallback.
"""

        # 4. Get agent response
        print(f"\n  Calling LLM for next modification...")
        try:
            response = chat_completion(prompt, temperature=0.2)
        except Exception as e:
            print(f"  [ERROR] LLM call failed: {e}")
            ledger_entry = {
                "iteration": iteration, "panel_score": score,
                "accepted": False, "change_summary": f"LLM error: {e}",
                "panel_metrics_summary": panel_metrics_summary,
                "panel_results": compact_panel_results,
                "rejection_reason": f"LLM error: {e}",
                "best_score": best_score,
            }
            append_ledger(campaign_dir, ledger_entry)
            append_campaign_history(campaign_dir, ledger_entry)
            ledger.append(ledger_entry)
            state["completed_iterations"] = iteration
            save_campaign_state(campaign_dir, state)
            continue

        # 5. Extract code
        new_code, parse_err = build_candidate_code(current_code, response)
        if new_code is None:
            print(f"  [REJECT] Could not build code from agent response: {parse_err}")
            # Save the raw response for debugging
            iter_dir = campaign_dir / f"iteration_{iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            (iter_dir / "agent_response.txt").write_text(response)
            ledger_entry = {
                "iteration": iteration, "panel_score": score,
                "accepted": False, "change_summary": "No applicable edit produced",
                "panel_metrics_summary": panel_metrics_summary,
                "panel_results": compact_panel_results,
                "rejection_reason": f"No applicable edit: {parse_err}",
                "best_score": best_score,
            }
            append_ledger(campaign_dir, ledger_entry)
            append_campaign_history(campaign_dir, ledger_entry)
            ledger.append(ledger_entry)
            state["completed_iterations"] = iteration
            save_campaign_state(campaign_dir, state)
            continue

        # 6. Guardrails
        diff_summary, diff_details = summarize_code_changes(current_code, new_code)

        ok, err = passes_syntax_check(new_code)
        if not ok:
            print(f"  [REJECT] Syntax error: {err[:200]}")
            ledger_entry = {
                "iteration": iteration, "panel_score": score,
                "accepted": False, "change_summary": diff_summary,
                "change_details": diff_details,
                "panel_metrics_summary": panel_metrics_summary,
                "panel_results": compact_panel_results,
                "rejection_reason": f"Syntax error: {err[:100]}",
                "best_score": best_score,
            }
            append_ledger(campaign_dir, ledger_entry)
            append_campaign_history(campaign_dir, ledger_entry)
            ledger.append(ledger_entry)
            state["completed_iterations"] = iteration
            save_campaign_state(campaign_dir, state)
            continue

        ok, err = preserves_fixed_elements(new_code)
        if not ok:
            print(f"  [REJECT] Missing required: {err}")
            ledger_entry = {
                "iteration": iteration, "panel_score": score,
                "accepted": False, "change_summary": diff_summary,
                "change_details": diff_details,
                "panel_metrics_summary": panel_metrics_summary,
                "panel_results": compact_panel_results,
                "rejection_reason": f"Missing element: {err[:100]}",
                "best_score": best_score,
            }
            append_ledger(campaign_dir, ledger_entry)
            append_campaign_history(campaign_dir, ledger_entry)
            ledger.append(ledger_entry)
            state["completed_iterations"] = iteration
            save_campaign_state(campaign_dir, state)
            continue

        ok, err = passes_smoke_test(new_code)
        if not ok:
            print(f"  [REJECT] Smoke test failed: {err[:200]}")
            ledger_entry = {
                "iteration": iteration, "panel_score": score,
                "accepted": False, "change_summary": diff_summary,
                "change_details": diff_details,
                "panel_metrics_summary": panel_metrics_summary,
                "panel_results": compact_panel_results,
                "rejection_reason": f"Smoke fail: {err[:100]}",
                "best_score": best_score,
            }
            append_ledger(campaign_dir, ledger_entry)
            append_campaign_history(campaign_dir, ledger_entry)
            ledger.append(ledger_entry)
            state["completed_iterations"] = iteration
            save_campaign_state(campaign_dir, state)
            continue

        # Check prepare.py wasn't modified
        if hashlib.md5(PREPARE_PY.read_bytes()).hexdigest() != prepare_checksum:
            print("  [ABORT] prepare.py was modified! Restoring.")
            # This shouldn't happen normally, but just in case
            state["completed_iterations"] = iteration
            save_campaign_state(campaign_dir, state)
            break

        # 7. Accept the code
        print(f"  [ACCEPTED] {diff_summary}")

        # Save iteration files
        iter_dir = campaign_dir / f"iteration_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "train.py.bak").write_text(new_code)
        (iter_dir / "agent_response.txt").write_text(response)

        # Apply
        current_code = new_code
        TRAIN_PY.write_text(new_code)

        ledger_entry = {
            "iteration": iteration, "panel_score": score,
            "accepted": True, "change_summary": diff_summary,
            "change_details": diff_details,
            "panel_metrics_summary": panel_metrics_summary,
            "panel_results": compact_panel_results,
            "prompt_base": prompt_base,
            "best_score": best_score,
        }
        append_ledger(campaign_dir, ledger_entry)
        append_campaign_history(campaign_dir, ledger_entry)
        ledger.append(ledger_entry)
        state["completed_iterations"] = iteration
        save_campaign_state(campaign_dir, state)

    # Campaign finished
    TRAIN_PY.write_text(best_code)
    if state["status"] != "paused":
        state["status"] = "completed"
    state["end_time"] = datetime.now(timezone.utc).isoformat()
    state["total_wall_time_s"] = time.time() - campaign_start
    save_campaign_state(campaign_dir, state)

    print(f"\n{'='*60}")
    print(f"CAMPAIGN {'PAUSED' if state['status']=='paused' else 'COMPLETE'}")
    print(f"Best score: {best_score:.4f} (iteration {state.get('best_iteration', '?')})")
    print(f"Total wall time: {state.get('total_wall_time_s', 0)/3600:.1f} hours")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="J1-J2 recipe-search agent loop")
    parser.add_argument("--campaign-dir", type=Path, default=None,
                        help="Campaign directory (default: auto-generated)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of agent iterations")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Training steps per delta")
    parser.add_argument("--time-budget-s", type=float, default=None,
                        help="Optional wall-clock time cap per delta (seconds)")
    parser.add_argument("--max-wall-time", type=str, default=None,
                        help="Max wall time for campaign, e.g. '4h' or '30m'")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last paused state")
    args = parser.parse_args()

    # Parse max wall time
    max_wall_time_s = None
    if args.max_wall_time:
        val = args.max_wall_time.strip().lower()
        if val.endswith("h"):
            max_wall_time_s = float(val[:-1]) * 3600
        elif val.endswith("m"):
            max_wall_time_s = float(val[:-1]) * 60
        else:
            max_wall_time_s = float(val)

    # Campaign dir
    if args.campaign_dir:
        campaign_dir = args.campaign_dir
    elif args.resume:
        # Find most recent campaign
        from prepare import RESULTS_DIR
        campaigns = sorted(RESULTS_DIR.glob("j1j2_*"))
        if not campaigns:
            print("No campaign found to resume.")
            sys.exit(1)
        campaign_dir = campaigns[-1]
    else:
        from prepare import RESULTS_DIR
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        campaign_dir = RESULTS_DIR / f"j1j2_{ts}"

    run_agentic_loop(
        campaign_dir=campaign_dir,
        n_iterations=args.iterations,
        max_steps=args.max_steps,
        time_budget_s=args.time_budget_s,
        max_wall_time_s=max_wall_time_s,
        resume=args.resume,
    )
