# Direct-Edit Workflow

This workflow treats `train.py` as the single mutable artifact while keeping the
physics and evaluation code fixed.

It is the closest component in this repository to the original
[`autoresearch`](https://github.com/karpathy/autoresearch) pattern that inspired
the project, adapted here for a quantum many-body setting.

## Core Rule

- `prepare.py` is fixed.
- `exact_diag.py` is fixed.
- `train.py` is the only file whose scientific recipe is meant to change during a run.

## One Iteration

Edit `train.py`, then evaluate it:

```bash
.venv/bin/python controller/run_direct_experiment.py \
  --campaign-dir results/fixed_panel_search \
  --description "brief description of the change" \
  --max-steps 50
```

Possible outcomes:

- `keep`: the current `train.py` becomes the new best snapshot
- `discard`: the change underperformed and `train.py` is restored automatically
- `crash`: evaluation failed and `train.py` is restored automatically

## Batch Mode

The repository also includes a conservative batch runner for small recipe sweeps:

```bash
.venv/bin/python controller/run_direct_batch.py \
  --campaign-dir results/fixed_panel_search \
  --max-steps 50 \
  --time-limit-s 1800
```

## Files Written

Each campaign directory contains:

- `results.tsv`
- `ledger.jsonl`
- `campaign_history.md`
- `best_train.py`
- `initial_train.py`
- `experiment_state.json`

## Fixed-Budget Rule

Each campaign keeps the evaluation budget fixed once created:

- `--max-steps`
- `--time-budget-s`
- `--panel-seed`

This avoids comparing recipe changes under different computational budgets.

## Design Choice

The workflow uses local snapshots instead of destructive git resets. That keeps
the keep/discard behavior reproducible while avoiding accidental loss of
unrelated work in the same repository.
