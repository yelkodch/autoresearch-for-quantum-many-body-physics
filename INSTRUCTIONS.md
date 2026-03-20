# Developer Notes

This file is a compact runbook for contributors working on the J1-J2 2D
recipe-search project, which currently serves as the first worked example of a
broader autoresearch-for-quantum-many-body-physics effort.

## Project Roles

- `prepare.py`: fixed lattice geometry, bonds, constants, CLI helpers
- `exact_diag.py`: exact 4x4 reference energies and sector gaps
- `train.py`: mutable model and training recipe
- `controller/run_panel_eval.py`: short evaluation panel
- `controller/run_direct_experiment.py`: keep/discard runner for direct edits
- `controller/post_training_eval.py`: post-training comparison against exact references

## Basic Checks

```bash
source .venv/bin/activate

python prepare.py
python exact_diag.py
python -m unittest tests.test_regressions

TV_DEVICE=cpu python train.py --j2 0.0 --sector sz0 --max-steps 5
```

## Direct-Edit Search

Run one direct-edit experiment:

```bash
.venv/bin/python controller/run_direct_experiment.py \
  --campaign-dir results/fixed_panel_search \
  --description "test a small recipe change" \
  --max-steps 50
```

Run a conservative batch of predefined edits:

```bash
.venv/bin/python controller/run_direct_batch.py \
  --campaign-dir results/fixed_panel_search \
  --max-steps 50 \
  --time-limit-s 1800
```

## API-Driven Search

```bash
cp .env.example .env
# add GEMINI_API_KEY and/or GROQ_API_KEY

./run_campaign.sh
./run_campaign.sh --resume
```

## Post-Training Evaluation

```bash
.venv/bin/python controller/post_training_eval.py \
  --baseline-path results/fixed_panel_search/initial_train.py \
  --candidate-path results/fixed_panel_search/best_train.py \
  --output-dir results/post_training_eval_example \
  --j2-values 0.0 0.5 1.0 \
  --seeds 11 22 33 \
  --steps 50 200
```

For an interactive view of exact references versus model output, open:

```bash
jupyter lab notebooks/j1j2_results_dashboard.ipynb
```

## What To Watch

- exact-energy error against ED on 4x4,
- local-energy variance,
- `sz1 - sz0` gap error,
- order-parameter anchors such as `m_Neel^2` and `m_stripe^2`,
- reproducibility across seeds.

## Public-Release Hygiene

- do not commit `.env`, `results/`, or `logs/`,
- keep documentation path-agnostic,
- avoid embedding machine-specific absolute paths in generated summaries,
- prefer neutral names such as "direct-edit search" over legacy or person-specific labels.
