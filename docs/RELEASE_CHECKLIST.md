# Release Checklist

Use this checklist before publishing a new snapshot of the repository.

## Privacy and Safety

- `.env` is not committed
- local virtual environments are ignored
- generated `results/` and `logs/` directories are ignored
- no machine-specific absolute paths appear in tracked documentation
- no personal names or internal labels remain in public-facing docs unless scientifically necessary

## Technical Sanity

- `python -m unittest tests.test_regressions` passes
- `python prepare.py` runs
- `python exact_diag.py` runs
- `TV_DEVICE=cpu python train.py --j2 0.0 --sector sz0 --max-steps 5` runs

## Documentation

- README matches the current repository state
- public results are summarized in `docs/RESULTS.md`
- methodology notes match the current evaluation scripts
- commands in docs are relative and reproducible
