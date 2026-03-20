# Security and Publication Audit

## Executive Summary

A publication-focused security and privacy audit was performed on the tracked
contents of this repository before the initial private upload. No critical or
high-severity issues were found in the versioned files. In particular, no API
keys, no machine-specific absolute filesystem paths, and no personal email
addresses were found in the tracked project contents.

The main residual risk is not in tracked source files, but in ignored local
artifacts such as `.env`, `results/`, local virtual environments, and local git
metadata. These are intentionally excluded from version control and should stay
excluded in future releases.

## Scope

The audit focused on tracked repository contents, excluding intentionally
ignored local artifacts:

- source files under the repository root
- documentation under `docs/`
- notebooks under `notebooks/`
- analysis data committed for public reference

Excluded from the audit scope for publication because they are not versioned:

- `.env`
- `.venv/`
- `results/`
- `logs/`
- local `.git/` metadata
- Python cache directories

## Checks Performed

1. Secret scanning by pattern
   Checked for likely API keys and tokens, including Gemini, Groq, OpenAI,
   GitHub token formats, and generic secret markers.

1b. Git-history secret scan
   Checked every commit reachable from `main` for the same token patterns to
   reduce the risk that a secret had been committed and later removed.

2. Personal-data scanning
   Checked for personal names, email addresses, local filesystem paths, hostnames,
   and workstation-specific prompt fragments.

3. Publication-language review
   Checked for internal-facing phrasing, placeholder wording, and references that
   would read as developer notes rather than public documentation.

4. Release-hygiene review
   Confirmed that `.env`, `results/`, `logs/`, `.venv/`, caches, and notebook
   checkpoints are git-ignored.

## Findings

### No critical findings in tracked files

No real secrets or personal machine paths were detected in tracked repository
contents at the time of the audit.

No matching secret patterns were detected in the reachable git history of the
private repository either.

### Informational note I1

Template variables remain intentionally present in
[.env.example](.env.example)
to document the expected environment variables. These are placeholders, not
active credentials.

### Informational note I2

An intentional attribution to Andrej Karpathy's
[`autoresearch`](https://github.com/karpathy/autoresearch) remains in the public
documentation because it is part of the methodological context of the project.

## Actions Completed

- Added or hardened ignores for `.env`, `results/`, `logs/`, `.venv/`,
  cache directories, and notebook checkpoints.
- Removed machine-specific absolute paths from tracked docs and notebooks.
- Replaced legacy internal naming in the public-facing docs with neutral wording.
- Configured the initial git commit for this repository to use a GitHub noreply
  address rather than a personal email address.

## Residual Risks

- Ignored runtime outputs may still contain local paths or historical internal
  names on the development machine. They are not part of the pushed repository,
  but future manual uploads should avoid attaching them without a fresh review.
- Future notebooks or generated markdown summaries can easily reintroduce local
  paths if copied directly from runtime outputs without sanitization.

## Recommendation

The current private repository state is suitable for review and iteration on
GitHub. Before any future public release, rerun the same audit after regenerating
figures, notebooks, or benchmark summaries.
