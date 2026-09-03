# Development and release protocol

This is the contract for how code reaches `main` and how `main` reaches users, for **RoboVerse**
(distribution `roboverse-py`, repo root) and its core **MetaSim**
(`roboverse-metasim`, import name `metasim`, `packages/metasim` in this monorepo). One protocol, one tag, two packages;
keep both in sync.

## 1. Branches

- `main` is always releasable: every commit on it passed the required checks below.
- Work happens on short-lived branches named `<type>/<scope>-<slug>`, e.g. `feat/superdex-backend`,
  `fix/hf-util-lock`, `maint/lint`. Types are the Conventional Commits types
  (`feat|fix|docs|style|refactor|test|chore|ci|perf`).
- No long-lived `develop`. Backports go to `release/vX.Y` branches created from the tag only when a
  patch release is needed.
- Branch protection on `main` (configured through the GitHub API, see §6): pull requests only, no
  force pushes, no deletion, required status checks must pass, conversations must be resolved.

## 2. Pull requests

- One concern per PR. Title is a Conventional Commit (`type(scope): summary`); the `pr-title`
  check enforces it because the squash-merge commit takes the PR title.
- Required checks: `lint` (ruff check + format at the pre-commit pin), `tests` (the whole
  `tests/` suite on CPU; tests that need a GPU simulator or `roboverse_data` skip with a reason),
  `task-contract`, `pr-title`, and `changelog` (every PR that touches `roboverse_pack/`,
  `roboverse_learn/`, `scripts/` or `examples/` adds a line under `## [Unreleased]` in
  `CHANGELOG.md`, or carries the `no-changelog` label).
- Simulator-backed parity runs (`scripts/parity_*.py`, `scripts/eval_*_cross_sim.py`) are executed
  locally before a release (§4) and their numbers go into the release notes.
- Every behaviour change ships with a test; every public API change ships with docs and a
  `CHANGELOG.md` entry that says what a user must do (migration line) if anything breaks.
- Reviews: at least one maintainer review before merge once a second maintainer is active
  (`required_approving_review_count` in §6 is 0 today for a single-maintainer repo; raise it then).
- Merge method: **squash** (linear history, one commit per PR, commit message = PR title + body).

## 3. Versioning

- Semantic Versioning. `MAJOR.MINOR.PATCH`; pre-releases `X.Y.ZrcN`.
  - PATCH: fixes only, no API change.
  - MINOR: new backends, new config fields, new queries; existing call sites keep working.
  - MAJOR: a documented breaking change to `BaseSimHandler`, the scenario cfg types, or the task
    registry contract.
- The version is stored in exactly two places that must agree: `pyproject.toml` and
  `packages/metasim/pyproject.toml` (`[project] version`). The packages are released in **lockstep**:
  one `vX.Y.Z` tag builds and publishes both, and `release.yml` refuses a tag that does not match
  both versions. `roboverse-py` depends on `roboverse-metasim>=X.Y,<X.Y+1`.
- Tags are `vX.Y.Z` on `main` and are immutable. A tag that turns out broken gets a new patch
  release, never a moved tag.
- Each package keeps its own `CHANGELOG.md` (`CHANGELOG.md`, `packages/metasim/CHANGELOG.md`);
  `changelog.yml` requires an `[Unreleased]` line in the one whose library code a PR touches. The
  GitHub Release body concatenates both sections for the version.

## 4. Release checklist (RoboVerse)

1. `main` is green and the merge queue is empty.
2. Run the `get_started` tutorials on MuJoCo and at least one other backend (`--headless`) and the
   parity scripts; record pass/skip/xfail counts and parity numbers in the release notes.
3. Open a `chore(release): vX.Y.Z` PR that
   - bumps `[project] version` in `pyproject.toml`,
   - renames `## [Unreleased]` in `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD` and adds a fresh
     empty `## [Unreleased]`,
4. Merge it, then tag the squash commit: `git tag -a vX.Y.Z -m "RoboVerse vX.Y.Z" && git push origin vX.Y.Z`.
5. `release.yml` runs on the tag: it verifies the tag equals the `pyproject.toml` version, builds
   the sdist and wheel, checks them with `twine`, creates the GitHub Release with the matching
   `CHANGELOG.md` section as body, and (when the `pypi` environment is configured, §5) publishes
   to PyPI through trusted publishing.
6. Announce in Discussions / Discord with the changelog link and the MetaSim version it pairs with.

Patch releases: branch `release/vX.Y` from the tag, cherry-pick fixes, repeat steps 3-5 with
`base = release/vX.Y`.

## 5. Publishing to PyPI (one-time setup)

- The distribution name is `roboverse-py`. PyPI already lists `roboverse-py` 0.1.17 with no
  author metadata; confirm the RoboVerseOrg account owns that project before the first automated
  upload (if it does not, pick `roboverse-pack` and update `pyproject.toml`).
- Add a *trusted publisher* on the PyPI project: owner `RoboVerseOrg`, repository `RoboVerse`,
  workflow `release.yml`, environment `pypi`.
- In the GitHub repo create the `pypi` environment (Settings → Environments), restricted to tags
  `v*`, with required reviewers = the release managers. Until this exists the publish job is
  skipped and the GitHub Release still happens.

## 6. Repository settings (applied through `gh api`; re-apply after any change)

```bash
gh api -X PUT repos/RoboVerseOrg/RoboVerse/branches/main/protection --input - <<'JSON'
{
  "required_status_checks": {"strict": true, "contexts": ["lint", "tests", "task-contract", "pr-title", "changelog"]},
  "enforce_admins": false,
  "required_pull_request_reviews": {"required_approving_review_count": 0, "dismiss_stale_reviews": true},
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_linear_history": true,
  "required_conversation_resolution": true
}
JSON
gh api -X PATCH repos/RoboVerseOrg/RoboVerse -f allow_squash_merge=true -f allow_merge_commit=false -f allow_rebase_merge=false -f delete_branch_on_merge=true -f squash_merge_commit_title=PR_TITLE -f squash_merge_commit_message=PR_BODY
```

## 7. Package boundary

MetaSim (`packages/metasim`) owns the simulator contract, config types and the registry; the repo
root owns content, learning code and examples (see `AGENTS.md`). A feature that spans both lands in
one PR with a CHANGELOG line in each package. The boundary is enforced by imports, not by repos:
nothing outside `packages/metasim` may import a `metasim.sim.*` underscore name, and
`roboverse_pack` must not import `roboverse_learn`.

The former standalone repository `RoboVerseOrg/MetaSim` is a read-only mirror of `packages/metasim`
kept for one release cycle so `git+https://github.com/RoboVerseOrg/MetaSim.git` installs keep
working; new work happens here.

## 8. Simulator backend versions

Simulator packages change their APIs between minor releases. The policy is declarative and checked:

- `packages/metasim/metasim/sim/_versions.py` lists, per backend, the distributions it depends on,
  the version range its code paths support (`spec`) and the exact version its test suite last
  passed with (`tested`). `get_sim_handler_class` refuses a version outside `spec` (message names
  the installed and supported versions; `METASIM_SKIP_VERSION_CHECK=1` downgrades it to a warning)
  and warns once on an untested version inside it.
- `python -m metasim doctor` prints the table for every backend (`--json` for scripts; exit 1 when
  an installed backend is unsupported). Run it first when a simulator misbehaves; paste it in issues.
- `backend-compat.yml` runs weekly (and on demand): it installs the **newest** release of each
  CPU-installable backend, runs `metasim doctor` and that backend's suite, and opens/updates a
  `backend-compat` issue on failure. GPU backends are covered by the merge-queue workflow.
- Dependabot (`.github/dependabot.yml`) groups simulator bumps into one weekly PR per package root.
- When a new simulator release passes: bump `tested`; when a range must widen or a shim is needed:
  change `spec` and the compat module (`metasim/sim/<backend>/_*_compat.py`) in the same PR, with a
  CHANGELOG line. Never widen `spec` without a passing run.
