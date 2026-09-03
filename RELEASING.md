# Development and release protocol

This is the contract for how code reaches `main` and how `main` reaches users, for **MetaSim**
(this repo, PyPI distribution `roboverse-metasim`, import name `metasim`) and its downstream
**RoboVerse** (`roboverse-py`). The same protocol lives in RoboVerse's `RELEASING.md`; keep both
in sync.

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
- Required checks: `lint` (ruff check + format at the pre-commit pin), `general` (the
  simulator-free test suite on Python 3.10 and 3.11), `pr-title`, and `changelog` (every PR that
  touches `metasim/` adds a line under `## [Unreleased]` in `CHANGELOG.md`, or carries the
  `no-changelog` label for pure refactors/tests).
- Simulator-backed suites (`-k mujoco|sapien3|isaacsim|isaacgym|newton|superdex`) run in the
  merge-queue workflow on the GPU runner (`premerge-ci.yml`) and locally before a release (§4).
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
- The version is stored **once**, in `pyproject.toml` (`[project] version`); `metasim.__version__`
  reads it from package metadata. Never hand-edit a version elsewhere.
- Tags are `vX.Y.Z` on `main` and are immutable. A tag that turns out broken gets a new patch
  release, never a moved tag.
- Downstream pinning: RoboVerse depends on `roboverse-metasim @ git+https://github.com/RoboVerseOrg/MetaSim.git@vX.Y.Z`
  (a tag, not `main`) and, once PyPI publishing is enabled, on `roboverse-metasim>=X.Y,<X.Y+1`.
  Bumping that pin is a RoboVerse PR that runs its full suite against the new MetaSim.

## 4. Release checklist (MetaSim)

1. `main` is green and the merge queue is empty.
2. Run the simulator-backed suites in the environments of `ENVIRONMENTS.md`:
   `pytest -k mujoco`, `-k sapien3`, `-k isaacsim`, `-k isaacgym`, `-k newton`, `-k superdex`.
   Record the pass/skip/xfail counts in the release notes; xfails must each name a tracked reason.
3. Open a `chore(release): vX.Y.Z` PR that
   - bumps `[project] version` in `pyproject.toml`,
   - renames `## [Unreleased]` in `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD` and adds a fresh
     empty `## [Unreleased]`,
   - updates `RELEASE_NOTES_NEXT.md` if it is used for the GitHub release text.
4. Merge it, then tag the squash commit: `git tag -a vX.Y.Z -m "MetaSim vX.Y.Z" && git push origin vX.Y.Z`.
5. `release.yml` runs on the tag: it verifies the tag equals the `pyproject.toml` version, builds
   the sdist and wheel, checks them with `twine`, creates the GitHub Release with the matching
   `CHANGELOG.md` section as body, and (when the `pypi` environment is configured, §5) publishes
   to PyPI through trusted publishing.
6. Open the RoboVerse pin-bump PR (§3) the same day. RoboVerse's own release follows.
7. Announce in Discussions / Discord with the changelog link.

Patch releases: branch `release/vX.Y` from the tag, cherry-pick fixes, repeat steps 3-5 with
`base = release/vX.Y`.

## 5. Publishing to PyPI (one-time setup)

- The distribution name is **`roboverse-metasim`** — `metasim` on PyPI belongs to an unrelated
  project. The import name stays `metasim`.
- Create the project on PyPI (upload the first version manually or reserve the name), then add a
  *trusted publisher*: owner `RoboVerseOrg`, repository `MetaSim`, workflow `release.yml`,
  environment `pypi`.
- In the GitHub repo create the `pypi` environment (Settings → Environments), restricted to tags
  `v*`, with required reviewers = the release managers. Until this exists the publish job is
  skipped and the GitHub Release still happens.

## 6. Repository settings (applied through `gh api`; re-apply after any change)

```bash
gh api -X PUT repos/RoboVerseOrg/MetaSim/branches/main/protection --input - <<'JSON'
{
  "required_status_checks": {"strict": true, "contexts": ["lint", "general (3.10)", "general (3.11)", "pr-title", "changelog"]},
  "enforce_admins": false,
  "required_pull_request_reviews": {"required_approving_review_count": 0, "dismiss_stale_reviews": true},
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_linear_history": true,
  "required_conversation_resolution": true
}
JSON
gh api -X PATCH repos/RoboVerseOrg/MetaSim -f allow_squash_merge=true -f allow_merge_commit=false -f allow_rebase_merge=false -f delete_branch_on_merge=true -f squash_merge_commit_title=PR_TITLE -f squash_merge_commit_message=PR_BODY
```

## 7. Cross-repo rule

MetaSim owns the simulator contract, config types and the registry; RoboVerse owns content,
learning code and examples (see `AGENTS.md`). A feature that spans both lands in MetaSim first,
gets released (or at least tagged as a pre-release), and only then does the RoboVerse PR bump the
pin and use it. RoboVerse `main` must never depend on MetaSim `main`.
