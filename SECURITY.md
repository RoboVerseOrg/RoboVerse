# Security policy

RoboVerse and MetaSim run simulators, load assets and trajectories from the Hugging Face Hub, and
execute policies served over HTTP. We treat anything that lets a crafted asset, trajectory, result
file or network message run code or read data it should not as a security issue.

## Reporting a vulnerability

Please report privately through GitHub's vulnerability reporting for this repository:
**Security → Report a vulnerability** (https://github.com/RoboVerseOrg/RoboVerse/security/advisories/new).
It reaches the maintainers only. Do not open a public issue for a suspected vulnerability.

If you cannot use GitHub, open an issue that says only "security contact requested" and a
maintainer will reply with a private channel within three working days.

What helps us: the affected version or commit, the component (simulator backend, asset loader,
trajectory format, `bench serve` / `RemotePolicy`, the CLI), a reproduction, and the impact you see.

## What to expect

- Acknowledgement within 3 working days, an assessment within 10.
- We agree a disclosure date with you. Our default is coordinated disclosure within **90 days** of the
  report, sooner when a fix ships earlier, and we credit reporters in the release notes unless they
  prefer not to be named.
- Fixes land in a patch release of the current minor (`1.0.x`) and are listed under `### Security`
  in the CHANGELOGs.

## Supported versions

| version | supported |
|---|---|
| 1.0.x (`main`) | yes |
| 1.0.0-beta and earlier | no: upgrade |

## Scope notes

- Trajectory files (`*.pkl`, `*.pkl.gz`) are Python pickles. Load them only from sources you trust;
  `metasim.utils.trajectory` episode files (`.npz` + JSON) are the pickle-free format and are preferred.
- `python -m metasim bench serve` binds to `127.0.0.1` by default. Exposing it on another interface
  is the operator's choice; the protocol carries no authentication.
- Simulator engines (MuJoCo, Isaac Sim, ...) are third-party software; issues in them should go to
  their maintainers, but tell us if RoboVerse makes one reachable in a new way.
