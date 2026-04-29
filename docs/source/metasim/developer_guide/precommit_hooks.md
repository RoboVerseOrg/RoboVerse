# Pre-commit Hooks

The RoboVerse downstream repository no longer ships a root `.pre-commit-config.yaml`.
For RoboVerse-only changes, run the applicable tests and checks described by the
pull request template and repository instructions.

MetaSim core development uses the pre-commit configuration in the standalone
MetaSim repository. Install hooks from that checkout when changing MetaSim:

```bash
cd MetaSim
pre-commit install
```

The [ruff VS Code extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
is still recommended for local formatting and lint feedback.
