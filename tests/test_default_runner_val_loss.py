"""Regressions for the IL per-epoch validation step in ``DefaultRunner.train``.

Two independent things are pinned here:

1. **Reduction pattern** (original test): the val loop must accumulate python
   floats (``loss.item()``) and average with ``np.mean``, not build
   ``torch.tensor(list_of_0d_tensors)`` (warns/copies, fragile for CUDA/grad).

2. **Model & mode** (this fix): the validation loss must be computed on the
   model that is *actually deployed* at eval time and in ``eval()`` mode.
   ``default_eval_runner.py`` loads ``ema_model`` when ``use_ema`` (else
   ``model``); the train loop selects the same object into ``policy`` and calls
   ``policy.eval()``. Validation previously called ``self.model.compute_loss``
   instead of ``policy.compute_loss`` — so with the shipped ``use_ema=True``
   default it scored the raw, non-EMA model that is never put in ``eval()``
   (dropout active). ``val_loss`` thus neither tracked the deployed EMA policy
   nor was epoch-comparable — a misleading model-selection signal.

The IL runner isn't importable here (it needs ``omegaconf``/``hydra``/``zarr``
which are absent), so the behavioral test extracts the real "eval for this
epoch" region from the source and executes it against tiny stub models that
record their ``.training`` flag at ``compute_loss`` time. This keeps the test
bound to the shipped code: it fails on the pre-fix source and passes on the
fixed source.
"""

from __future__ import annotations

import pathlib
import textwrap
import types
import warnings

import numpy as np
import pytest
import torch

_RUNNER = pathlib.Path(__file__).resolve().parents[1] / "roboverse_learn" / "il" / "runners" / "default_runner.py"


# --------------------------------------------------------------------------- #
# Reduction pattern (unchanged behavior)
# --------------------------------------------------------------------------- #
@pytest.mark.general
def test_val_loss_reduction_uses_item_and_npmean():
    src = _RUNNER.read_text()
    assert "val_losses.append(loss.item())" in src, "val losses must be accumulated as python floats"
    assert "torch.tensor(val_losses)" not in src, "torch.tensor over a list of 0-d tensors is fragile; use np.mean"
    assert "np.mean(val_losses)" in src, "val loss should be averaged with np.mean (consistent with train loop)"


@pytest.mark.general
def test_item_then_npmean_is_warning_free_and_correct():
    """The new pattern (item() + np.mean) averages correctly with no warnings.

    (The old ``torch.tensor(list_of_0d_tensors)`` form is fragile for CUDA/grad
    tensors and warns on some torch versions — the fix sidesteps it entirely.)
    """
    losses = [torch.tensor(1.0), torch.tensor(3.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        result = np.mean([loss.item() for loss in losses])
    assert result == 2.0


# --------------------------------------------------------------------------- #
# Model & mode (this fix): validate the deployed model in eval() mode.
# --------------------------------------------------------------------------- #
class _StubModel(torch.nn.Module):
    """Records ``self.training`` every time ``compute_loss`` is called."""

    def __init__(self):
        super().__init__()
        self.recorded_training_flags: list[bool] = []

    def compute_loss(self, batch):
        self.recorded_training_flags.append(self.training)
        return torch.tensor(0.0)

    def predict_action(self, obs_dict):
        # Unused in tests: the sample_every branch is skipped.
        return {"action_pred": torch.zeros(1), "action": torch.zeros(1)}


def _extract_eval_region() -> str:
    """Return the dedented ``# eval for this epoch ... policy.train()`` block from the runner."""
    lines = _RUNNER.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if "# ========= eval for this epoch" in ln)
    end = next(i for i, ln in enumerate(lines) if ln.strip() == "policy.train()")
    return textwrap.dedent("\n".join(lines[start : end + 1]))


def _run_eval_region(*, use_ema: bool, n_val_batches: int = 3, max_val_steps: int = 2):
    """Exec the real eval/validation region with stub models; return the stub ``self``.

    Epoch/period values are chosen so only the *validation* branch runs
    (``sample_every`` and ``checkpoint`` branches are skipped), keeping the
    stub surface minimal.
    """
    params = types.SimpleNamespace(
        use_ema=use_ema,
        val_every=1,  # epoch(2) % 1 == 0  -> run validation
        sample_every=4,  # epoch(2) % 4 != 0  -> skip sampling
        checkpoint_every=5,  # (epoch+1=3) % 5 != 0 -> skip checkpoint
        num_epochs=100,  # epoch+1 < num_epochs -> skip checkpoint
        max_val_steps=max_val_steps,
        tqdm_interval_sec=0.0,
    )
    cfg = types.SimpleNamespace(train_config=types.SimpleNamespace(training_params=params))
    stub_self = types.SimpleNamespace(
        model=_StubModel(),
        ema_model=_StubModel(),
        epoch=2,
        cfg=cfg,
        output_dir=".",
        save_checkpoint=lambda *a, **k: None,
    )
    dataset = types.SimpleNamespace(postprocess=lambda batch, device: batch)
    val_dataloader = [object() for _ in range(n_val_batches)]

    ns = {
        "torch": torch,
        "np": np,
        "tqdm": __import__("tqdm"),
        "self": stub_self,
        "cfg": cfg,
        "dataset": dataset,
        "device": torch.device("cpu"),
        "val_dataloader": val_dataloader,
        "step_log": {},
    }
    # Runs the vetted eval/validation region extracted from the repo source.
    exec(compile(_extract_eval_region(), "<eval_region>", "exec"), ns)
    return stub_self, ns


@pytest.mark.general
def test_validation_runs_on_deployed_ema_model_in_eval_mode():
    """use_ema=True (shipped default): val runs on ema_model, in eval(), restored to train().

    Pre-fix this FAILS: validation called ``self.model.compute_loss`` (train mode,
    non-deployed) so ``ema_model`` recorded nothing and ``self.model`` recorded
    ``True`` flags.
    """
    stub_self, ns = _run_eval_region(use_ema=True, max_val_steps=2)

    ema_flags = stub_self.ema_model.recorded_training_flags
    model_flags = stub_self.model.recorded_training_flags

    # (a) validation loss is computed on the DEPLOYED model (ema_model), not self.model
    assert len(ema_flags) == 2, "validation must run on the deployed EMA model"
    assert model_flags == [], "the raw non-EMA self.model must NOT be used for validation"

    # (b) that model is in eval() mode during validation ...
    assert ema_flags == [False, False], "deployed model must be in eval() mode during validation"
    # ... and restored to train() afterward.
    assert stub_self.ema_model.training is True, "deployed model must be back in train() after eval"

    # a val_loss was actually produced
    assert "val_loss" in ns["step_log"]


@pytest.mark.general
def test_validation_uses_eval_mode_when_no_ema():
    """use_ema=False: policy is self.model; it must validate in eval() and restore train().

    Guards against a fix that eval-then-leaves the training model in eval mode.
    """
    stub_self, ns = _run_eval_region(use_ema=False, max_val_steps=2)

    model_flags = stub_self.model.recorded_training_flags
    assert len(model_flags) == 2, "validation must run on self.model when use_ema=False"
    assert model_flags == [False, False], "model must be in eval() mode during validation"
    assert stub_self.model.training is True, "the training model must be restored to train() after eval"
    # ema_model is unused in this branch
    assert stub_self.ema_model.recorded_training_flags == []
    assert "val_loss" in ns["step_log"]
