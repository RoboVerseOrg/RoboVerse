from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.benchmark.rendering import seed_stability_metrics as ssm


def _write_constant_exr(path: Path, value: float, shape: tuple[int, int, int] = (8, 8, 3)) -> None:
    arr = np.full(shape, value, dtype=np.float32)
    bgr = arr[..., ::-1]  # cv2 expects BGR
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), bgr)
    assert ok, f"cv2.imwrite failed for {path}"


def test_load_exr_stack_returns_float32_rgb_stack(tmp_path) -> None:
    paths = []
    for index, value in enumerate([0.1, 0.2, 0.3]):
        p = tmp_path / f"img_{index}.exr"
        _write_constant_exr(p, value)
        paths.append(p)
    stack = ssm.load_exr_stack(paths)
    assert stack.dtype == np.float32
    assert stack.shape == (3, 8, 8, 3)
    assert np.allclose(stack[0], 0.1, atol=1e-3)
    assert np.allclose(stack[2], 0.3, atol=1e-3)


def test_load_exr_stack_preserves_rgb_channel_order(tmp_path) -> None:
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    arr[..., 0] = 0.9  # R
    arr[..., 1] = 0.5  # G
    arr[..., 2] = 0.1  # B
    path = tmp_path / "rgb.exr"
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = arr[..., ::-1]
    assert cv2.imwrite(str(path), bgr)
    stack = ssm.load_exr_stack([path])
    assert np.allclose(stack[0, ..., 0], 0.9, atol=1e-3)
    assert np.allclose(stack[0, ..., 1], 0.5, atol=1e-3)
    assert np.allclose(stack[0, ..., 2], 0.1, atol=1e-3)


def test_load_exr_stack_raises_on_shape_mismatch(tmp_path) -> None:
    a = tmp_path / "a.exr"
    b = tmp_path / "b.exr"
    _write_constant_exr(a, 0.1, shape=(8, 8, 3))
    _write_constant_exr(b, 0.1, shape=(16, 16, 3))
    with pytest.raises(ValueError) as excinfo:
        ssm.load_exr_stack([a, b])
    assert "shape mismatch" in str(excinfo.value).lower()
    assert "b.exr" in str(excinfo.value)


def test_load_exr_stack_raises_on_missing_file(tmp_path) -> None:
    a = tmp_path / "a.exr"
    _write_constant_exr(a, 0.1)
    missing = tmp_path / "missing.exr"
    with pytest.raises(FileNotFoundError) as excinfo:
        ssm.load_exr_stack([a, missing])
    assert "missing.exr" in str(excinfo.value)


def test_per_pixel_stack_stats_shape_and_dtype() -> None:
    rng = np.random.default_rng(0)
    stack = rng.random((5, 4, 4, 3), dtype=np.float32)
    mean, std = ssm.per_pixel_stack_stats(stack)
    assert mean.dtype == np.float32 and std.dtype == np.float32
    assert mean.shape == (4, 4, 3) and std.shape == (4, 4, 3)


def test_per_pixel_stack_stats_zero_for_identical_inputs() -> None:
    base = np.full((4, 4, 3), 0.42, dtype=np.float32)
    stack = np.stack([base, base, base, base], axis=0)
    mean, std = ssm.per_pixel_stack_stats(stack)
    assert np.allclose(mean, 0.42)
    assert np.allclose(std, 0.0)


def test_per_pixel_stack_stats_matches_numpy_population_std() -> None:
    rng = np.random.default_rng(7)
    stack = rng.random((6, 3, 3, 3), dtype=np.float32)
    _, std = ssm.per_pixel_stack_stats(stack)
    assert np.allclose(std, np.std(stack, axis=0, ddof=0), atol=1e-6)


def test_crop_returns_a_view_into_the_image() -> None:
    image = np.zeros((10, 10, 3), dtype=np.float32)
    image[2:5, 3:6, :] = 0.7
    crop = ssm.crop(image, [3, 2, 6, 5])
    assert crop.shape == (3, 3, 3)
    assert np.allclose(crop, 0.7)


def test_crop_raises_on_empty_box() -> None:
    image = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        ssm.crop(image, [3, 3, 3, 3])


def test_group_means_collapses_named_rois() -> None:
    per_roi = {
        "left_display_case": 0.10,
        "center_bottle": 0.20,
        "robot_left_arm": 0.30,
        "target_cube": 0.40,
    }
    grouping = {
        "glass": ["left_display_case", "center_bottle"],
        "non_glass": ["robot_left_arm", "target_cube"],
    }
    grouped = ssm.group_means(per_roi, grouping)
    assert np.isclose(grouped["glass"], 0.15)
    assert np.isclose(grouped["non_glass"], 0.35)


def test_metric_runner_rmse_mae_psnr_zero_for_identical_inputs() -> None:
    runner = ssm.MetricRunner.numerical_only()
    a = np.full((8, 8, 3), 0.4, dtype=np.float32)
    b = a.copy()
    assert runner.rmse(a, b) == pytest.approx(0.0)
    assert runner.mae(a, b) == pytest.approx(0.0)
    assert runner.psnr(a, b) >= 99.0


def test_metric_runner_psnr_clipping() -> None:
    runner = ssm.MetricRunner.numerical_only()
    a = np.zeros((4, 4, 3), dtype=np.float32)
    b = np.full((4, 4, 3), 0.5, dtype=np.float32)
    psnr = runner.psnr(a, b)
    assert 5.0 < psnr < 15.0


def test_metric_runner_skips_lpips_and_flip_when_disabled() -> None:
    runner = ssm.MetricRunner.numerical_only()
    a = np.zeros((4, 4, 3), dtype=np.float32)
    b = np.zeros((4, 4, 3), dtype=np.float32)
    assert runner.lpips(a, b) is None
    assert runner.flip(a, b) is None


def test_pairwise_metrics_zero_rmse_for_identical_inputs() -> None:
    img = np.full((4, 4, 3), 0.5, dtype=np.float32)
    stack = np.stack([img, img, img], axis=0)
    summary = ssm.pairwise_metrics(stack, runner=ssm.MetricRunner.numerical_only(), roi_boxes=None, roi_grouping=None)
    assert summary["global"]["rmse"]["mean"] == pytest.approx(0.0)
    assert summary["global"]["mae"]["mean"] == pytest.approx(0.0)
    assert summary["global"]["rmse"]["std"] == pytest.approx(0.0)


def test_pairwise_metrics_counts_unordered_pairs() -> None:
    rng = np.random.default_rng(1)
    stack = rng.random((4, 4, 4, 3), dtype=np.float32)
    summary = ssm.pairwise_metrics(stack, runner=ssm.MetricRunner.numerical_only(), roi_boxes=None, roi_grouping=None)
    assert summary["global"]["rmse"]["n_pairs"] == 6


def test_pairwise_metrics_emits_grouped_roi_keys() -> None:
    rng = np.random.default_rng(2)
    stack = rng.random((3, 8, 8, 3), dtype=np.float32)
    boxes = {"glass_a": [0, 0, 4, 4], "non_glass_a": [4, 4, 8, 8]}
    grouping = {"glass": ["glass_a"], "non_glass": ["non_glass_a"]}
    summary = ssm.pairwise_metrics(
        stack, runner=ssm.MetricRunner.numerical_only(), roi_boxes=boxes, roi_grouping=grouping
    )
    assert "glass" in summary
    assert "non_glass" in summary
    assert summary["glass"]["rmse"]["n_pairs"] == 3


def test_pairwise_metrics_raises_for_fewer_than_two_images() -> None:
    img = np.zeros((4, 4, 3), dtype=np.float32)
    stack = np.stack([img], axis=0)
    with pytest.raises(ValueError):
        ssm.pairwise_metrics(stack, runner=ssm.MetricRunner.numerical_only(), roi_boxes=None, roi_grouping=None)


def test_vs_reference_metrics_returns_n_rows() -> None:
    rng = np.random.default_rng(3)
    stack = rng.random((4, 8, 8, 3), dtype=np.float32)
    reference = rng.random((8, 8, 3), dtype=np.float32)
    rows = ssm.vs_reference_metrics(
        stack, reference, runner=ssm.MetricRunner.numerical_only(), roi_boxes=None, roi_grouping=None
    )
    assert len(rows) == 4
    for index, row in enumerate(rows):
        assert row["seed_index"] == index
        for m in ("rmse", "mae", "psnr"):
            assert m in row


def test_vs_reference_metrics_zero_when_seed_equals_reference() -> None:
    img = np.full((4, 4, 3), 0.25, dtype=np.float32)
    stack = np.stack([img, img], axis=0)
    rows = ssm.vs_reference_metrics(
        stack, img, runner=ssm.MetricRunner.numerical_only(), roi_boxes=None, roi_grouping=None
    )
    assert rows[0]["rmse"] == pytest.approx(0.0)
    assert rows[1]["rmse"] == pytest.approx(0.0)


def test_vs_reference_metrics_raises_on_shape_mismatch() -> None:
    stack = np.zeros((2, 4, 4, 3), dtype=np.float32)
    reference = np.zeros((8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        ssm.vs_reference_metrics(
            stack, reference, runner=ssm.MetricRunner.numerical_only(), roi_boxes=None, roi_grouping=None
        )


def test_vs_reference_metrics_emits_grouped_roi_columns() -> None:
    img = np.full((8, 8, 3), 0.3, dtype=np.float32)
    stack = np.stack([img, img], axis=0)
    boxes = {"glass_a": [0, 0, 4, 4], "non_glass_a": [4, 4, 8, 8]}
    grouping = {"glass": ["glass_a"], "non_glass": ["non_glass_a"]}
    rows = ssm.vs_reference_metrics(
        stack, img, runner=ssm.MetricRunner.numerical_only(), roi_boxes=boxes, roi_grouping=grouping
    )
    assert "glass_rmse" in rows[0]
    assert "non_glass_rmse" in rows[0]
    assert rows[0]["glass_rmse"] == pytest.approx(0.0)


def test_render_std_map_writes_png_and_returns_max(tmp_path) -> None:
    rng = np.random.default_rng(4)
    std_image = rng.random((16, 16, 3), dtype=np.float32) * 0.05
    out = tmp_path / "std.png"
    max_value = ssm.render_std_map(std_image, out, color_scale=None)
    assert out.exists()
    assert max_value > 0.0
    second = tmp_path / "std2.png"
    second_max = ssm.render_std_map(std_image, second, color_scale=max_value)
    assert second.exists()
    assert second_max == pytest.approx(max_value)
