from pathlib import Path

import numpy as np
import pytest

from endo3reval.vda import (
    evaluate_files,
    official_vda_sequence_metrics,
)


def test_official_vda_sequence_alignment_recovers_affine_disparity() -> None:
    ground_truth = [
        np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32),
        np.array([[0.6, 1.2], [1.8, 2.4]], dtype=np.float32),
    ]
    gt_disparities = [1.0 / value for value in ground_truth]
    predicted = [0.5 * value + 0.2 for value in gt_disparities]

    result = official_vda_sequence_metrics(
        predicted,
        ground_truth,
        min_depth=1e-3,
        max_depth=100.0,
        device="cpu",
    )

    assert result["alignment"]["scale"] == pytest.approx(2.0, rel=1e-5)
    assert result["alignment"]["shift"] == pytest.approx(-0.4, rel=1e-5)
    assert result["metrics"]["abs_relative_difference"] < 1e-5
    assert result["metrics"]["rmse_linear"] < 1e-5
    assert result["metrics"]["delta1_acc"] == pytest.approx(1.0)


def test_file_adapter_converts_endo3r_depth_to_disparity(tmp_path: Path) -> None:
    predictions = {}
    ground_truth = {}
    for identifier, depth in enumerate((0.5, 1.0, 2.0)):
        prediction_path = tmp_path / "pred_{:04d}.npy".format(identifier)
        gt_path = tmp_path / "gt_{:04d}.npy".format(identifier)
        np.save(str(prediction_path), np.full((2, 3), depth, dtype=np.float32))
        np.save(str(gt_path), np.full((4, 6), depth * 1000.0, dtype=np.float32))
        predictions[identifier] = prediction_path
        ground_truth[identifier] = gt_path

    result = evaluate_files(
        predictions,
        ground_truth,
        frame_ids=[0, 1, 2],
        ground_truth_scale=0.001,
        ground_truth_channel=0,
        min_depth=1e-3,
        max_depth=100.0,
        device="cpu",
    )

    assert result["metrics"]["abs_relative_difference"] < 1e-5
    assert result["metrics"]["delta1_acc"] == pytest.approx(1.0)
