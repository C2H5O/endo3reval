from pathlib import Path

import numpy as np
import pytest

from endo3reval.vda import (
    EvaluationError,
    evaluate_files,
    load_ground_truth,
    official_vda_sequence_metrics,
)


def test_prediction_must_match_locked_reference_grid(tmp_path: Path) -> None:
    prediction_path = tmp_path / "pred_000000.npy"
    ground_truth_path = tmp_path / "depth_000000.npy"
    np.save(prediction_path, np.ones((4, 6), dtype=np.float32))
    np.save(ground_truth_path, np.ones((4, 6), dtype=np.float32) * 1000.0)
    with pytest.raises(EvaluationError, match="configured evaluation shape"):
        evaluate_files(
            {0: prediction_path},
            {0: ground_truth_path},
            [0],
            ground_truth_scale=0.001,
            ground_truth_channel=0,
            min_depth=0.001,
            max_depth=100.0,
            device="cpu",
            evaluation_shape=(256, 320),
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
    assert result["evaluation_shape_hxw"] == [2, 3]


def test_ground_truth_resize_matches_vggtoda3_nearest_neighbor(tmp_path: Path) -> None:
    path = tmp_path / "depth.npy"
    np.save(
        path,
        np.array([[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float32),
    )
    resized = load_ground_truth(path, 0.001, 0, target_shape=(2, 4))
    np.testing.assert_allclose(
        resized,
        np.array([[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]], dtype=np.float32),
    )
