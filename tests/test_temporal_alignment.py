import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from endo3reval.data import SequenceRecord
from endo3reval.temporal_alignment import (
    evaluate_tae_files,
    preflight_tae,
    read_scared_camera,
    tae_torch,
    vda_relative_pose,
)


def _record(tmp_path: Path, identifiers=(0, 1, 2)) -> tuple[SequenceRecord, dict[int, Path]]:
    keyframe = tmp_path / "dataset8" / "keyframe_0"
    rgb_directory = keyframe / "data" / "left"
    gt_directory = keyframe / "data" / "depth"
    camera_directory = keyframe / "data" / "frame_data"
    prediction_directory = tmp_path / "predictions"
    for directory in (rgb_directory, gt_directory, camera_directory, prediction_directory):
        directory.mkdir(parents=True, exist_ok=True)
    rgb_by_id = {}
    gt_by_id = {}
    prediction_by_id = {}
    for identifier in identifiers:
        rgb_path = rgb_directory / "frame_{:06d}.png".format(identifier)
        assert cv2.imwrite(str(rgb_path), np.zeros((4, 6, 3), dtype=np.uint8))
        gt_path = gt_directory / "depth_{:06d}.npy".format(identifier)
        np.save(str(gt_path), np.ones((4, 6), dtype=np.float32) * 1000.0)
        prediction_path = prediction_directory / "depth_{:06d}.npy".format(identifier)
        np.save(str(prediction_path), np.ones((4, 6), dtype=np.float32))
        camera_path = camera_directory / "frame_data{:06d}.json".format(identifier)
        camera_path.write_text(
            json.dumps(
                {
                    "camera-calibration": {
                        "KL": [[100, 0, 3], [0, 100, 2], [0, 0, 1]]
                    },
                    "camera-pose": np.eye(4).tolist(),
                }
            ),
            encoding="utf-8",
        )
        rgb_by_id[identifier] = rgb_path
        gt_by_id[identifier] = gt_path
        prediction_by_id[identifier] = prediction_path
    return (
        SequenceRecord(
            dataset_id=8,
            keyframe_id="keyframe_0",
            sequence_id="dataset_8/keyframe_0",
            keyframe_directory=keyframe,
            frame_directory=rgb_directory,
            rgb_by_id=rgb_by_id,
            ground_truth_directory=gt_directory,
            ground_truth_by_id=gt_by_id,
        ),
        prediction_by_id,
    )


def test_identity_sequence_tae_is_zero_and_reports_vggtoda3_contract(tmp_path: Path) -> None:
    record, predictions = _record(tmp_path)
    result = evaluate_tae_files(
        record,
        predictions,
        record.frame_ids,
        disparity_scale=1.0,
        disparity_shift=0.0,
        evaluation_shape=(4, 6),
        tae_config={"enabled": True, "require_all_pairs": True},
    )
    assert result["tae"] == pytest.approx(0.0)
    assert result["evaluated_pair_count"] == 2
    assert result["tae_denominator"] == "2 * (num_frames - 1)"
    assert result["tae_k_usage"] == "first_frame_K_for_both_directions_matching_vda"


def test_preflight_rejects_missing_camera_before_inference(tmp_path: Path) -> None:
    record, _ = _record(tmp_path)
    next((record.keyframe_directory / "data/frame_data").glob("*000001.json")).unlink()
    with pytest.raises(FileNotFoundError, match="TAE dataset cameras missing"):
        preflight_tae(record, {"enabled": True, "require_all_pairs": True})


def test_projection_collision_uses_vda_direct_assignment() -> None:
    depth1 = torch.full((1, 2), 0.5, dtype=torch.float64)
    angle = math.radians(-80)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=torch.float64,
    )
    translation = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
    last_z = -sine * 0.5 + cosine * 0.5
    depth2 = torch.tensor([[last_z, 1.0]], dtype=torch.float64)
    error = tae_torch(
        depth1,
        depth2,
        rotation,
        translation,
        np.eye(3),
        torch.ones_like(depth1, dtype=torch.bool),
    )
    assert float(error) == pytest.approx(0.0, abs=1e-12)


def test_scared_pose_and_intrinsics_match_vggtoda3_conversion(tmp_path: Path) -> None:
    record, _ = _record(tmp_path, identifiers=(0,))
    camera = record.keyframe_directory / "data/frame_data/frame_data000000.json"
    value = json.loads(camera.read_text(encoding="utf-8"))
    value["camera-pose"][0][3] = 1000
    camera.write_text(json.dumps(value), encoding="utf-8")
    intrinsics, pose = read_scared_camera(
        camera, record.rgb_by_id[0], (2, 3), translation_scale=0.001
    )
    np.testing.assert_allclose(intrinsics, [[50, 0, 1.5], [0, 50, 1], [0, 0, 1]])
    assert pose[0, 3] == pytest.approx(-1.0)
    np.testing.assert_allclose(vda_relative_pose(np.eye(4), np.eye(4)), np.eye(4))
