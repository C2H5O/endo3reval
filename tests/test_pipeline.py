import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from endo3reval.pipeline import run_pipeline


def test_evaluate_stage_wires_vggtoda3_spatial_and_tae_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keyframe = tmp_path / "scared/dataset8/keyframe_0"
    rgb_directory = keyframe / "data/left"
    gt_directory = keyframe / "data/depth"
    camera_directory = keyframe / "data/frame_data"
    output_root = tmp_path / "outputs"
    prediction_directory = (
        output_root / "predictions/dataset_8_keyframe_0/left/depth"
    )
    for directory in (
        rgb_directory,
        gt_directory,
        camera_directory,
        prediction_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for identifier in (0, 1):
        assert cv2.imwrite(
            str(rgb_directory / "frame_{:06d}.png".format(identifier)),
            np.zeros((4, 6, 3), dtype=np.uint8),
        )
        np.save(
            str(gt_directory / "depth_{:06d}.npy".format(identifier)),
            np.ones((4, 6), dtype=np.float32) * 1000.0,
        )
        np.save(
            str(prediction_directory / "depth_{:06d}.npy".format(identifier)),
            np.ones((4, 6), dtype=np.float32),
        )
        (camera_directory / "frame_data{:06d}.json".format(identifier)).write_text(
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
    config = {
        "dataset": {
            "root": str(tmp_path / "scared"),
            "dataset_ids": [8],
            "frame_sources": ["left"],
            "ground_truth_directory": "data/depth",
            "ground_truth_scale": 0.001,
            "ground_truth_channel": 0,
        },
        "endo3r": {
            "repository": str(tmp_path / "Endo3R"),
            "python": sys.executable,
            "checkpoint": str(tmp_path / "endo3r.pth"),
            "device": "cpu",
        },
        "evaluation": {
            "device": "cpu",
            "min_depth": 0.001,
            "max_depth": 100.0,
            "require_all_frames": True,
            "tae": {"enabled": True, "require_all_pairs": True},
            "result_file": "evaluation_vda.json",
        },
        "output_root": str(output_root),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        "endo3reval.pipeline._preflight",
        lambda *args: {"official_sources": {}, "status": "ok"},
    )

    result = run_pipeline(config_path, stage="evaluate")

    assert result["protocol"] == "video-depth-anything-depth+video-depth-anything-tae-scared-v2"
    assert result["evaluation_resolution_hw"] == [4, 6]
    assert result["evaluation_resolution_source"] == "native_endo3r_depth_output"
    assert result["metrics"]["abs_relative_difference"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert result["metrics"]["tae"] == pytest.approx(0.0)
    assert result["complete_gt_coverage"]
    assert result["complete_tae_coverage"]
    assert not result["full_test_set"]
    written = json.loads((output_root / "evaluation_vda.json").read_text(encoding="utf-8"))
    assert written["sequences"][0]["temporal"]["status"] == "complete"
