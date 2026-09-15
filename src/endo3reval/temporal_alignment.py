"""Video-Depth-Anything TAE with only SCARED/Endo3R I/O adapters.

The projection, collision handling, masking, bidirectional pairing, empty-pair
value, and denominator match vggtoda3's ``evaluation/temporal_alignment.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch

from endo3reval.data import SequenceRecord, frame_id, index_by_frame_id
from endo3reval.vda import SCARED_MAX_DEPTH, load_endo3r_prediction


TAE_REFERENCE = "DepthAnything/Video-Depth-Anything benchmark/eval/eval_tae.py"
VDA_TAE_METADATA = {
    "tae_reference": TAE_REFERENCE,
    "tae_unit": "percent",
    "tae_direction": "lower_is_better",
    "tae_bidirectional": True,
    "tae_pairing": "adjacent_frames",
    "tae_projection_collision": "direct_assignment_matching_vda",
    "tae_empty_projection": "zero_error_matching_vda",
    "tae_alignment": "single_sequence_disparity_scale_shift_fit",
    "tae_denominator": "2 * (num_frames - 1)",
}


def compute_errors_torch(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """AbsRel helper matching the Video-Depth-Anything reference."""
    return torch.mean(torch.abs(gt - pred) / gt)


def tae_torch(
    depth1: torch.Tensor,
    depth2: torch.Tensor,
    R_2_1: torch.Tensor,
    T_2_1: torch.Tensor,
    K: np.ndarray,
    mask: torch.Tensor,
):
    """Pairwise TAE matching the operational VDA reference literally."""
    height, width = depth1.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xx, yy = torch.meshgrid(torch.arange(width), torch.arange(height))
    xx, yy = xx.t(), yy.t()
    xx = xx.to(dtype=depth1.dtype, device=depth1.device)
    yy = yy.to(dtype=depth1.dtype, device=depth1.device)
    x = (xx - cx) * depth1 / fx
    y = (yy - cy) * depth1 / fy
    points3d = torch.stack((x.flatten(), y.flatten(), depth1.flatten()), dim=1)
    translation = torch.tensor(T_2_1, dtype=depth1.dtype, device=depth1.device)
    transformed = torch.matmul(points3d, R_2_1.T) + translation
    x_world = transformed[:, 0]
    y_world = transformed[:, 1]
    z_world = transformed[:, 2]
    x_plane = torch.round((x_world * fx) / z_world + cx).to(dtype=torch.long)
    y_plane = torch.round((y_world * fy) / z_world + cy).to(dtype=torch.long)
    valid = (
        (x_plane >= 0)
        & (x_plane < width)
        & (y_plane >= 0)
        & (y_plane < height)
    )
    if valid.sum() == 0:
        return 0
    depth_projected = torch.zeros(
        (height, width), dtype=depth1.dtype, device=depth1.device
    )
    # Direct indexed assignment, including its order-dependent collision
    # behavior, is intentionally inherited from VDA/vggtoda3.
    depth_projected[y_plane[valid], x_plane[valid]] = z_world[valid]
    valid = (depth_projected > 0) & (depth2 > 0) & mask
    if valid.sum() == 0:
        return 0
    return compute_errors_torch(depth2[valid], depth_projected[valid])


def camera_index(
    record: SequenceRecord, frame_data_relative_directory: str
) -> Tuple[Path, Dict[int, Path]]:
    directory = record.keyframe_directory / frame_data_relative_directory
    paths = sorted(path for path in directory.glob("*.json") if path.is_file())
    return directory, index_by_frame_id(paths)


def preflight_tae(record: SequenceRecord, tae_config: Mapping[str, Any]) -> None:
    """Reject incomplete camera data before starting expensive inference."""
    if not bool(tae_config.get("enabled", True)):
        return
    directory, cameras = camera_index(
        record,
        str(tae_config.get("frame_data_relative_directory", "data/frame_data")),
    )
    if bool(tae_config.get("require_all_pairs", True)):
        missing = sorted(set(record.frame_ids) - set(cameras))
        if missing:
            raise FileNotFoundError(
                "TAE dataset cameras missing in {}: {}".format(directory, missing[:20])
            )


def read_scared_camera(
    path: Path,
    rgb_path: Path,
    output_shape: Tuple[int, int],
    translation_scale: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return resized K and a VDA-compatible camera-to-world pose."""
    record = json.loads(path.read_text(encoding="utf-8"))
    intrinsics = np.asarray(record["camera-calibration"]["KL"], dtype=np.float64)
    raw_w2c = np.asarray(record["camera-pose"], dtype=np.float64)
    if intrinsics.shape != (3, 3) or raw_w2c.shape != (4, 4):
        raise ValueError("TAE requires KL[3,3] and camera-pose[4,4]: {}".format(path))
    if not (np.isfinite(intrinsics).all() and np.isfinite(raw_w2c).all()):
        raise ValueError("Non-finite TAE camera: {}".format(path))
    rotation = raw_w2c[:3, :3]
    if (
        intrinsics[0, 0] <= 0
        or intrinsics[1, 1] <= 0
        or not np.allclose(intrinsics[2], [0, 0, 1])
    ):
        raise ValueError("Invalid pinhole intrinsics: {}".format(path))
    if (
        not np.allclose(raw_w2c[3], [0, 0, 0, 1])
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3)
    ):
        raise ValueError("TAE camera-pose is not a rigid SCARED world-to-camera transform")
    if not np.isfinite(translation_scale) or translation_scale <= 0:
        raise ValueError("pose_translation_scale must be finite and positive")
    w2c_metres = raw_w2c.copy()
    w2c_metres[:3, 3] *= translation_scale
    pose_c2w = np.linalg.inv(w2c_metres)
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read RGB image for TAE intrinsics: {}".format(rgb_path))
    height, width = image.shape[:2]
    intrinsics = np.diag(
        [output_shape[1] / width, output_shape[0] / height, 1.0]
    ) @ intrinsics
    return intrinsics, pose_c2w


def vda_relative_pose(T_1_c2w: np.ndarray, T_2_c2w: np.ndarray) -> np.ndarray:
    """Map camera-1 coordinates to camera 2 using VDA's convention."""
    return np.linalg.inv(T_2_c2w) @ T_1_c2w


def _metadata(**values: Any) -> Dict[str, Any]:
    return {**VDA_TAE_METADATA, **values}


def evaluate_tae_files(
    record: SequenceRecord,
    prediction_by_id: Mapping[int, Path],
    frame_ids: Sequence[int],
    disparity_scale: float,
    disparity_shift: float,
    evaluation_shape: Tuple[int, int],
    tae_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate a complete Endo3R sequence with vggtoda3's VDA TAE path."""
    if not bool(tae_config.get("enabled", True)):
        return _metadata(status="disabled", tae=None)
    directory, cameras = camera_index(
        record,
        str(tae_config.get("frame_data_relative_directory", "data/frame_data")),
    )
    strict = bool(tae_config.get("require_all_pairs", True))
    scale = np.asarray([disparity_scale], dtype=np.float64)
    shift = np.asarray([disparity_shift], dtype=np.float64)
    frames = []
    skipped = []
    for identifier in frame_ids:
        if identifier not in cameras:
            skipped.append({"frame_id": identifier, "reason": "missing_dataset_camera"})
            continue
        intrinsics, pose_c2w = read_scared_camera(
            cameras[identifier],
            record.rgb_by_id[identifier],
            evaluation_shape,
            float(tae_config.get("pose_translation_scale", 0.001)),
        )
        disparity = load_endo3r_prediction(prediction_by_id[identifier])
        if disparity.shape != evaluation_shape:
            raise ValueError(
                "TAE expected native Endo3R shape {}, found {} at frame {}".format(
                    evaluation_shape, disparity.shape, identifier
                )
            )
        disparity = np.clip(disparity, a_min=1e-3, a_max=None)
        aligned = np.clip(scale * disparity + shift, a_min=1e-3, a_max=None)
        depth = np.clip(np.reciprocal(aligned), a_min=1e-3, a_max=SCARED_MAX_DEPTH)
        frames.append((identifier, depth, intrinsics, pose_c2w))
    if skipped and strict:
        first = skipped[0]
        raise RuntimeError(
            "TAE frame {}: {}; camera directory: {}".format(
                first["frame_id"], first["reason"], directory
            )
        )
    if len(frames) < 2:
        if strict:
            raise RuntimeError("TAE requires at least two frames in {}".format(record.sequence_id))
        return _metadata(
            status="unavailable",
            tae=None,
            evaluated_frame_count=len(frames),
            evaluated_pair_count=0,
            skipped_frames=skipped,
            camera_directory=str(directory),
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    error_sum = 0.0
    intrinsic_differences = []
    for index in range(len(frames) - 1):
        _, depth1_numpy, intrinsics, pose1 = frames[index]
        _, depth2_numpy, intrinsics2, pose2 = frames[index + 1]
        intrinsic_differences.append(float(np.max(np.abs(intrinsics2 - intrinsics))))
        transform_2_1 = vda_relative_pose(pose1, pose2)
        rotation_2_1 = torch.from_numpy(transform_2_1[:3, :3]).to(device=device)
        translation_2_1 = torch.from_numpy(transform_2_1[:3, 3]).to(device=device)
        depth1 = torch.from_numpy(depth1_numpy).to(device=device)
        depth2 = torch.from_numpy(depth2_numpy).to(device=device)
        mask1 = torch.ones_like(depth1, dtype=torch.bool)
        mask2 = torch.ones_like(depth2, dtype=torch.bool)
        error1 = tae_torch(
            depth1, depth2, rotation_2_1, translation_2_1, intrinsics, mask2
        )
        transform_1_2 = np.linalg.inv(transform_2_1)
        rotation_1_2 = torch.from_numpy(transform_1_2[:3, :3]).to(device=device)
        translation_1_2 = torch.from_numpy(transform_1_2[:3, 3]).to(device=device)
        # vggtoda3/VDA deliberately use the first frame's K in both directions.
        error2 = tae_torch(
            depth2, depth1, rotation_1_2, translation_1_2, intrinsics, mask1
        )
        error_sum += error1
        error_sum += error2
    pair_count = len(frames) - 1
    tae = error_sum / (2 * pair_count) * 100
    return _metadata(
        status="complete" if not skipped else "partial",
        tae=float(tae),
        evaluated_frame_count=len(frames),
        evaluated_pair_count=pair_count,
        skipped_frames=skipped,
        camera_directory=str(directory),
        camera_source="SCARED frame_data KL and camera-pose",
        scared_raw_pose_convention="world_to_camera",
        vda_pose_convention="camera_to_world",
        relative_transform="inv(T_2_c2w) @ T_1_c2w maps camera_1 to camera_2",
        pose_translation_scale=float(tae_config.get("pose_translation_scale", 0.001)),
        tae_mask="all_true_no_additional_scared_evaluation_mask",
        tae_k_usage="first_frame_K_for_both_directions_matching_vda",
        adjacent_intrinsics_all_equal=all(value == 0.0 for value in intrinsic_differences),
        max_adjacent_intrinsics_abs_difference=max(intrinsic_differences),
    )


__all__ = [
    "TAE_REFERENCE",
    "VDA_TAE_METADATA",
    "compute_errors_torch",
    "evaluate_tae_files",
    "preflight_tae",
    "read_scared_camera",
    "tae_torch",
    "vda_relative_pose",
]
