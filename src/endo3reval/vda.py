"""Video Depth Anything benchmark protocol adapted to Endo3R depth files.

The scale/shift alignment and metric equations follow the official files at
the pinned source SHAs below.  Dataset/file loading and Endo3R depth-to-
disparity conversion are the only project-specific adapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import cv2
import torch


OFFICIAL_REPOSITORY = "https://github.com/DepthAnything/Video-Depth-Anything"
OFFICIAL_EVAL_SHA = "411d416aceb3a827331efc5fcdcdba15b3023936"
OFFICIAL_METRIC_SHA = "6cfd4106d1e929fc2f688b80b01ae8769bea7f77"
METRIC_NAMES = (
    "abs_relative_difference",
    "rmse_linear",
    "delta1_acc",
)


class EvaluationError(RuntimeError):
    """Raised when predictions cannot be evaluated with the VDA protocol."""


def abs_relative_difference(
    output: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()


def rmse_linear(
    output: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    diff = output - target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    return torch.sqrt(mse).mean()


def threshold_percentage(
    output: torch.Tensor,
    target: torch.Tensor,
    threshold_value: float,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_value, one, zero)
    if valid_mask is not None:
        valid_mask_cpu = valid_mask.cpu()
        bit_mat[~valid_mask_cpu] = 0
        n = valid_mask_cpu.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n
    return threshold_mat.mean()


def delta1_acc(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    return threshold_percentage(prediction, target, 1.25, valid_mask)


def depth_to_disparity(depth: np.ndarray) -> np.ndarray:
    disparity = np.zeros_like(depth)
    positive = depth > 0
    disparity[positive] = 1.0 / depth[positive]
    return disparity


def _load_array(path: Path, channel: int = 0) -> np.ndarray:
    if path.suffix.casefold() == ".npy":
        value = np.load(str(path), allow_pickle=False)
    else:
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None:
            raise EvaluationError("Could not read depth image: {}".format(path))
    value = np.asarray(value)
    if value.ndim == 3:
        if value.shape[-1] <= 4:
            value = value[..., channel]
        elif value.shape[0] <= 4:
            value = value[channel]
    value = np.squeeze(value)
    if value.ndim != 2:
        raise EvaluationError(
            "Depth array must be 2D after channel selection: {} has {}".format(
                path, value.shape
            )
        )
    value = value.astype(np.float32, copy=False)
    if not np.all(np.isfinite(value)):
        raise EvaluationError("Depth contains NaN or Inf: {}".format(path))
    return value


def load_ground_truth(path: Path, scale: float, channel: int) -> np.ndarray:
    value = _load_array(path, channel=channel) * float(scale)
    value[value == 0] = -1
    return value


def load_endo3r_prediction(path: Path, target_shape: Tuple[int, int]) -> np.ndarray:
    depth = _load_array(path)
    if depth.shape != target_shape:
        # Official VDA get_infer uses cv2.resize without an interpolation override.
        depth = cv2.resize(depth, (target_shape[1], target_shape[0]))
    # Official VDA evaluates relative disparity. Endo3R saves Z depth, so this
    # boundary adapter converts only the representation, not the algorithm.
    return depth_to_disparity(depth)


def official_vda_sequence_metrics(
    predicted_disparities: Sequence[np.ndarray],
    ground_truth_depths: Sequence[np.ndarray],
    min_depth: float,
    max_depth: float,
    device: str,
) -> Dict[str, Any]:
    """Apply the official VDA sequence-global alignment and three metrics."""
    if len(predicted_disparities) != len(ground_truth_depths):
        raise EvaluationError("Prediction and GT sequence lengths differ")
    if not predicted_disparities:
        raise EvaluationError("Cannot evaluate an empty sequence")

    infs = np.stack(predicted_disparities, axis=0)
    gts = np.stack(ground_truth_depths, axis=0)
    valid_mask = np.logical_and(gts > min_depth, gts < max_depth)
    if not np.any(valid_mask):
        raise EvaluationError("No valid ground-truth pixels in sequence")

    gt_disp_masked = 1.0 / (
        gts[valid_mask].reshape((-1, 1)).astype(np.float64) + 1e-8
    )
    infs = np.clip(infs, a_min=1e-3, a_max=None)
    pred_disp_masked = infs[valid_mask].reshape((-1, 1)).astype(np.float64)

    ones = np.ones_like(pred_disp_masked)
    matrix = np.concatenate([pred_disp_masked, ones], axis=-1)
    solution = np.linalg.lstsq(matrix, gt_disp_masked, rcond=None)[0]
    scale, shift = solution
    aligned_prediction = scale * infs + shift
    aligned_prediction = np.clip(aligned_prediction, a_min=1e-3, a_max=None)

    predicted_depth = depth_to_disparity(aligned_prediction)
    predicted_depth = np.clip(
        predicted_depth, a_min=1e-3, a_max=max_depth
    )

    valid_counts = valid_mask.sum((-1, -2))
    valid_frames = valid_counts > 0
    predicted_tensor = torch.from_numpy(predicted_depth[valid_frames]).to(device)
    ground_truth_tensor = torch.from_numpy(gts[valid_frames]).to(device)
    valid_mask_tensor = torch.from_numpy(valid_mask[valid_frames]).to(device)

    metric_values = {
        "abs_relative_difference": float(
            abs_relative_difference(
                predicted_tensor, ground_truth_tensor, valid_mask_tensor
            ).item()
        ),
        "rmse_linear": float(
            rmse_linear(
                predicted_tensor, ground_truth_tensor, valid_mask_tensor
            ).item()
        ),
        "delta1_acc": float(
            delta1_acc(
                predicted_tensor, ground_truth_tensor, valid_mask_tensor
            ).item()
        ),
    }
    return {
        "metrics": metric_values,
        "alignment": {
            "domain": "disparity",
            "scope": "one scale and shift for the complete sequence",
            "scale": float(np.asarray(scale).reshape(-1)[0]),
            "shift": float(np.asarray(shift).reshape(-1)[0]),
        },
        "frame_count": len(predicted_disparities),
        "valid_frame_count": int(valid_frames.sum()),
        "valid_pixel_count": int(valid_mask.sum()),
    }


def evaluate_files(
    prediction_by_id: Mapping[int, Path],
    ground_truth_by_id: Mapping[int, Path],
    frame_ids: Sequence[int],
    ground_truth_scale: float,
    ground_truth_channel: int,
    min_depth: float,
    max_depth: float,
    device: str,
) -> Dict[str, Any]:
    predicted_disparities: List[np.ndarray] = []
    ground_truth_depths: List[np.ndarray] = []
    for identifier in frame_ids:
        ground_truth = load_ground_truth(
            ground_truth_by_id[identifier],
            scale=ground_truth_scale,
            channel=ground_truth_channel,
        )
        prediction = load_endo3r_prediction(
            prediction_by_id[identifier], target_shape=ground_truth.shape
        )
        predicted_disparities.append(prediction)
        ground_truth_depths.append(ground_truth)
    result = official_vda_sequence_metrics(
        predicted_disparities,
        ground_truth_depths,
        min_depth=min_depth,
        max_depth=max_depth,
        device=device,
    )
    result["frame_ids"] = list(frame_ids)
    return result
