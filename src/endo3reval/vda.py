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
SCARED_MIN_DEPTH = 1e-3
SCARED_MAX_DEPTH = 100.0


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


def load_ground_truth(
    path: Path,
    scale: float,
    channel: int,
    target_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    value = _load_array(path, channel=channel) * float(scale)
    if target_shape is not None and value.shape != target_shape:
        value = cv2.resize(
            value,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return value


def load_endo3r_prediction(path: Path) -> np.ndarray:
    depth = _load_array(path)
    # Match vggtoda3's student-depth boundary exactly. Endo3R saves Z depth;
    # the shared evaluator consumes relative disparity. The native Endo3R grid
    # is deliberately preserved here.
    return np.reciprocal(np.clip(depth, a_min=1e-3, a_max=None))


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
    evaluation_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    ground_truth_depth_values: List[np.ndarray] = []
    predicted_disparity_values: List[np.ndarray] = []
    valid_pixel_count = 0
    for identifier in frame_ids:
        # Preserve Endo3R's native output grid. Only GT is resized to that grid.
        prediction = load_endo3r_prediction(prediction_by_id[identifier])
        if evaluation_shape is None:
            evaluation_shape = tuple(prediction.shape)
        elif prediction.shape != evaluation_shape:
            raise EvaluationError(
                "Endo3R native prediction must match the configured evaluation "
                "shape: "
                "expected {}, found {} at frame {}".format(
                    evaluation_shape, prediction.shape, identifier
                )
            )
        ground_truth = load_ground_truth(
            ground_truth_by_id[identifier],
            scale=ground_truth_scale,
            channel=ground_truth_channel,
            target_shape=evaluation_shape,
        )
        prediction = np.clip(prediction, a_min=1e-3, a_max=None)
        valid = (ground_truth > min_depth) & (ground_truth < max_depth)
        if not np.any(valid):
            continue
        ground_truth_depth_values.append(ground_truth[valid])
        predicted_disparity_values.append(prediction[valid])
        valid_pixel_count += int(valid.sum())
    if evaluation_shape is None:
        raise EvaluationError("Cannot evaluate an empty sequence")
    if not ground_truth_depth_values:
        raise EvaluationError("No valid ground-truth pixels in sequence")
    # Match vggtoda3: one global float64 lstsq over all GT-valid pixels.
    gt_disp_masked = 1.0 / (
        np.concatenate(ground_truth_depth_values)
        .reshape((-1, 1))
        .astype(np.float64)
        + 1e-8
    )
    pred_disp_masked = (
        np.concatenate(predicted_disparity_values)
        .reshape((-1, 1))
        .astype(np.float64)
    )
    matrix = np.concatenate(
        [pred_disp_masked, np.ones_like(pred_disp_masked)], axis=-1
    )
    scale, shift = np.linalg.lstsq(matrix, gt_disp_masked, rcond=None)[0]

    metric_sums = np.zeros(len(METRIC_NAMES), dtype=np.float64)
    valid_frame_count = 0
    metric_functions = (abs_relative_difference, rmse_linear, delta1_acc)
    for identifier in frame_ids:
        prediction = load_endo3r_prediction(prediction_by_id[identifier])
        if prediction.shape != evaluation_shape:
            raise EvaluationError(
                "Endo3R native prediction shapes differ within one sequence"
            )
        ground_truth = load_ground_truth(
            ground_truth_by_id[identifier],
            scale=ground_truth_scale,
            channel=ground_truth_channel,
            target_shape=evaluation_shape,
        )
        valid = (ground_truth > min_depth) & (ground_truth < max_depth)
        if not np.any(valid):
            continue
        disparity = np.clip(prediction, a_min=1e-3, a_max=None)
        aligned = np.clip(scale * disparity + shift, a_min=1e-3, a_max=None)
        predicted_depth = np.clip(
            depth_to_disparity(aligned), a_min=1e-3, a_max=max_depth
        )
        prediction_tensor = torch.from_numpy(predicted_depth[None])
        ground_truth_tensor = torch.from_numpy(ground_truth[None])
        valid_tensor = torch.from_numpy(valid[None])
        for index, function in enumerate(metric_functions):
            metric_sums[index] += function(
                prediction_tensor, ground_truth_tensor, valid_tensor
            ).item()
        valid_frame_count += 1
    if valid_frame_count == 0:
        raise EvaluationError("No valid frames remain in sequence")
    metric_values = metric_sums / valid_frame_count
    result = {
        "metrics": {
            name: float(value) for name, value in zip(METRIC_NAMES, metric_values)
        },
        "alignment": {
            "domain": "disparity",
            "scope": "one scale and shift for the complete sequence",
            "scale": float(np.asarray(scale).reshape(-1)[0]),
            "shift": float(np.asarray(shift).reshape(-1)[0]),
        },
        "frame_count": len(frame_ids),
        "valid_frame_count": valid_frame_count,
        "valid_pixel_count": valid_pixel_count,
    }
    result["frame_ids"] = list(frame_ids)
    result["native_prediction_resolution_hw"] = list(evaluation_shape)
    result["evaluation_shape_hxw"] = list(evaluation_shape)
    result["evaluation_size"] = [evaluation_shape[1], evaluation_shape[0]]
    return result
