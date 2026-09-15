"""End-to-end orchestration for Endo3R inference and VDA evaluation."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from endo3reval.config import atomic_write_json, load_config, project_path
from endo3reval.data import SequenceRecord, discover_sequences
from endo3reval.endo3r import (
    OFFICIAL_DEMO_SHA,
    OFFICIAL_REPOSITORY as ENDO3R_REPOSITORY,
    PreflightError,
    SUPPORTED_PYTHON_VERSION,
    build_demo_command,
    prediction_index,
    prediction_paths,
    predictions_complete,
    probe_python,
    resolve_python,
    run_demo,
    validate_checkpoints_cached,
)
from endo3reval.temporal_alignment import (
    TAE_REFERENCE,
    VDA_TAE_METADATA,
    evaluate_tae_files,
    preflight_tae,
)
from endo3reval.vda import (
    METRIC_NAMES,
    OFFICIAL_EVAL_SHA,
    OFFICIAL_METRIC_SHA,
    OFFICIAL_REPOSITORY as VDA_REPOSITORY,
    SCARED_MAX_DEPTH,
    SCARED_MIN_DEPTH,
    evaluate_files,
)


STAGES = ("preflight", "infer", "evaluate", "all")
OFFICIAL_MIN_DEPTH = SCARED_MIN_DEPTH


def _relative_to_repository(value: str, repository: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository / path
    return path.resolve()


def _safe_reset_scene(scene_root: Path, output_root: Path) -> None:
    prediction_root = (output_root / "predictions").resolve()
    resolved = scene_root.resolve()
    try:
        relative = resolved.relative_to(prediction_root)
    except ValueError as error:
        raise RuntimeError(
            "Refusing to clear path outside prediction root: {}".format(resolved)
        ) from error
    if not relative.parts:
        raise RuntimeError("Refusing to clear the prediction root itself")
    if resolved.exists():
        shutil.rmtree(str(resolved))


def _runtime_values(config: Mapping[str, Any]) -> Dict[str, Any]:
    project_root = Path(str(config["_project_root"]))
    dataset_config = config["dataset"]
    endo3r_config = config["endo3r"]
    evaluation_config = config["evaluation"]

    repository = project_path(endo3r_config["repository"], config)
    checkpoint = project_path(endo3r_config["checkpoint"], config)
    python_executable = resolve_python(
        str(endo3r_config["python"]), project_root
    )
    output_root = project_path(config["output_root"], config)
    dataset_root = project_path(dataset_config["root"], config)
    required_files: Dict[str, Path] = {"endo3r": checkpoint}
    for index, value in enumerate(
        endo3r_config.get("required_runtime_files", [])
    ):
        path = _relative_to_repository(str(value), repository)
        required_files["runtime_{}".format(index)] = path

    min_depth = float(evaluation_config.get("min_depth", OFFICIAL_MIN_DEPTH))
    if min_depth != OFFICIAL_MIN_DEPTH:
        raise PreflightError(
            "VDA min_depth is algorithmically fixed at {}; received {}".format(
                OFFICIAL_MIN_DEPTH, min_depth
            )
        )
    max_depth = float(evaluation_config.get("max_depth", SCARED_MAX_DEPTH))
    if max_depth != SCARED_MAX_DEPTH:
        raise PreflightError(
            "vggtoda3 SCARED max_depth is fixed at {}; received {}".format(
                SCARED_MAX_DEPTH, max_depth
            )
        )
    ground_truth_scale = float(dataset_config.get("ground_truth_scale", 0.001))
    if ground_truth_scale != 0.001:
        raise PreflightError(
            "vggtoda3 SCARED GT scale is fixed at 0.001; received {}".format(
                ground_truth_scale
            )
        )
    return {
        "project_root": project_root,
        "dataset_root": dataset_root,
        "repository": repository,
        "checkpoint": checkpoint,
        "python": python_executable,
        "output_root": output_root,
        "required_files": required_files,
    }


def _preflight(
    config: Mapping[str, Any], runtime: Mapping[str, Any]
) -> Dict[str, Any]:
    endo3r_config = config["endo3r"]
    evaluation_config = config["evaluation"]
    repository = runtime["repository"]
    output_root = runtime["output_root"]
    demo_path = repository / "demo.py"
    if not demo_path.is_file():
        raise PreflightError(
            "Official Endo3R demo.py is missing: {}. Clone {}".format(
                demo_path, ENDO3R_REPOSITORY
            )
        )
    runtime_root = output_root / ".runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    device = str(endo3r_config.get("device", "cuda:0"))
    cuda_visible_devices = endo3r_config.get("cuda_visible_devices")
    python_health = probe_python(
        runtime["python"],
        repository,
        expected_version=str(
            endo3r_config.get("python_version", SUPPORTED_PYTHON_VERSION)
        ),
        cuda_visible_devices=(
            None if cuda_visible_devices is None else str(cuda_visible_devices)
        ),
        device=device,
    )
    if str(evaluation_config.get("device", "cuda")).startswith("cuda"):
        if not python_health["cuda_available"]:
            raise PreflightError("VDA evaluation requests CUDA, but CUDA is unavailable")
    checkpoint_health = validate_checkpoints_cached(
        runtime["required_files"], runtime_root / "checkpoint_validation.json"
    )
    result = {
        "status": "ok",
        "official_sources": {
            "endo3r": ENDO3R_REPOSITORY,
            "endo3r_demo_blob_sha": OFFICIAL_DEMO_SHA,
            "video_depth_anything": VDA_REPOSITORY,
            "vda_eval_blob_sha": OFFICIAL_EVAL_SHA,
            "vda_metric_blob_sha": OFFICIAL_METRIC_SHA,
            "vda_tae_reference": TAE_REFERENCE,
        },
        "python": python_health,
        "checkpoints": checkpoint_health,
    }
    atomic_write_json(runtime_root / "preflight.json", result)
    return result


def _matched_frame_ids(
    record: SequenceRecord,
    predictions: Mapping[int, Path],
    require_all: bool,
) -> Tuple[int, ...]:
    rgb_ids = set(record.rgb_by_id)
    prediction_ids = set(predictions)
    gt_ids = set(record.ground_truth_by_id)
    if prediction_ids != rgb_ids:
        raise RuntimeError(
            "Prediction IDs do not match RGB IDs for {}. Missing: {}; extra: {}"
            .format(
                record.sequence_id,
                sorted(rgb_ids - prediction_ids)[:20],
                sorted(prediction_ids - rgb_ids)[:20],
            )
        )
    missing_gt = sorted(rgb_ids - gt_ids)
    if require_all and missing_gt:
        raise RuntimeError(
            "Ground truth is missing RGB frame IDs for {}: {}".format(
                record.sequence_id, missing_gt[:20]
            )
        )
    matched = tuple(sorted(rgb_ids & gt_ids & prediction_ids))
    if not matched:
        raise RuntimeError("No matched frames for {}".format(record.sequence_id))
    return matched


def run_pipeline(
    config_path: Path,
    stage: str = "all",
    force_inference: bool = False,
    limit_sequences: Optional[int] = None,
) -> Dict[str, Any]:
    if stage not in STAGES:
        raise ValueError("stage must be one of {}".format(STAGES))
    if limit_sequences is not None and limit_sequences <= 0:
        raise ValueError("limit_sequences must be positive")

    config = load_config(config_path)
    configured_cuda = config["endo3r"].get("cuda_visible_devices")
    if configured_cuda is not None:
        # torch is imported during module loading, but CUDA is initialized only
        # on first use. Set visibility before preflight/evaluation touches it.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(configured_cuda)
    runtime = _runtime_values(config)
    output_root = runtime["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = _preflight(config, runtime)

    dataset_config = config["dataset"]
    records, skipped = discover_sequences(
        runtime["dataset_root"],
        [int(value) for value in dataset_config.get("dataset_ids", [8, 9])],
        [str(value) for value in dataset_config.get("frame_sources", [])],
        str(dataset_config.get("ground_truth_directory", "data/depthmap_rectified")),
    )
    if limit_sequences is not None:
        records = records[:limit_sequences]
    if stage in ("preflight", "evaluate", "all"):
        tae_config = config["evaluation"].get("tae", {})
        for record in records:
            preflight_tae(record, tae_config)

    manifest_path = output_root / "run_manifest.json"
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "stage": stage,
        "config": str(config["_config_path"]),
        "dataset_root": str(runtime["dataset_root"]),
        "output_root": str(output_root),
        "preflight": preflight,
        "skipped_sequences": skipped,
        "sequences": [],
    }
    atomic_write_json(manifest_path, manifest)
    if stage == "preflight":
        manifest["status"] = "complete"
        atomic_write_json(manifest_path, manifest)
        print("[preflight] runtime, checkpoints, and dataset are valid", flush=True)
        return manifest

    endo3r_config = config["endo3r"]
    evaluation_config = config["evaluation"]
    device = str(endo3r_config.get("device", "cuda:0"))
    cuda_visible_devices = endo3r_config.get("cuda_visible_devices")
    require_all = bool(evaluation_config.get("require_all_frames", True))
    sequence_results: List[Dict[str, Any]] = []
    commands: List[List[str]] = []

    try:
        for position, record in enumerate(records, start=1):
            scene_root, depth_directory = prediction_paths(output_root, record)
            command = build_demo_command(
                runtime["python"],
                runtime["repository"],
                record,
                scene_root,
                runtime["checkpoint"],
                device=device,
                resolution=int(endo3r_config.get("resolution", 320)),
                kf_every=int(endo3r_config.get("kf_every", 1)),
            )
            commands.append(command)
            complete = predictions_complete(record, depth_directory)
            sequence_state: Dict[str, Any] = record.to_dict()
            sequence_state.update(
                {
                    "prediction_directory": str(depth_directory),
                    "inference": "reused" if complete else "pending",
                }
            )
            manifest["sequences"].append(sequence_state)
            atomic_write_json(manifest_path, manifest)

            if stage in ("infer", "all") and (force_inference or not complete):
                _safe_reset_scene(scene_root, output_root)
                scene_root.mkdir(parents=True, exist_ok=True)
                log_path = output_root / "logs" / (
                    record.sequence_id.replace("/", "_") + ".log"
                )
                print(
                    "[Endo3R] {}/{} {}".format(
                        position, len(records), record.sequence_id
                    ),
                    flush=True,
                )
                print("[Endo3R] command={}".format(command), flush=True)
                sequence_state["inference"] = "running"
                sequence_state["log"] = str(log_path)
                atomic_write_json(manifest_path, manifest)
                elapsed = run_demo(
                    command,
                    runtime["repository"],
                    None if cuda_visible_devices is None else str(cuda_visible_devices),
                    log_path,
                )
                if not predictions_complete(record, depth_directory):
                    raise RuntimeError(
                        "Endo3R did not save exactly one depth map per RGB frame: {}"
                        .format(depth_directory)
                    )
                sequence_state["inference"] = "complete"
                sequence_state["inference_seconds"] = elapsed
                atomic_write_json(manifest_path, manifest)
            elif not complete:
                raise RuntimeError(
                    "Predictions are incomplete for evaluation: {}".format(
                        depth_directory
                    )
                )

            if stage in ("evaluate", "all"):
                predictions = prediction_index(depth_directory)
                matched_ids = _matched_frame_ids(record, predictions, require_all)
                evaluated = evaluate_files(
                    predictions,
                    record.ground_truth_by_id,
                    matched_ids,
                    ground_truth_scale=float(
                        dataset_config.get("ground_truth_scale", 0.001)
                    ),
                    ground_truth_channel=int(
                        dataset_config.get("ground_truth_channel", 0)
                    ),
                    min_depth=OFFICIAL_MIN_DEPTH,
                    max_depth=SCARED_MAX_DEPTH,
                    device=str(evaluation_config.get("device", "cuda")),
                )
                alignment = evaluated["alignment"]
                evaluation_shape = tuple(evaluated["evaluation_shape_hxw"])
                temporal = evaluate_tae_files(
                    record,
                    predictions,
                    matched_ids,
                    disparity_scale=float(alignment["scale"]),
                    disparity_shift=float(alignment["shift"]),
                    evaluation_shape=evaluation_shape,
                    tae_config=evaluation_config.get("tae", {}),
                )
                evaluated["temporal"] = temporal
                evaluated["metrics"]["tae"] = temporal["tae"]
                evaluated["matched_frame_count"] = evaluated["frame_count"]
                evaluated.update(record.to_dict())
                evaluated["prediction_directory"] = str(depth_directory)
                sequence_results.append(evaluated)
                sequence_state["evaluation"] = "complete"
                sequence_state["metrics"] = evaluated["metrics"]
                atomic_write_json(manifest_path, manifest)

        result: Dict[str, Any] = {
            "schema_version": 2,
            "model": "Endo3R",
            "dataset": "SCARED",
            "dataset_ids": [
                int(value) for value in dataset_config.get("dataset_ids", [8, 9])
            ],
            "protocol": "video-depth-anything-depth+video-depth-anything-tae-scared-v2",
            **VDA_TAE_METADATA,
            "core_algorithms_modified": False,
            "adapter": (
                "official Endo3R Z-depth .npy -> reciprocal disparity -> official "
                "VDA sequence-global disparity scale/shift, spatial metrics, and TAE"
            ),
            "official_sources": preflight["official_sources"],
            "config": str(config["_config_path"]),
            "commands": commands,
            "sequence_count": len(records),
            "sequences": sequence_results or manifest["sequences"],
            "skipped_sequences": skipped,
        }
        if sequence_results:
            result["metrics"] = {
                name: float(
                    np.mean([item["metrics"][name] for item in sequence_results])
                )
                for name in METRIC_NAMES
            }
            temporal_values = [
                item["metrics"]["tae"]
                for item in sequence_results
                if item["metrics"]["tae"] is not None
            ]
            result["metrics"]["tae"] = (
                float(np.mean(temporal_values)) if temporal_values else None
            )
            result["metric_aggregation"] = "macro mean over evaluated sequences"
            evaluation_resolutions = sorted(
                {
                    tuple(item["evaluation_shape_hxw"])
                    for item in sequence_results
                }
            )
            result["evaluation_resolution_hw"] = (
                list(evaluation_resolutions[0])
                if len(evaluation_resolutions) == 1
                else None
            )
            result["evaluation_resolutions_hw"] = [
                list(shape) for shape in evaluation_resolutions
            ]
            result["evaluation_resolution_source"] = "native_endo3r_depth_output"
            result["tae_sequence_count"] = len(temporal_values)
            result["complete_gt_coverage"] = not skipped and all(
                item["matched_frame_count"] == item["frame_count"]
                and item["valid_frame_count"] > 0
                for item in sequence_results
            )
            result["complete_tae_coverage"] = bool(temporal_values) and all(
                item["temporal"]["status"] == "complete"
                for item in sequence_results
            )
            result["full_test_set"] = (
                limit_sequences is None
                and set(result["dataset_ids"]) == {8, 9}
                and not skipped
                and len(sequence_results) == len(records)
                and result["complete_gt_coverage"]
                and result["complete_tae_coverage"]
            )
            result_file = output_root / str(
                evaluation_config.get("result_file", "evaluation_vda.json")
            )
        else:
            result_file = output_root / "inference_manifest.json"
        atomic_write_json(result_file, result)
        manifest["status"] = "complete"
        manifest["result_file"] = str(result_file)
        atomic_write_json(manifest_path, manifest)
        print("wrote result: {}".format(result_file), flush=True)
        return result
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = "{}: {}".format(type(error).__name__, error)
        atomic_write_json(manifest_path, manifest)
        raise
