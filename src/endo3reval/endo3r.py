"""Unmodified execution adapter for the official Endo3R demo entry point."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from endo3reval.config import atomic_write_json
from endo3reval.data import SequenceRecord, frame_id, index_by_frame_id


OFFICIAL_REPOSITORY = "https://github.com/wrld/Endo3R"
OFFICIAL_DEMO_SHA = "b444081d680d198253aefa85ce6f7c88ea49b7f2"


class PreflightError(RuntimeError):
    """Raised before GPU inference when runtime prerequisites are invalid."""


def clean_environment(cuda_visible_devices: Optional[str]) -> Dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    return environment


def resolve_python(value: str, project_root: Path) -> Path:
    expanded_value = os.path.expandvars(os.path.expanduser(value))
    candidate = Path(expanded_value)
    if candidate.is_absolute() or candidate.parent != Path("."):
        if not candidate.is_absolute():
            candidate = project_root / candidate
        resolved = candidate.resolve()
    else:
        located = shutil.which(expanded_value)
        if located is None:
            raise PreflightError("Python executable not found: {}".format(value))
        resolved = Path(located).resolve()
    if not resolved.is_file():
        raise PreflightError("Python executable is not a file: {}".format(resolved))
    return resolved


_PYTHON_PROBE = r"""
import importlib.metadata
import json
import sys
import numpy
import cv2
import torch
import torchvision
import scipy
import roma
import matplotlib
import tqdm
import viser
import einops
import gdown
import trimesh
import pyglet
import evo
import kornia
import demo

source = numpy.zeros((3, 4), dtype=numpy.uint8)
resized = cv2.resize(source, (8, 6), interpolation=cv2.INTER_NEAREST)
distribution_names = [
    "numpy", "opencv-python", "torch", "torchvision", "scipy", "roma",
    "matplotlib", "tqdm", "viser", "einops", "gdown", "trimesh",
    "pyglet", "evo", "kornia",
]
versions = {}
for name in distribution_names:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = "not-reported"
print("__ENDO3REVAL_PROBE__" + json.dumps({
    "python": ".".join(str(part) for part in sys.version_info[:3]),
    "versions": versions,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "resize_shape": list(resized.shape),
    "official_demo_import": "ok",
}))
"""


def probe_python(
    python_executable: Path,
    working_directory: Path,
    expected_version: str,
    cuda_visible_devices: Optional[str],
    device: str,
) -> Dict[str, Any]:
    completed = subprocess.run(
        [str(python_executable), "-s", "-c", _PYTHON_PROBE],
        cwd=str(working_directory),
        env=clean_environment(cuda_visible_devices),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    marker = "__ENDO3REVAL_PROBE__"
    payload_line = next(
        (line for line in completed.stdout.splitlines() if line.startswith(marker)),
        None,
    )
    if completed.returncode != 0 or payload_line is None:
        detail = (completed.stderr or completed.stdout).strip()
        raise PreflightError(
            "Endo3R Python probe failed (exit {}): {}".format(
                completed.returncode, detail[-3000:]
            )
        )
    payload = json.loads(payload_line[len(marker) :])
    if payload["python"] != expected_version:
        raise PreflightError(
            "Expected Python {}, found {} at {}".format(
                expected_version, payload["python"], python_executable
            )
        )
    if payload["resize_shape"] != [6, 8]:
        raise PreflightError("OpenCV/NumPy resize ABI probe failed")
    if device.startswith("cuda") and not payload["cuda_available"]:
        raise PreflightError(
            "Endo3R device is {}, but PyTorch cannot access CUDA".format(device)
        )
    payload["python_executable"] = str(python_executable)
    return payload


def _checkpoint_identity(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def validate_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise PreflightError("Required checkpoint is missing: {}".format(path))
    identity = _checkpoint_identity(path)
    if identity["size_bytes"] < 1024 * 1024:
        raise PreflightError(
            "Checkpoint is unexpectedly small: {} ({} bytes)".format(
                path, identity["size_bytes"]
            )
        )
    if not zipfile.is_zipfile(path):
        return dict(identity, archive_format="legacy-or-raw", crc="not-applicable")
    try:
        with zipfile.ZipFile(str(path), "r") as archive:
            names = archive.namelist()
            bad_member = archive.testzip()
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise PreflightError(
            "Checkpoint archive cannot be read: {} ({})".format(path, error)
        ) from error
    if bad_member is not None:
        raise PreflightError(
            "Checkpoint archive is corrupted: {} (bad member: {})".format(
                path, bad_member
            )
        )
    if not any(name.endswith("data.pkl") for name in names):
        raise PreflightError(
            "ZIP checkpoint does not contain torch data.pkl: {}".format(path)
        )
    return dict(identity, archive_format="pytorch-zip", crc="ok")


def validate_checkpoints_cached(
    checkpoints: Mapping[str, Path], cache_path: Path
) -> Dict[str, Dict[str, Any]]:
    cached: Mapping[str, Any] = {}
    if cache_path.is_file():
        try:
            value = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                cached = value.get("checkpoints", {})
        except (OSError, json.JSONDecodeError):
            cached = {}
    results: Dict[str, Dict[str, Any]] = {}
    for name, path in checkpoints.items():
        if not path.is_file():
            raise PreflightError("Required runtime file is missing: {}".format(path))
        identity = _checkpoint_identity(path)
        previous = cached.get(name) if isinstance(cached, dict) else None
        if (
            isinstance(previous, dict)
            and all(previous.get(key) == value for key, value in identity.items())
            and previous.get("crc") in ("ok", "not-applicable")
        ):
            results[name] = previous
            continue
        print("[preflight] validating checkpoint CRC: {}".format(path), flush=True)
        results[name] = validate_checkpoint(path)
    atomic_write_json(cache_path, {"checkpoints": results})
    return results


def build_demo_command(
    python_executable: Path,
    repository: Path,
    record: SequenceRecord,
    save_path: Path,
    checkpoint: Path,
    device: str,
    resolution: int,
    kf_every: int,
) -> List[str]:
    return [
        str(python_executable),
        "-s",
        str(repository / "demo.py"),
        "--demo_path",
        str(record.frame_directory),
        "--kf_every",
        str(kf_every),
        "--save_path",
        str(save_path),
        "--ckpt_path",
        str(checkpoint),
        "--device",
        device,
        "--resolution",
        str(resolution),
        "--save_result",
    ]


def prediction_paths(
    output_root: Path, record: SequenceRecord
) -> Tuple[Path, Path]:
    safe_name = record.sequence_id.replace("/", "_").replace("\\", "_")
    scene_root = output_root / "predictions" / safe_name
    depth_directory = scene_root / record.frame_directory.name / "depth"
    return scene_root, depth_directory


def prediction_index(directory: Path) -> Dict[int, Path]:
    if not directory.is_dir():
        return {}
    paths = sorted(directory.glob("*.npy"))
    return index_by_frame_id(paths)


def predictions_complete(record: SequenceRecord, directory: Path) -> bool:
    return set(prediction_index(directory)) == set(record.rgb_by_id)


def run_demo(
    command: Sequence[str],
    repository: Path,
    cuda_visible_devices: Optional[str],
    log_path: Path,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=str(repository),
            env=clean_environment(cuda_visible_devices),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Could not capture Endo3R output")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code != 0:
        raise RuntimeError(
            "Official Endo3R demo.py failed with exit code {}. Log: {}".format(
                return_code, log_path
            )
        )
    return elapsed
