"""Discovery and numeric frame matching for preprocessed SCARED datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".heic"}
DEPTH_SUFFIXES = {".npy", ".png", ".tif", ".tiff", ".exr"}
DATASET_PATTERN = re.compile(r"^dataset[\s_-]*0*(\d+)$", re.IGNORECASE)
KEYFRAME_PATTERN = re.compile(r"^key[\s_-]*frame[\s_-]*(.+)$", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(\d+)")


class DiscoveryError(RuntimeError):
    """Raised when the requested preprocessed SCARED split is unusable."""


def natural_key(value: Union[str, Path]) -> List[Union[int, str]]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    ]


def frame_id(path: Union[str, Path]) -> int:
    matches = NUMBER_PATTERN.findall(Path(path).stem)
    if not matches:
        raise DiscoveryError("No numeric frame ID in filename: {}".format(path))
    return int(matches[-1])


def index_by_frame_id(paths: Iterable[Path]) -> Dict[int, Path]:
    indexed: Dict[int, Path] = {}
    for path in paths:
        identifier = frame_id(path)
        if identifier in indexed:
            raise DiscoveryError(
                "Duplicate frame ID {}: {} and {}".format(
                    identifier, indexed[identifier], path
                )
            )
        indexed[identifier] = path
    return indexed


def _files(directory: Path, suffixes: set) -> List[Path]:
    if not directory.is_dir():
        return []
    values = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in suffixes
    ]
    values.sort(key=natural_key)
    return values


@dataclass(frozen=True)
class SequenceRecord:
    dataset_id: int
    keyframe_id: str
    sequence_id: str
    keyframe_directory: Path
    frame_directory: Path
    rgb_by_id: Mapping[int, Path]
    ground_truth_directory: Path
    ground_truth_by_id: Mapping[int, Path]

    @property
    def frame_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self.rgb_by_id))

    @property
    def frame_count(self) -> int:
        return len(self.rgb_by_id)

    def to_dict(self) -> Dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "keyframe_id": self.keyframe_id,
            "sequence_id": self.sequence_id,
            "keyframe_directory": str(self.keyframe_directory),
            "frame_directory": str(self.frame_directory),
            "frame_source": self.frame_directory.name,
            "ground_truth_directory": str(self.ground_truth_directory),
            "frame_count": self.frame_count,
            "frame_ids": list(self.frame_ids),
        }


def _dataset_directories(root: Path, dataset_ids: Sequence[int]) -> Dict[int, Path]:
    if not root.is_dir():
        raise DiscoveryError("SCARED root is not a directory: {}".format(root))
    discovered: Dict[int, Path] = {}
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = DATASET_PATTERN.match(path.name)
        if not match:
            continue
        identifier = int(match.group(1))
        if identifier in discovered:
            raise DiscoveryError(
                "Duplicate dataset ID {} below {}".format(identifier, root)
            )
        discovered[identifier] = path
    missing = [identifier for identifier in dataset_ids if identifier not in discovered]
    if missing:
        raise DiscoveryError(
            "Missing SCARED dataset IDs {} below {}".format(missing, root)
        )
    return {identifier: discovered[identifier] for identifier in dataset_ids}


def _select_frames(
    keyframe_directory: Path, frame_sources: Sequence[str]
) -> Optional[Tuple[Path, Dict[int, Path]]]:
    for source in frame_sources:
        directory = keyframe_directory / "data" / source
        paths = _files(directory, IMAGE_SUFFIXES)
        if paths:
            return directory, index_by_frame_id(paths)
    return None


def discover_sequences(
    root: Path,
    dataset_ids: Sequence[int],
    frame_sources: Sequence[str],
    ground_truth_relative: str,
) -> Tuple[List[SequenceRecord], List[str]]:
    """Discover usable dataset/keyframe sequences; missing-RGB keyframes are skipped."""
    if not dataset_ids:
        raise DiscoveryError("dataset_ids must not be empty")
    if not frame_sources:
        raise DiscoveryError("frame_sources must not be empty")
    records: List[SequenceRecord] = []
    skipped: List[str] = []
    for dataset_id, dataset_directory in _dataset_directories(
        root.resolve(), dataset_ids
    ).items():
        keyframes = [
            path
            for path in dataset_directory.iterdir()
            if path.is_dir() and KEYFRAME_PATTERN.match(path.name)
        ]
        keyframes.sort(key=natural_key)
        if not keyframes:
            raise DiscoveryError(
                "No keyframe directories below {}".format(dataset_directory)
            )
        for keyframe in keyframes:
            selected = _select_frames(keyframe, frame_sources)
            if selected is None:
                skipped.append("{}: no RGB frames".format(keyframe))
                continue
            frame_directory, rgb_by_id = selected
            gt_directory = keyframe / ground_truth_relative
            gt_paths = _files(gt_directory, DEPTH_SUFFIXES)
            if not gt_paths:
                skipped.append("{}: no ground-truth depth".format(keyframe))
                continue
            ground_truth_by_id = index_by_frame_id(gt_paths)
            sequence_id = "dataset_{}/{}".format(dataset_id, keyframe.name)
            records.append(
                SequenceRecord(
                    dataset_id=dataset_id,
                    keyframe_id=keyframe.name,
                    sequence_id=sequence_id,
                    keyframe_directory=keyframe,
                    frame_directory=frame_directory,
                    rgb_by_id=rgb_by_id,
                    ground_truth_directory=gt_directory,
                    ground_truth_by_id=ground_truth_by_id,
                )
            )
    if not records:
        raise DiscoveryError("No usable SCARED sequences found below {}".format(root))
    records.sort(key=lambda item: (item.dataset_id, natural_key(item.keyframe_id)))
    return records, skipped
