from pathlib import Path

import numpy as np

from endo3reval.data import discover_sequences, frame_id


def _sequence(root: Path, dataset_id: int, keyframe: str) -> None:
    keyframe_root = root / "dataset{}".format(dataset_id) / keyframe
    rgb = keyframe_root / "data" / "left_rectified"
    gt = keyframe_root / "data" / "depthmap_rectified"
    rgb.mkdir(parents=True)
    gt.mkdir(parents=True)
    for identifier in (2, 10):
        (rgb / "frame_{:04d}.png".format(identifier)).touch()
        np.save(
            str(gt / "depth_{:04d}.npy".format(identifier)),
            np.full((3, 4), 500.0, dtype=np.float32),
        )


def test_frame_id_uses_last_numeric_group() -> None:
    assert frame_id("camera2_frame_0010.npy") == 10


def test_discovers_preprocessed_scared_datasets_8_and_9(tmp_path: Path) -> None:
    _sequence(tmp_path, 8, "keyframe_0")
    _sequence(tmp_path, 9, "key_frame_1")
    empty = tmp_path / "dataset8" / "keyframe_4" / "data"
    empty.mkdir(parents=True)

    records, skipped = discover_sequences(
        tmp_path,
        dataset_ids=[8, 9],
        frame_sources=["left_rectified", "left"],
        ground_truth_relative="data/depthmap_rectified",
    )

    assert [record.sequence_id for record in records] == [
        "dataset_8/keyframe_0",
        "dataset_9/key_frame_1",
    ]
    assert records[0].frame_ids == (2, 10)
    assert records[0].frame_directory.name == "left_rectified"
    assert len(skipped) == 1
