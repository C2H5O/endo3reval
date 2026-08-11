from pathlib import Path
import zipfile

import pytest

import endo3reval.endo3r as endo3r
from endo3reval.data import SequenceRecord
from endo3reval.endo3r import (
    PreflightError,
    SUPPORTED_PYTHON_VERSION,
    build_demo_command,
    clean_environment,
    validate_checkpoint,
)


def test_python_runtime_contract_is_3_10_20() -> None:
    assert SUPPORTED_PYTHON_VERSION == "3.10.20"


def _record(tmp_path: Path) -> SequenceRecord:
    keyframe = tmp_path / "dataset8" / "keyframe_0"
    frames = keyframe / "data" / "left"
    gt = keyframe / "data" / "depthmap_rectified"
    frames.mkdir(parents=True)
    gt.mkdir(parents=True)
    image = frames / "000000.png"
    depth = gt / "000000.npy"
    image.touch()
    depth.touch()
    return SequenceRecord(
        dataset_id=8,
        keyframe_id="keyframe_0",
        sequence_id="dataset_8/keyframe_0",
        keyframe_directory=keyframe,
        frame_directory=frames,
        rgb_by_id={0: image},
        ground_truth_directory=gt,
        ground_truth_by_id={0: depth},
    )


def test_child_environment_isolated_from_user_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tmp/other")
    monkeypatch.setenv("PYTHONHOME", "/tmp/python")
    environment = clean_environment("2")
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"


def test_demo_command_calls_official_entry_point(tmp_path: Path) -> None:
    record = _record(tmp_path)
    command = build_demo_command(
        tmp_path / "env" / "bin" / "python",
        tmp_path / "Endo3R",
        record,
        tmp_path / "outputs" / "sequence",
        tmp_path / "endo3r.pth",
        "cuda:0",
        320,
        1,
    )
    assert command[1:3] == ["-s", str(tmp_path / "Endo3R" / "demo.py")]
    assert command[command.index("--demo_path") + 1] == str(record.frame_directory)
    assert "--save_result" in command


def test_checkpoint_crc_failure_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pth"
    with zipfile.ZipFile(str(checkpoint), "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("checkpoint/data.pkl", b"x" * (1024 * 1024 + 1))
    assert validate_checkpoint(checkpoint)["crc"] == "ok"

    real_archive = zipfile.ZipFile

    class CorruptArchive:
        def __init__(self, path, mode):
            self.archive = real_archive(path, mode)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def namelist(self):
            return self.archive.namelist()

        @staticmethod
        def testzip():
            return "checkpoint/data/31"

    monkeypatch.setattr(endo3r.zipfile, "ZipFile", CorruptArchive)
    with pytest.raises(PreflightError, match="data/31"):
        validate_checkpoint(checkpoint)
