import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "train_server", Path(__file__).parents[1] / "scripts" / "train_server.py"
)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def manifest(tmp_path, image="a.jpg"):
    path = tmp_path / "train.csv"
    path.write_text(
        "reference,positive,cf_a,cf_b,modification\n"
        + f"{image},{image},{image},{image},make blue\n" * 2,
        encoding="utf-8",
    )
    return path


def test_validate_missing_images_and_zero_batches(tmp_path):
    path = manifest(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing image"):
        server.validate_data(path, tmp_path, 2)
    (tmp_path / "a.jpg").touch()
    assert server.validate_data(path, tmp_path, 2) == 2
    with pytest.raises(ValueError, match="zero training batches"):
        server.validate_data(path, tmp_path, 4)


def test_reject_path_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes image root"):
        server.validate_data(manifest(tmp_path, "../outside.jpg"), tmp_path, 2)


def test_reject_incomplete_rows(tmp_path):
    with pytest.raises(ValueError, match="Incomplete pseudo-edit"):
        server.validate_data(manifest(tmp_path, ""), tmp_path, 2)
