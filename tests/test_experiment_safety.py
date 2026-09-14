import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
import torch
from test_trainer import ToyPic2WordModel

from pic2word.data.cirr import CIRRQuery, CIRRSplit
from pic2word.evaluation.cirr import evaluate_cirr
from pic2word.preflight import validate_training_inputs
from pic2word.retrieval.index import CandidateIndex
from pic2word.training import Pic2WordTrainer, TrainerConfig


def test_resume_mid_epoch_does_not_skip_remaining_batches():
    torch.manual_seed(0)
    model = ToyPic2WordModel()
    config = TrainerConfig(precision="fp32", warmup_steps=0)
    trainer = Pic2WordTrainer(model, config, device="cpu")
    batches = [torch.randn(4, 1, 2, 2) for _ in range(4)]
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    trainer.train_epoch(batches, prompt="a photo of *", max_steps=2)
    assert trainer.state.epoch == 0
    assert trainer.state.batches_in_epoch == 2
    with TemporaryDirectory() as directory:
        checkpoint = trainer.save_checkpoint(Path(directory) / "last.pt")
        resumed = Pic2WordTrainer(ToyPic2WordModel(), config, device="cpu")
        resumed.load_checkpoint(checkpoint)
        resumed.train_epoch(batches, prompt="a photo of *")
    assert resumed.state.global_step == 4
    assert resumed.state.epoch == 1
    assert resumed.state.batches_in_epoch == 0
    reference_model = ToyPic2WordModel()
    reference_model.load_state_dict(initial)
    reference = Pic2WordTrainer(reference_model, config, device="cpu")
    reference.train_epoch(batches, prompt="a photo of *")
    for a, b in zip(reference.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_preflight_rejects_missing_images_and_zero_batch_epoch(tmp_path):
    manifest = tmp_path / "train.csv"
    manifest.write_text("image\na.jpg\nb.jpg\n", encoding="utf-8")
    config = {
        "model": {"prompt": "a photo of *"},
        "training": {"batch_size_per_device": 2, "epochs": 1,
                     "save_every_epochs": 1, "num_workers": 0},
        "data": {"train_manifest": str(manifest), "image_root": str(tmp_path)},
    }
    with pytest.raises(FileNotFoundError, match="training images missing"):
        validate_training_inputs(config)
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).touch()
    assert validate_training_inputs(config)["samples"] == 2
    config["training"]["batch_size_per_device"] = 4
    with pytest.raises(ValueError, match="zero steps"):
        validate_training_inputs(config)


def test_cirr_saves_query_predictions_and_excludes_reference(tmp_path):
    paths = {name: tmp_path / f"{name}.jpg" for name in ("ref", "target", "other")}
    query = CIRRQuery(1, "ref", "target", "make it blue", tuple(paths))
    dataset = CIRRSplit(tmp_path, "val", "rc2", (query,), paths)
    index = CandidateIndex(list(paths.values()), torch.tensor([[1., 0.], [1., 0.], [0., 1.]]))
    with patch("pic2word.evaluation.cirr.encode_composed_query",
               return_value=(torch.tensor([1., 0.]), None)):
        result = evaluate_cirr(None, index, dataset)
    assert result.global_recall[1] == 100
    record = result.predictions[0]
    assert record["pair_id"] == 1
    assert record["global_top50"] == ["target", "other"]
    assert "ref" not in record["group_top3"]
    json.dumps(record)
