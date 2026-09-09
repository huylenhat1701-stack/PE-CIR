from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import Tensor, nn

from pic2word.models.pic2word_model import Pic2WordOutput
from pic2word.training.trainer import Pic2WordTrainer, TrainerConfig


class ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0))

    @property
    def logit_scale(self) -> Tensor:
        return self.scale

    def freeze(self) -> None:
        self.eval()


class ToyPic2WordModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = ToyBackbone()
        self.mapping_network = nn.Linear(4, 4)

    def forward(self, images: Tensor, prompts: list[str]) -> Pic2WordOutput:
        del prompts
        image_features = images.flatten(start_dim=1).detach()
        pseudo_tokens = self.mapping_network(image_features)
        return Pic2WordOutput(image_features, pseudo_tokens, pseudo_tokens)


class TrainerTests(unittest.TestCase):
    def test_training_step_and_checkpoint_round_trip(self) -> None:
        model = ToyPic2WordModel()
        trainer = Pic2WordTrainer(
            model,  # type: ignore[arg-type]
            TrainerConfig(warmup_steps=2, precision="fp32"),
            device="cpu",
        )
        original_weight = model.mapping_network.weight.detach().clone()
        batch = torch.eye(4).reshape(4, 1, 2, 2)
        metrics = trainer.train_step(batch, "a photo of *")

        self.assertEqual(metrics.step, 1)
        self.assertTrue(torch.isfinite(torch.tensor(metrics.total_loss)))
        self.assertFalse(torch.equal(original_weight, model.mapping_network.weight))

        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
            trainer.save_checkpoint(checkpoint_path)
            restored_model = ToyPic2WordModel()
            restored = Pic2WordTrainer(
                restored_model,  # type: ignore[arg-type]
                TrainerConfig(warmup_steps=2, precision="fp32"),
                device="cpu",
            )
            restored.load_checkpoint(checkpoint_path)

            self.assertEqual(restored.state.global_step, 1)
            self.assertTrue(
                torch.equal(model.mapping_network.weight, restored_model.mapping_network.weight)
            )


if __name__ == "__main__":
    unittest.main()
