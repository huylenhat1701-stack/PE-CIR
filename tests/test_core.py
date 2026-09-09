from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pic2word.config import MappingNetworkConfig, TrainingConfig
from pic2word.retrieval.prompts import (
    build_domain_prompt,
    build_object_prompt,
    build_sentence_prompt,
    validate_prompt,
)

try:
    import torch
except ModuleNotFoundError:  # The light smoke test can run before ML dependencies are installed.
    torch = None


class PromptTests(unittest.TestCase):
    def test_paper_prompt_templates(self) -> None:
        self.assertEqual(build_domain_prompt("origami"), "a origami of *")
        self.assertEqual(
            build_object_prompt(["cat", "dog"]),
            "a photo of *, cat, dog",
        )
        self.assertEqual(
            build_sentence_prompt("with longer sleeves"),
            "a photo of *, with longer sleeves",
        )

    def test_prompt_requires_one_placeholder(self) -> None:
        with self.assertRaises(ValueError):
            validate_prompt("a photo without an image token")
        with self.assertRaises(ValueError):
            validate_prompt("a * next to *")

    def test_configuration_validation(self) -> None:
        TrainingConfig()
        MappingNetworkConfig(image_embedding_dim=768, token_embedding_dim=768)
        with self.assertRaises(ValueError):
            MappingNetworkConfig(image_embedding_dim=0, token_embedding_dim=768)


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class TensorCoreTests(unittest.TestCase):
    def test_cc3m_image_dataloader(self) -> None:
        from PIL import Image
        from torchvision.transforms import ToTensor

        from pic2word.data.cc3m import CC3MImageDataset, build_cc3m_dataloader

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_root = root / "images"
            image_root.mkdir()
            Image.new("RGB", (8, 8), color="red").save(image_root / "first.jpg")
            Image.new("RGB", (8, 8), color="blue").save(image_root / "second.jpg")
            manifest_path = root / "manifest.csv"
            manifest_path.write_text("image\nfirst.jpg\nsecond.jpg\n", encoding="utf-8")

            dataset = CC3MImageDataset(manifest_path, image_root, ToTensor())
            self.assertEqual(len(dataset), 2)
            self.assertEqual(tuple(dataset[0].shape), (3, 8, 8))

            loader = build_cc3m_dataloader(
                manifest_path,
                image_root,
                ToTensor(),
                batch_size=2,
                num_workers=0,
                shuffle=False,
            )
            batch = next(iter(loader))
            self.assertEqual(tuple(batch.shape), (2, 3, 8, 8))

    def test_mapping_network_shape_and_gradient(self) -> None:
        from pic2word.models.mapping_network import MappingNetwork

        config = MappingNetworkConfig(
            image_embedding_dim=768,
            token_embedding_dim=768,
            hidden_dim=512,
            hidden_layers=2,
            dropout=0.1,
        )
        model = MappingNetwork(config)
        image_features = torch.randn(4, 768)
        pseudo_tokens = model(image_features)
        self.assertEqual(tuple(pseudo_tokens.shape), (4, 768))

        pseudo_tokens.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_token_injection_preserves_pseudo_token_gradient(self) -> None:
        from pic2word.models.token_injection import inject_pseudo_token

        token_ids = torch.tensor([[1, 99, 2], [3, 99, 4]])
        token_embeddings = torch.randn(2, 3, 8)
        pseudo_tokens = torch.randn(2, 8, requires_grad=True)
        output = inject_pseudo_token(
            token_ids,
            token_embeddings,
            pseudo_tokens,
            placeholder_token_id=99,
        )
        self.assertTrue(torch.equal(output[:, 1], pseudo_tokens))

        output.sum().backward()
        self.assertIsNotNone(pseudo_tokens.grad)

    def test_symmetric_contrastive_loss(self) -> None:
        from pic2word.training.losses import symmetric_contrastive_loss

        image_features = torch.eye(4, requires_grad=True)
        text_features = torch.eye(4, requires_grad=True)
        result = symmetric_contrastive_loss(
            image_features,
            text_features,
            logit_scale=10.0,
        )
        self.assertEqual(tuple(result.logits.shape), (4, 4))
        self.assertTrue(torch.isfinite(result.total))
        result.total.backward()
        self.assertIsNotNone(text_features.grad)

    def test_pic2word_forward_only_trains_mapping_network(self) -> None:
        from torch import nn

        from pic2word.models.clip_backbone import FrozenCLIPBackbone
        from pic2word.models.mapping_network import MappingNetwork
        from pic2word.models.pic2word_model import Pic2WordModel

        class FakeVisual:
            output_dim = 4

        class FakeTransformer(nn.Module):
            def get_cast_dtype(self):
                return torch.float32

            def forward(self, values, attn_mask=None):
                del attn_mask
                return values + values.mean(dim=1, keepdim=True)

        class FakeCLIP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.visual = FakeVisual()
                self.token_embedding = nn.Embedding(100, 4)
                self.positional_embedding = nn.Parameter(torch.zeros(5, 4))
                self.transformer = FakeTransformer()
                self.attn_mask = None
                self.ln_final = nn.Identity()
                self.text_pool_type = "argmax"
                self.text_projection = nn.Parameter(torch.eye(4))
                self.logit_scale = nn.Parameter(torch.tensor(1.0))

            def encode_image(self, images, normalize=False):
                del normalize
                return images

        def fake_tokenizer(prompts):
            del prompts
            return torch.tensor([[98, 5, 99, 0, 0], [98, 5, 99, 0, 0]])

        backbone = FrozenCLIPBackbone(FakeCLIP(), fake_tokenizer)
        mapping = MappingNetwork(
            MappingNetworkConfig(
                image_embedding_dim=4,
                token_embedding_dim=4,
                hidden_dim=8,
                hidden_layers=2,
                dropout=0.0,
            )
        )
        model = Pic2WordModel(backbone, mapping)
        model.train()

        result = model(torch.randn(2, 4), ["a photo of *", "a photo of *"])
        self.assertEqual(tuple(result.image_features.shape), (2, 4))
        self.assertEqual(tuple(result.pseudo_tokens.shape), (2, 4))
        self.assertEqual(tuple(result.text_features.shape), (2, 4))

        result.text_features.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in mapping.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in backbone.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in backbone.parameters()))


if __name__ == "__main__":
    unittest.main()
