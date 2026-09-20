from __future__ import annotations

import json
import runpy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import pytest
from PIL import Image
from torchvision.transforms import ToTensor

from pic2word.data import PseudoEditDataset, load_fashioniq_split
from pic2word.models import CFPECIRModel
from pic2word.training import counterfactual_margin_loss, factorization_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP regression")
def test_b5_verifier_backward_under_cuda_amp() -> None:
    train_script = Path(__file__).resolve().parents[1] / "scripts/train_cfpe.py"
    loss_fn = runpy.run_path(str(train_script))["training_verifier_loss"]
    torch.manual_seed(0)
    model = CFPECIRModel(16, variant="B5", slots=4, hidden_dim=12).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=4096.0)
    reference, text, positive, cf_a, cf_b = [torch.randn(4, 16, device="cuda") for _ in range(5)]
    with torch.autocast("cuda", dtype=torch.float16):
        values = model(reference, text)
        loss = loss_fn(model, values, text, positive, cf_a, cf_b)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    for module in (model.verifier, model.factorizer, model.composer):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)
    scaler.step(optimizer)
    scaler.update()


def test_cfpe_variants_and_reranking() -> None:
    model = CFPECIRModel(16, variant="B5", slots=4, hidden_dim=12)
    output = model(torch.randn(3, 16), torch.randn(3, 16))
    assert output.query.shape == (3, 16)
    assert output.gates.shape == (3, 4)
    assert torch.allclose(output.query.norm(dim=-1), torch.ones(3), atol=1e-5)
    candidates = torch.randn(3, 5, 16)
    scores = model.verify(candidates, output, torch.randn(3, 16))
    assert scores.preserve.shape == (3, 5)
    reranked = model.rerank_scores(torch.randn(3, 5), scores)
    assert reranked.shape == (3, 5)
    assert model.uses_counterfactuals and model.uses_verifier


def test_cf_margin_prefers_positive() -> None:
    query = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[0.0, 1.0]])
    good = counterfactual_margin_loss(query, positive, negative, negative)
    bad = counterfactual_margin_loss(query, negative, positive, positive)
    assert good < bad


def test_factorization_loss_is_finite() -> None:
    loss = factorization_loss(
        torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8), torch.rand(2, 4)
    )
    assert torch.isfinite(loss)


def test_pseudo_edit_dataset_and_fashioniq_loader() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        images = root / "images"
        images.mkdir()
        for name in ("a.png", "b.png", "c.png", "d.png"):
            Image.new("RGB", (4, 4), "red").save(images / name)
        manifest = root / "pseudo.csv"
        manifest.write_text(
            "reference,positive,cf_a,cf_b,modification,attribute_type\n"
            "a.png,b.png,c.png,d.png,make it blue,color\n",
            encoding="utf-8",
        )
        dataset = PseudoEditDataset(manifest, images, ToTensor())
        assert dataset[0]["modification"] == "make it blue"

        (root / "captions").mkdir()
        (root / "image_splits").mkdir()
        for category in ("dress", "shirt", "toptee"):
            (root / "image_splits" / f"split.{category}.val.json").write_text(
                json.dumps(["a", "b"]), encoding="utf-8"
            )
            (root / "captions" / f"cap.{category}.val.json").write_text(
                json.dumps([{"candidate": "a", "target": "b", "captions": ["blue", "long"]}]),
                encoding="utf-8",
            )
        fashion = load_fashioniq_split(root)
        assert len(fashion.queries) == 3
        assert fashion.queries[0].modification == "blue and long"
