"""Two-stage retrieval for Preserve/Edit models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from pic2word.models import CFPECIRModel, FrozenCLIPBackbone
from pic2word.retrieval.index import CandidateIndex


@dataclass(frozen=True, slots=True)
class CFPERetrievalResult:
    path: Path
    stage1_score: float
    final_score: float
    preserve_probability: float | None = None
    edit_probability: float | None = None
    violation_probability: float | None = None


def search_cfpe(
    model: CFPECIRModel,
    backbone: FrozenCLIPBackbone,
    index: CandidateIndex,
    reference_path: str | Path,
    modification: str,
    *,
    top_k: int = 50,
    return_k: int | None = None,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.5,
) -> list[CFPERetrievalResult]:
    """Retrieve globally, then rerank only the Stage-1 Top-K for B5."""

    if top_k <= 0 or (return_k is not None and return_k <= 0):
        raise ValueError("Ranking counts must be positive")
    requested = return_k if return_k is not None else top_k
    reference_path = Path(reference_path).resolve()
    with Image.open(reference_path) as image:
        image_tensor = backbone.preprocess(image.convert("RGB")).unsqueeze(0).to(backbone.device)
    with torch.no_grad():
        reference = backbone.encode_image(image_tensor, normalize=True)
        text = backbone.encode_text([modification], normalize=True)
        output = model(reference, text)
        stage1 = index.search(output.query, top_k=max(top_k, requested), exclude_paths={reference_path})
        if not stage1:
            return []
        if not model.uses_verifier:
            limit = min(return_k or top_k, len(stage1))
            return [CFPERetrievalResult(item.path, item.score, item.score) for item in stage1[:limit]]
        tail = stage1[top_k:]
        stage1 = stage1[:top_k]
        path_to_index = {path: i for i, path in enumerate(index.paths)}
        positions = torch.tensor([path_to_index[item.path] for item in stage1], dtype=torch.long)
        candidates = index.features[positions].to(backbone.device).unsqueeze(0)
        constraints = model.verify(candidates, output, text)
        stage1_scores = torch.tensor(
            [[item.score for item in stage1]], device=backbone.device, dtype=candidates.dtype
        )
        final = model.rerank_scores(
            stage1_scores, constraints, alpha=alpha, beta=beta, gamma=gamma
        )[0]
        order = final.argsort(descending=True).tolist()
        limit = min(return_k or top_k, len(order))
        results = []
        for position in order[:limit]:
            item = stage1[position]
            results.append(CFPERetrievalResult(
                item.path,
                item.score,
                float(final[position]),
                float(constraints.preserve[0, position]),
                float(constraints.edit[0, position]),
                float(constraints.violation[0, position]),
            ))
        results.extend(CFPERetrievalResult(item.path, item.score, item.score) for item in tail)
        return results[:requested]
