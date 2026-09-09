"""Compose an image and modification text into a Pic2Word retrieval query."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor

from pic2word.models.pic2word_model import Pic2WordModel
from pic2word.retrieval.index import CandidateIndex, RetrievalResult
from pic2word.retrieval.prompts import build_sentence_prompt


def encode_composed_query(
    model: Pic2WordModel,
    query_image: str | Path,
    modification: str,
) -> tuple[Tensor, str]:
    """Create one normalized query feature from an image and modification sentence."""

    image_path = Path(query_image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Query image not found: {image_path}")
    prompt = build_sentence_prompt(modification)
    model.eval()
    with Image.open(image_path) as image:
        image_tensor = model.backbone.preprocess(image.convert("RGB")).unsqueeze(0)
    model_device = next(model.mapping_network.parameters()).device
    image_tensor = image_tensor.to(model_device)
    with torch.no_grad():
        output = model(image_tensor, [prompt])
        query_feature = torch.nn.functional.normalize(output.text_features, dim=-1)
    return query_feature, prompt


def search_pic2word(
    model: Pic2WordModel,
    index: CandidateIndex,
    query_image: str | Path,
    modification: str,
    *,
    top_k: int = 5,
    exclude_query: bool = True,
) -> tuple[list[RetrievalResult], str]:
    """Return Top-K candidates for a composed image-text query."""

    query_path = Path(query_image).resolve()
    query_feature, prompt = encode_composed_query(model, query_path, modification)
    excluded = {query_path} if exclude_query else set()
    return index.search(query_feature, top_k=top_k, exclude_paths=excluded), prompt
