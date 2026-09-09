"""Local validation evaluator for the CIRR benchmark."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pic2word.data.cirr import CIRRSplit
from pic2word.evaluation.metrics import recall_at_k
from pic2word.models.pic2word_model import Pic2WordModel
from pic2word.retrieval.index import CandidateIndex
from pic2word.retrieval.search import encode_composed_query


@dataclass(frozen=True, slots=True)
class CIRREvaluationResult:
    """Official CIRR global and group/subset recall percentages."""

    global_recall: dict[int, float]
    group_recall: dict[int, float]
    query_count: int


def evaluate_cirr(
    model: Pic2WordModel,
    index: CandidateIndex,
    dataset: CIRRSplit,
    *,
    global_ks: Sequence[int] = (1, 5, 10, 50),
    group_ks: Sequence[int] = (1, 2, 3),
    max_queries: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> CIRREvaluationResult:
    """Evaluate Pic2Word on CIRR while excluding each reference image."""

    if max_queries is not None and max_queries <= 0:
        raise ValueError("max_queries must be greater than zero")
    queries = dataset.queries[:max_queries]
    if not queries:
        raise ValueError("No CIRR queries selected for evaluation")

    indexed_paths = set(index.paths)
    expected_paths = {path.resolve() for path in dataset.candidate_paths}
    if indexed_paths != expected_paths:
        missing_count = len(expected_paths - indexed_paths)
        extra_count = len(indexed_paths - expected_paths)
        raise ValueError(
            "Candidate index does not match this CIRR image split "
            f"({missing_count} missing, {extra_count} extra); rebuild the index"
        )

    path_to_id = {path.resolve(): image_id for image_id, path in dataset.image_paths.items()}
    global_rankings: list[list[str]] = []
    group_rankings: list[list[str]] = []
    targets: list[str] = []

    for completed, query in enumerate(queries, start=1):
        reference_path = dataset.path_for(query.reference_id)
        query_feature, _ = encode_composed_query(model, reference_path, query.caption)
        global_results = index.search(
            query_feature,
            top_k=max(global_ks),
            exclude_paths={reference_path},
        )
        group_results = index.search(
            query_feature,
            top_k=max(group_ks),
            exclude_paths={reference_path},
            allowed_paths={dataset.path_for(image_id) for image_id in query.group_members},
        )
        global_rankings.append([path_to_id[result.path] for result in global_results])
        group_rankings.append([path_to_id[result.path] for result in group_results])
        targets.append(query.target_id)
        if progress is not None:
            progress(completed, len(queries))

    return CIRREvaluationResult(
        global_recall=recall_at_k(global_rankings, targets, global_ks),
        group_recall=recall_at_k(group_rankings, targets, group_ks),
        query_count=len(queries),
    )
