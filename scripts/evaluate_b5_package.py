"""Comprehensive B5 verifier→retrieval package evaluation (no training).

Covers all 8 requirements:
  (1) CIRR: Stage-1 + Final Recall R@1/R@5/R@10/R@50 + Group Recall, K=50 & K=100
  (2) Fashion-IQ: Stage-1 + Final R@10/R@50 + Mean Recall, K=50 & K=100
  (3) Per-query: stage1_target_rank, final_target_rank, rank_delta;
      count improved / unchanged / worsened; mean & median rank change
  (4) Rerank-eligible analysis: queries where target in Stage-1 Top-K ->
      verifier pushed up / down / unchanged
  (5) B4 Stage-1 vs B5 Stage-1 vs B5 Final
  (6) Ablation: Preserve-only / Edit-only / Preserve+Edit / Full verifier
  (7) Top-K sensitivity: K=50 and K=100
  (8) Save stage1_score, final_score, p_P, p_E, p_V +
      checkpoint / config / seed / git commit

No architecture changes, no training, no B2/B3.

Usage:
  python scripts/evaluate_b5_package.py ^
    --b1-checkpoint checkpoints/local_final/b1_best.pt ^
    --b4-checkpoint checkpoints/local_final/b4_best.pt ^
    --b5-checkpoint checkpoints/local_final/b5_best.pt ^
    --cirr-root      data/cirr ^
    --fashioniq-root data/fashioniq ^
    --index-dir      indexes/ ^
    --run-dir        runs/b5_package ^
    --device         cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Ensure 'src' is in sys.path regardless of how script is invoked
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
from PIL import Image

from pic2word.data import load_cirr_split, load_fashioniq_split
from pic2word.evaluation.metrics import recall_at_k
from pic2word.models import CFPECIRModel, FrozenCLIPBackbone
from pic2word.retrieval import CandidateIndex, build_candidate_index


# Ablation reranking modes (Eq. 30): score = S_G + a*log(p_P) + b*log(p_E) + g*log(1-p_V)
ABLATION_MODES: dict[str, tuple[float, float, float]] = {
    "preserve_only": (1.0, 0.0, 0.0),
    "edit_only":     (0.0, 1.0, 0.0),
    "preserve_edit": (0.5, 0.5, 0.0),
    "full_verifier": (0.5, 0.5, 0.5),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _rank(ranking: list[str], target: str) -> int | None:
    """Return 1-indexed rank of target, or None if absent."""
    try:
        return ranking.index(target) + 1
    except ValueError:
        return None


# ── Model loading ─────────────────────────────────────────────────────────────

def load_checkpoint(
    path: Path, device: str
) -> tuple[CFPECIRModel, FrozenCLIPBackbone, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt["config"]["model"]
    backbone = FrozenCLIPBackbone.from_pretrained(
        model_name=cfg["backbone"],
        pretrained=cfg.get("pretrained", "openai"),
        cache_dir=cfg.get("cache_dir", "checkpoints/clip"),
        device=device,
    )
    model = CFPECIRModel(
        backbone.image_embedding_dim,
        variant=ckpt["variant"],
        slots=int(cfg.get("slots", 8)),
        hidden_dim=int(cfg.get("hidden_dim", 512)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, backbone, ckpt


# ── Core query runner ─────────────────────────────────────────────────────────

def run_query(
    model: CFPECIRModel,
    backbone: FrozenCLIPBackbone,
    index: CandidateIndex,
    reference_path: Path,
    modification: str,
    top_k: int,
    path_to_id: dict[Path, str],
    device: str,
) -> dict:
    """Run one query; return full stage-1 ranking + per-ablation final rankings
    + verifier scores for Top-K candidates.

    B1/B4 (no verifier): final_rankings == stage1_ranking for all modes.
    B5: verifier applied to Top-K; each ablation mode gets its own ranking.
    """
    with Image.open(reference_path) as img:
        img_t = backbone.preprocess(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        ref_emb = backbone.encode_image(img_t, normalize=True)
        txt_emb = backbone.encode_text([modification], normalize=True)
        out = model(ref_emb, txt_emb)
        # Stage-1: full gallery sorted by cosine similarity
        s1_items = index.search(
            out.query,
            top_k=len(index.paths),
            exclude_paths={reference_path.resolve()},
        )

    s1_ranking = [path_to_id[it.path] for it in s1_items]
    # B1 / B4 - no verifier
    if not model.uses_verifier:
        return {
            "stage1_ranking":  s1_ranking,
            "final_rankings":  {m: s1_ranking for m in ABLATION_MODES},
            "verifier_scores": {},
        }

    # B5 - apply verifier to Stage-1 Top-K, keep tail unchanged
    topk_items = s1_items[:top_k]
    tail_items = s1_items[top_k:]
    p2i = {p: i for i, p in enumerate(index.paths)}

    with torch.no_grad():
        positions = torch.tensor(
            [p2i[it.path] for it in topk_items], dtype=torch.long
        )
        cands = index.features[positions].to(device).unsqueeze(0)   # (1, K, D)
        cs    = model.verify(cands, out, txt_emb)

    s1t = torch.tensor(
        [it.score for it in topk_items], device=device, dtype=cands.dtype
    )
    p_P = cs.preserve[0]    # (K,)
    p_E = cs.edit[0]        # (K,)
    p_V = cs.violation[0]   # (K,)
    eps = torch.finfo(s1t.dtype).eps

    verifier_scores: dict[str, dict] = {
        path_to_id[it.path]: {
            "stage1_score": float(s1t[i]),
            "p_P": float(p_P[i]),
            "p_E": float(p_E[i]),
            "p_V": float(p_V[i]),
        }
        for i, it in enumerate(topk_items)
    }

    tail_ids = [path_to_id[it.path] for it in tail_items]
    final_rankings: dict[str, list[str]] = {}

    for mode, (alpha, beta, gamma) in ABLATION_MODES.items():
        fs = (
            s1t
            + alpha * p_P.clamp_min(eps).log()
            + beta  * p_E.clamp_min(eps).log()
            + gamma * (1.0 - p_V).clamp_min(eps).log()
        )
        for i, it in enumerate(topk_items):
            verifier_scores[path_to_id[it.path]][f"final_score_{mode}"] = float(fs[i])
        order = fs.argsort(descending=True).tolist()
        final_rankings[mode] = [path_to_id[topk_items[j].path] for j in order] + tail_ids

    return {
        "stage1_ranking":  s1_ranking,
        "final_rankings":  final_rankings,
        "verifier_scores": verifier_scores,
    }


# ── Rank statistics (req. 3 & 4) ─────────────────────────────────────────────

def compute_rank_stats(
    query_results: list[dict],
    targets: list[str],
    top_k: int,
) -> dict[str, dict]:
    """Per-query and aggregate rank stats for every ablation mode."""
    out: dict[str, dict] = {}
    for mode in ABLATION_MODES:
        per_query: list[dict] = []
        deltas: list[int] = []
        improved = unchanged = worsened = 0
        el_imp = el_unch = el_wors = 0

        for res, tgt in zip(query_results, targets):
            s1r = _rank(res["stage1_ranking"], tgt)
            fr  = _rank(res["final_rankings"][mode], tgt)
            d   = (s1r - fr) if (s1r is not None and fr is not None) else None

            if d is not None:
                deltas.append(d)
                if d > 0:    improved  += 1
                elif d == 0: unchanged += 1
                else:        worsened  += 1

            # rerank-eligible: target was inside Stage-1 Top-K
            eligible = s1r is not None and s1r <= top_k
            if eligible and d is not None:
                if d > 0:    el_imp  += 1
                elif d == 0: el_unch += 1
                else:        el_wors += 1

            per_query.append({
                "stage1_target_rank": s1r,
                "final_target_rank":  fr,
                "rank_delta":         d,
                "rerank_eligible":    eligible,
            })

        out[mode] = {
            "per_query":         per_query,
            "improved":          improved,
            "unchanged":         unchanged,
            "worsened":          worsened,
            "mean_rank_delta":   statistics.fmean(deltas)  if deltas else None,
            "median_rank_delta": statistics.median(deltas) if deltas else None,
            "rerank_eligible": {
                "total":    el_imp + el_unch + el_wors,
                "improved": el_imp,
                "unchanged": el_unch,
                "worsened": el_wors,
            },
        }
    return out


# ── Dataset evaluators ────────────────────────────────────────────────────────

def eval_cirr(
    model: CFPECIRModel,
    backbone: FrozenCLIPBackbone,
    index: CandidateIndex,
    dataset,
    top_k: int,
    path_to_id: dict,
    device: str,
) -> dict:
    results: list[dict] = []
    targets: list[str]  = []
    meta:    list[dict] = []

    for n, q in enumerate(dataset.queries, 1):
        ref = Path(dataset.path_for(q.reference_id))
        r = run_query(model, backbone, index, ref, q.caption, top_k, path_to_id, device)
        results.append(r)
        targets.append(q.target_id)
        meta.append({
            "pair_id": q.pair_id, "reference_id": q.reference_id,
            "target_id": q.target_id, "caption": q.caption,
            "group_members": list(q.group_members),
        })
        if n == 1 or n % 200 == 0 or n == len(dataset.queries):
            print(f"    {n}/{len(dataset.queries)}", flush=True)

    mode_metrics: dict[str, dict] = {}
    for mode in ABLATION_MODES:
        finals = [r["final_rankings"][mode] for r in results]
        s1s    = [r["stage1_ranking"]       for r in results]
        stage1_groups = []
        groups = []
        for r, q in zip(results, dataset.queries):
            gset = set(q.group_members)
            stage1_group = [x for x in r["stage1_ranking"]
                            if x in gset and x != q.reference_id]
            g = [x for x in r["final_rankings"][mode]
                 if x in gset and x != q.reference_id]
            stage1_groups.append(stage1_group[:3])
            groups.append(g[:3])
        mode_metrics[mode] = {
            "stage1_recall": recall_at_k(s1s,    targets, (1, 5, 10, 50)),
            "final_recall":  recall_at_k(finals, targets, (1, 5, 10, 50)),
            "stage1_group_recall": recall_at_k(stage1_groups, targets, (1, 2, 3)),
            "group_recall":  recall_at_k(groups, targets, (1, 2, 3)),
        }

    return {
        "mode_metrics":    mode_metrics,
        "rank_stats":      compute_rank_stats(results, targets, top_k),
        "meta":            meta,
        "verifier_scores": [r["verifier_scores"] for r in results],
    }


def eval_fashioniq(
    model: CFPECIRModel,
    backbone: FrozenCLIPBackbone,
    indexes: dict[str, CandidateIndex],
    dataset,
    top_k: int,
    id_for_path_by_cat: dict[str, dict],
    device: str,
) -> dict:
    by_cat_r: dict[str, list] = defaultdict(list)
    by_cat_t: dict[str, list] = defaultdict(list)
    all_r: list[dict] = []
    all_t: list[str]  = []
    meta:  list[dict] = []

    for n, q in enumerate(dataset.queries, 1):
        ref = Path(dataset.path_for(q.category, q.reference_id))
        r = run_query(
            model, backbone, indexes[q.category], ref,
            q.modification, top_k, id_for_path_by_cat[q.category], device,
        )
        by_cat_r[q.category].append(r)
        by_cat_t[q.category].append(q.target_id)
        all_r.append(r)
        all_t.append(q.target_id)
        meta.append({
            "category": q.category, "reference_id": q.reference_id,
            "target_id": q.target_id, "captions": q.captions,
        })
        if n == 1 or n % 200 == 0 or n == len(dataset.queries):
            print(f"    {n}/{len(dataset.queries)}", flush=True)

    mode_metrics: dict[str, dict] = {}
    for mode in ABLATION_MODES:
        per_cat: dict = {}
        s1_per_cat: dict = {}
        for cat in by_cat_r:
            finals = [r["final_rankings"][mode] for r in by_cat_r[cat]]
            s1s    = [r["stage1_ranking"]       for r in by_cat_r[cat]]
            per_cat[cat]    = recall_at_k(finals, by_cat_t[cat], (10, 50))
            s1_per_cat[cat] = recall_at_k(s1s,   by_cat_t[cat], (10, 50))
        mean_r = {
            k: sum(v[k] for v in per_cat.values()) / len(per_cat)
            for k in (10, 50)
        }
        stage1_mean_r = {
            k: sum(v[k] for v in s1_per_cat.values()) / len(s1_per_cat)
            for k in (10, 50)
        }
        mode_metrics[mode] = {
            "stage1_per_category": s1_per_cat,
            "final_per_category":  per_cat,
            "stage1_mean_recall": stage1_mean_r,
            "mean_recall":         mean_r,
        }

    return {
        "mode_metrics": mode_metrics,
        "rank_stats":   compute_rank_stats(all_r, all_t, top_k),
        "meta":         meta,
        "verifier_scores": [r["verifier_scores"] for r in all_r],
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _save_cirr_per_query(run_dir: Path, key: str, res: dict) -> None:
    rows = []
    for i, (m, vs) in enumerate(zip(res["meta"], res["verifier_scores"])):
        row = dict(m)
        # stage1_rank is identical across all modes
        row["stage1_target_rank"] = (
            res["rank_stats"]["full_verifier"]["per_query"][i]["stage1_target_rank"]
        )
        row["rerank_eligible"] = (
            res["rank_stats"]["full_verifier"]["per_query"][i]["rerank_eligible"]
        )
        for mode in ABLATION_MODES:
            pq = res["rank_stats"][mode]["per_query"][i]
            row[f"final_target_rank_{mode}"] = pq["final_target_rank"]
            row[f"rank_delta_{mode}"]        = pq["rank_delta"]
        row["verifier_top_k_scores"] = vs
        rows.append(row)
    (run_dir / f"cirr_{key}_per_query.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


def _save_fiq_per_query(run_dir: Path, key: str, res: dict) -> None:
    rows = [
        {
            **m,
            **{
                f"rank_info_{mode}": res["rank_stats"][mode]["per_query"][i]
                for mode in ABLATION_MODES
            },
            "verifier_top_k_scores": res["verifier_scores"][i],
        }
        for i, m in enumerate(res["meta"])
    ]
    (run_dir / f"fashioniq_{key}_per_query.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


def _strip_per_query(rank_stats: dict) -> dict:
    return {
        mode: {k: v for k, v in s.items() if k != "per_query"}
        for mode, s in rank_stats.items()
    }


def _print_cirr_summary(variant: str, mode_metrics: dict) -> None:
    rep = "full_verifier" if variant == "B5" else "preserve_only"
    mm  = mode_metrics[rep]
    print(f"    Stage-1 R@1/5/10/50 : {mm['stage1_recall']}")
    print(f"    Final   R@1/5/10/50 : {mm['final_recall']}")
    print(f"    Stage-1 Group R@1/2/3: {mm['stage1_group_recall']}")
    print(f"    Final   Group R@1/2/3: {mm['group_recall']}")


def _print_fiq_summary(variant: str, mode_metrics: dict) -> None:
    rep = "full_verifier" if variant == "B5" else "preserve_only"
    mm  = mode_metrics[rep]
    print(f"    Stage-1 per-cat : {mm['stage1_per_category']}")
    print(f"    Final   per-cat : {mm['final_per_category']}")
    print(f"    Stage-1 Mean Recall: {mm['stage1_mean_recall']}")
    print(f"    Final   Mean Recall: {mm['mean_recall']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b1-checkpoint", type=Path, required=True)
    ap.add_argument("--b4-checkpoint", type=Path, required=True)
    ap.add_argument("--b5-checkpoint", type=Path, required=True)
    ap.add_argument("--cirr-root",      type=Path, default=None)
    ap.add_argument("--fashioniq-root", type=Path, default=None)
    ap.add_argument("--index-dir",  type=Path, required=True)
    ap.add_argument("--run-dir",    type=Path, required=True)
    ap.add_argument("--device",     choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--index-batch-size", type=int, default=8)
    args = ap.parse_args()

    if not args.cirr_root and not args.fashioniq_root:
        ap.error("Provide at least one of --cirr-root / --fashioniq-root")

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    args.index_dir.mkdir(parents=True, exist_ok=True)

    commit = _git_commit()
    print(f"Git commit : {commit}")

    ckpt_paths = {
        "B1": args.b1_checkpoint,
        "B4": args.b4_checkpoint,
        "B5": args.b5_checkpoint,
    }
    metadata: dict = {
        "git_commit": commit,
        "git_commit_scope": "evaluation workspace; not the historical training commit",
        "device": args.device,
        "top_k_values": [50, 100],
        "ablation_modes": {
            k: {"alpha": a, "beta": b, "gamma": g}
            for k, (a, b, g) in ABLATION_MODES.items()
        },
        "checkpoints": {
            name: {"path": str(pp), "sha256": _sha256(pp)}
            for name, pp in ckpt_paths.items()
        },
    }

    # Load all checkpoints
    print("\nLoading checkpoints ...")
    models:    dict[str, CFPECIRModel]       = {}
    backbones: dict[str, FrozenCLIPBackbone] = {}
    for name, ckpt_path in ckpt_paths.items():
        print(f"  {name} <- {ckpt_path.name}")
        m, b, ckpt = load_checkpoint(ckpt_path, args.device)
        models[name]    = m
        backbones[name] = b
        metadata["checkpoints"][name]["variant"] = ckpt["variant"]
        metadata["checkpoints"][name]["config"]  = ckpt["config"]
        metadata["checkpoints"][name]["training_seed"] = ckpt["config"]["training"]["seed"]

    # CIRR
    if args.cirr_root:
        print("\n=== CIRR ===")
        ds = load_cirr_split(args.cirr_root)
        if missing := ds.missing_images():
            raise FileNotFoundError(f"CIRR: {len(missing)} images missing")

        cirr_idx_path = args.index_dir / "cirr_val.pt"
        if cirr_idx_path.is_file():
            shared_idx = CandidateIndex.load(cirr_idx_path)
            if set(shared_idx.paths) != {p.resolve() for p in ds.candidate_paths}:
                raise ValueError("Cached CIRR index does not match the validation image split")
            print(f"  Index loaded: {cirr_idx_path.name}")
        else:
            print("  Building CIRR candidate index ...")
            shared_idx = build_candidate_index(
                backbones["B5"], ds.candidate_paths,
                batch_size=args.index_batch_size,
                progress=lambda n, t: print(f"    indexed {n}/{t}") if n % 512 == 0 or n == t else None,
            )
            shared_idx.save(cirr_idx_path)
            print(f"  Saved: {cirr_idx_path.name}")

        p2id = {pp.resolve(): k for k, pp in ds.image_paths.items()}
        cirr_out: dict = {}

        for variant in ("B1", "B4", "B5"):
            for top_k in (50, 100):
                key = f"{variant}_K{top_k}"
                print(f"\n  [{key}]")
                res = eval_cirr(
                    models[variant], backbones[variant],
                    shared_idx, ds, top_k, p2id, args.device,
                )
                cirr_out[key] = {
                    "model": variant, "top_k": top_k,
                    "mode_metrics": res["mode_metrics"],
                    "rank_stats":   _strip_per_query(res["rank_stats"]),
                }
                _save_cirr_per_query(run_dir, key, res)
                _print_cirr_summary(variant, res["mode_metrics"])

        (run_dir / "cirr_metrics.json").write_text(
            json.dumps(cirr_out, indent=2), encoding="utf-8"
        )
        print("\n  Saved: cirr_metrics.json")

    # Fashion-IQ
    if args.fashioniq_root:
        print("\n=== Fashion-IQ ===")
        ds = load_fashioniq_split(args.fashioniq_root)
        if missing := ds.missing_images():
            raise FileNotFoundError(f"Fashion-IQ: {len(missing)} images missing")

        # Build shared indexes (same frozen CLIP for all variants)
        fiq_indexes: dict[str, CandidateIndex] = {}
        for cat in ds.image_paths:
            idx_path = args.index_dir / f"fashioniq_{cat}_val.pt"
            if idx_path.is_file():
                fiq_indexes[cat] = CandidateIndex.load(idx_path)
                if set(fiq_indexes[cat].paths) != {p.resolve() for p in ds.candidates(cat)}:
                    raise ValueError(f"Cached Fashion-IQ {cat} index does not match the validation split")
            else:
                print(f"  Building FashionIQ {cat} index ...")
                fiq_indexes[cat] = build_candidate_index(
                    backbones["B5"], ds.candidates(cat),
                    batch_size=args.index_batch_size,
                )
                fiq_indexes[cat].save(idx_path)

        id_for_path_by_cat = {
            cat: {pp.resolve(): k for k, pp in ds.image_paths[cat].items()}
            for cat in ds.image_paths
        }

        fiq_out: dict = {}
        for variant in ("B1", "B4", "B5"):
            for top_k in (50, 100):
                key = f"{variant}_K{top_k}"
                print(f"\n  [{key}]")
                res = eval_fashioniq(
                    models[variant], backbones[variant],
                    fiq_indexes, ds, top_k, id_for_path_by_cat, args.device,
                )
                fiq_out[key] = {
                    "model": variant, "top_k": top_k,
                    "mode_metrics": res["mode_metrics"],
                    "rank_stats":   _strip_per_query(res["rank_stats"]),
                }
                _save_fiq_per_query(run_dir, key, res)
                _print_fiq_summary(variant, res["mode_metrics"])

        (run_dir / "fashioniq_metrics.json").write_text(
            json.dumps(fiq_out, indent=2), encoding="utf-8"
        )
        print("\n  Saved: fashioniq_metrics.json")

    # Metadata
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nDone. Results -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
