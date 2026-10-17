"""B1 vs B4-new comprehensive comparison on 4,432 clean validation samples.

Evaluates:
1. Counterfactual Triplet Ranking ({positive, cf_a, cf_b}):
   - Target rank (1, 2, 3), mean & median target rank
   - Query count improved / worsened / same
   - Top-1 (both-win percentage)
   - Bootstrap 95% CI for Delta R@1
2. Clean Gallery Retrieval (4,432 queries against all unique positive targets):
   - Target rank (1 to N_gallery), mean & median target rank
   - Query count improved / worsened / same
   - Top-1 and Top-5 retrieval recall (%)
   - Bootstrap 95% CI for Delta R@1 and Delta R@5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))




def load_model(ckpt_path: Path, backbone, device: torch.device):
    from pic2word.models import CFPECIRModel
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]["model"]
    model = CFPECIRModel(
        backbone.image_embedding_dim,
        variant=ckpt["variant"],
        slots=int(cfg.get("slots", 8)),
        hidden_dim=int(cfg.get("hidden_dim", 512)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


from pic2word.evaluation.metrics import cluster_bootstrap_ci, naive_bootstrap_ci


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Path to clean_val.csv")
    parser.add_argument("--image-root", type=Path, required=True, help="Root folder for CC3M images")
    parser.add_argument("--b1-checkpoint", type=Path, required=True)
    parser.add_argument("--b4-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/b1_vs_b4_clean_val_comparison"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-boot", type=int, default=10000, help="Number of bootstrap iterations (default: 10,000)")
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "comparison.log", encoding="utf-8"),
        ],
    )
    log = logging.getLogger("b1_vs_b4")

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    log.info("Using device: %s", device)

    # 1. Read manifest
    with args.manifest.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    total_queries = len(rows)
    log.info("Loaded %d validation queries from %s", total_queries, args.manifest)

    # 2. Collect unique images
    unique_images = sorted(list(
        set(r["reference"] for r in rows) |
        set(r["positive"] for r in rows) |
        set(r["cf_a"] for r in rows) |
        set(r["cf_b"] for r in rows)
    ))
    log.info("Total unique images to encode: %d", len(unique_images))

    # 3. Load Frozen CLIP Backbone
    from pic2word.models import FrozenCLIPBackbone
    log.info("Loading Frozen CLIP ViT-L-14 backbone...")
    backbone = FrozenCLIPBackbone.from_pretrained(
        model_name="ViT-L-14-quickgelu", pretrained="openai", cache_dir="checkpoints/clip", device=device
    )

    # 4. Pre-encode unique images
    log.info("Encoding unique images in batches of %d...", args.batch_size)
    img_embs: dict[str, torch.Tensor] = {}
    batch_paths: list[str] = []
    batch_tensors: list[torch.Tensor] = []

    def flush_image_batch():
        nonlocal batch_paths, batch_tensors
        if not batch_tensors:
            return
        stack = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            embs = backbone.encode_image(stack, normalize=True).cpu()
        for p, emb in zip(batch_paths, embs):
            img_embs[p] = emb
        batch_paths = []
        batch_tensors = []

    start_t = time.monotonic()
    for idx, rel_path in enumerate(unique_images, 1):
        full_p = args.image_root / rel_path
        if not full_p.is_file():
            raise FileNotFoundError(f"Missing image file: {full_p}")
        with Image.open(full_p) as img:
            batch_tensors.append(backbone.preprocess(img.convert("RGB")))
            batch_paths.append(rel_path)
        if len(batch_tensors) >= args.batch_size:
            flush_image_batch()
            if idx % 500 == 0 or idx == len(unique_images):
                log.info("  Encoded images: %d / %d", idx, len(unique_images))
    flush_image_batch()
    log.info("Finished encoding images in %.1fs", time.monotonic() - start_t)

    # 5. Pre-encode modification texts
    modifications = [r["modification"] for r in rows]
    log.info("Encoding %d modification texts...", len(modifications))
    text_embs: list[torch.Tensor] = []
    for i in range(0, len(modifications), args.batch_size):
        chunk = modifications[i:i + args.batch_size]
        with torch.no_grad():
            embs = backbone.encode_text(chunk, normalize=True).cpu()
        text_embs.append(embs)
    all_text_embs = torch.cat(text_embs, dim=0)

    # 6. Load B1 and B4 models
    log.info("Loading B1 checkpoint: %s", args.b1_checkpoint)
    b1_model = load_model(args.b1_checkpoint, backbone, device)
    log.info("Loading B4 checkpoint: %s", args.b4_checkpoint)
    b4_model = load_model(args.b4_checkpoint, backbone, device)

    # 7. Compute query embeddings for B1 and B4
    log.info("Computing query embeddings for B1 and B4...")
    ref_embs = torch.stack([img_embs[r["reference"]] for r in rows])
    
    b1_queries_list: list[torch.Tensor] = []
    b4_queries_list: list[torch.Tensor] = []
    for i in range(0, total_queries, args.batch_size):
        r_chunk = ref_embs[i:i + args.batch_size].to(device)
        t_chunk = all_text_embs[i:i + args.batch_size].to(device)
        with torch.no_grad():
            b1_q = b1_model(r_chunk, t_chunk).query.cpu()
            b4_q = b4_model(r_chunk, t_chunk).query.cpu()
        b1_queries_list.append(b1_q)
        b4_queries_list.append(b4_q)
    b1_queries = torch.cat(b1_queries_list, dim=0)
    b4_queries = torch.cat(b4_queries_list, dim=0)

    # Normalize queries
    b1_queries = torch.nn.functional.normalize(b1_queries, dim=-1)
    b4_queries = torch.nn.functional.normalize(b4_queries, dim=-1)

    # Candidate embeddings for each query
    pos_embs = torch.stack([img_embs[r["positive"]] for r in rows])
    cfa_embs = torch.stack([img_embs[r["cf_a"]] for r in rows])
    cfb_embs = torch.stack([img_embs[r["cf_b"]] for r in rows])

    # =========================================================================
    # PART 1: COUNTERFACTUAL TRIPLET RANKING ({pos, cf_a, cf_b})
    # =========================================================================
    log.info("Evaluating Part 1: Counterfactual Triplet Ranking...")
    b1_s_pos = (b1_queries * pos_embs).sum(dim=-1).numpy()
    b1_s_cfa = (b1_queries * cfa_embs).sum(dim=-1).numpy()
    b1_s_cfb = (b1_queries * cfb_embs).sum(dim=-1).numpy()

    b4_s_pos = (b4_queries * pos_embs).sum(dim=-1).numpy()
    b4_s_cfa = (b4_queries * cfa_embs).sum(dim=-1).numpy()
    b4_s_cfb = (b4_queries * cfb_embs).sum(dim=-1).numpy()

    # Discrete target rank (1 = beat both, 2 = beat one, 3 = beat neither)
    b1_triplet_rank = 1 + (b1_s_cfa >= b1_s_pos).astype(int) + (b1_s_cfb >= b1_s_pos).astype(int)
    b4_triplet_rank = 1 + (b4_s_cfa >= b4_s_pos).astype(int) + (b4_s_cfb >= b4_s_pos).astype(int)

    triplet_delta = b1_triplet_rank - b4_triplet_rank  # positive = B4 improved (lower rank number)
    triplet_improved = int(np.sum(triplet_delta > 0))
    triplet_same = int(np.sum(triplet_delta == 0))
    triplet_worsened = int(np.sum(triplet_delta < 0))

    b1_triplet_r1 = float(np.mean(b1_triplet_rank == 1) * 100.0)
    b4_triplet_r1 = float(np.mean(b4_triplet_rank == 1) * 100.0)
    
    # Reference clusters for cluster bootstrap
    ref_clusters = np.array([r["reference"] for r in rows])

    # Bootstrap CI for Delta R@1 (Both-Win: Naive vs Cluster-Robust)
    diff_triplet_r1 = (b4_triplet_rank == 1).astype(float) - (b1_triplet_rank == 1).astype(float)
    delta_r1_triplet, ci_low_triplet_naive, ci_high_triplet_naive = naive_bootstrap_ci(diff_triplet_r1, n_boot=args.n_boot)
    _, ci_low_triplet_cluster, ci_high_triplet_cluster = cluster_bootstrap_ci(diff_triplet_r1, ref_clusters, n_boot=args.n_boot)

    # Bootstrap CI for Delta CF-A Win and CF-B Win
    diff_cfa = (b4_s_cfa < b4_s_pos).astype(float) - (b1_s_cfa < b1_s_pos).astype(float)
    delta_cfa, ci_low_cfa_cluster, ci_high_cfa_cluster = cluster_bootstrap_ci(diff_cfa, ref_clusters, n_boot=args.n_boot)

    diff_cfb = (b4_s_cfb < b4_s_pos).astype(float) - (b1_s_cfb < b1_s_pos).astype(float)
    delta_cfb, ci_low_cfb_cluster, ci_high_cfb_cluster = cluster_bootstrap_ci(diff_cfb, ref_clusters, n_boot=args.n_boot)

    # =========================================================================
    # PART 2: GALLERY RETRIEVAL RANKING (all unique positive targets)
    # =========================================================================
    log.info("Evaluating Part 2: Gallery Retrieval Ranking across all unique target images...")
    unique_positives = sorted(list(set(r["positive"] for r in rows)))
    gallery_pos_to_idx = {p: i for i, p in enumerate(unique_positives)}
    gallery_embs = torch.stack([img_embs[p] for p in unique_positives])
    gallery_embs = torch.nn.functional.normalize(gallery_embs, dim=-1)

    log.info("Gallery size: %d candidate target images", len(unique_positives))

    # Cosine similarities: [4432, N_gallery]
    sims_b1 = torch.matmul(b1_queries, gallery_embs.T).numpy()
    sims_b4 = torch.matmul(b4_queries, gallery_embs.T).numpy()

    gt_indices = np.array([gallery_pos_to_idx[r["positive"]] for r in rows])

    # Calculate target ranks in gallery (1-indexed)
    b1_gallery_ranks = np.zeros(total_queries, dtype=int)
    b4_gallery_ranks = np.zeros(total_queries, dtype=int)

    for i in range(total_queries):
        gt_idx = gt_indices[i]
        b1_gallery_ranks[i] = 1 + np.sum(sims_b1[i] > sims_b1[i, gt_idx])
        b4_gallery_ranks[i] = 1 + np.sum(sims_b4[i] > sims_b4[i, gt_idx])

    gallery_delta = b1_gallery_ranks - b4_gallery_ranks
    gallery_improved = int(np.sum(gallery_delta > 0))
    gallery_same = int(np.sum(gallery_delta == 0))
    gallery_worsened = int(np.sum(gallery_delta < 0))

    b1_gal_r1 = float(np.mean(b1_gallery_ranks <= 1) * 100.0)
    b4_gal_r1 = float(np.mean(b4_gallery_ranks <= 1) * 100.0)
    b1_gal_r5 = float(np.mean(b1_gallery_ranks <= 5) * 100.0)
    b4_gal_r5 = float(np.mean(b4_gallery_ranks <= 5) * 100.0)
    b1_gal_r10 = float(np.mean(b1_gallery_ranks <= 10) * 100.0)
    b4_gal_r10 = float(np.mean(b4_gallery_ranks <= 10) * 100.0)

    # Bootstrap CI for Gallery Delta R@1 and Delta R@5 (Naive vs Cluster-Robust)
    diff_gal_r1 = (b4_gallery_ranks <= 1).astype(float) - (b1_gallery_ranks <= 1).astype(float)
    delta_gal_r1, ci_gal_r1_low_naive, ci_gal_r1_high_naive = naive_bootstrap_ci(diff_gal_r1, n_boot=args.n_boot)
    _, ci_gal_r1_low_cluster, ci_gal_r1_high_cluster = cluster_bootstrap_ci(diff_gal_r1, ref_clusters, n_boot=args.n_boot)

    diff_gal_r5 = (b4_gallery_ranks <= 5).astype(float) - (b1_gallery_ranks <= 5).astype(float)
    delta_gal_r5, ci_gal_r5_low_naive, ci_gal_r5_high_naive = naive_bootstrap_ci(diff_gal_r5, n_boot=args.n_boot)
    _, ci_gal_r5_low_cluster, ci_gal_r5_high_cluster = cluster_bootstrap_ci(diff_gal_r5, ref_clusters, n_boot=args.n_boot)


    # =========================================================================
    # COMPILE FINAL REPORT & METRICS
    # =========================================================================
    report = {
        "dataset": "clean_val.csv",
        "total_queries": total_queries,
        "unique_reference_clusters": len(np.unique(ref_clusters)),
        "gallery_size": len(unique_positives),
        "triplet_counterfactual": {
            "b1_mean_rank": float(np.mean(b1_triplet_rank)),
            "b1_median_rank": float(np.median(b1_triplet_rank)),
            "b4_mean_rank": float(np.mean(b4_triplet_rank)),
            "b4_median_rank": float(np.median(b4_triplet_rank)),
            "b1_r1_both_win": b1_triplet_r1,
            "b4_r1_both_win": b4_triplet_r1,
            "movement": {
                "improved": triplet_improved,
                "same": triplet_same,
                "worsened": triplet_worsened,
            },
            "bootstrap_delta_r1_naive": {
                "delta": delta_r1_triplet,
                "ci_95_lower": ci_low_triplet_naive,
                "ci_95_upper": ci_high_triplet_naive,
                "ci_width": ci_high_triplet_naive - ci_low_triplet_naive,
            },
            "bootstrap_delta_r1_cluster_robust": {
                "delta": delta_r1_triplet,
                "ci_95_lower": ci_low_triplet_cluster,
                "ci_95_upper": ci_high_triplet_cluster,
                "ci_width": ci_high_triplet_cluster - ci_low_triplet_cluster,
            },
        },
        "gallery_retrieval": {
            "b1_mean_rank": float(np.mean(b1_gallery_ranks)),
            "b1_median_rank": float(np.median(b1_gallery_ranks)),
            "b4_mean_rank": float(np.mean(b4_gallery_ranks)),
            "b4_median_rank": float(np.median(b4_gallery_ranks)),
            "b1_top1": b1_gal_r1,
            "b4_top1": b4_gal_r1,
            "b1_top5": b1_gal_r5,
            "b4_top5": b4_gal_r5,
            "b1_top10": b1_gal_r10,
            "b4_top10": b4_gal_r10,
            "movement": {
                "improved": gallery_improved,
                "same": gallery_same,
                "worsened": gallery_worsened,
            },
            "bootstrap_delta_r1_naive": {
                "delta": delta_gal_r1,
                "ci_95_lower": ci_gal_r1_low_naive,
                "ci_95_upper": ci_gal_r1_high_naive,
                "ci_width": ci_gal_r1_high_naive - ci_gal_r1_low_naive,
            },
            "bootstrap_delta_r1_cluster_robust": {
                "delta": delta_gal_r1,
                "ci_95_lower": ci_gal_r1_low_cluster,
                "ci_95_upper": ci_gal_r1_high_cluster,
                "ci_width": ci_gal_r1_high_cluster - ci_gal_r1_low_cluster,
            },
            "bootstrap_delta_r5_naive": {
                "delta": delta_gal_r5,
                "ci_95_lower": ci_gal_r5_low_naive,
                "ci_95_upper": ci_gal_r5_high_naive,
                "ci_width": ci_gal_r5_high_naive - ci_gal_r5_low_naive,
            },
            "bootstrap_delta_r5_cluster_robust": {
                "delta": delta_gal_r5,
                "ci_95_lower": ci_gal_r5_low_cluster,
                "ci_95_upper": ci_gal_r5_high_cluster,
                "ci_width": ci_gal_r5_high_cluster - ci_gal_r5_low_cluster,
            },
        },
    }

    (out_dir / "comparison_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Console summary output
    print("\n" + "=" * 80)
    print("=== BÁO CÁO ĐỐI ĐẦU B1 VS B4-NEW TRÊN 4.432 MẪU CLEAN VAL ===")
    print("=" * 80)
    print(f"Tổng số query: {total_queries} | Số cụm Reference Image độc lập: {len(np.unique(ref_clusters))}")
    print("1. BÀI THI COUNTERFACTUAL TRIPLET (3 ỨNG VIÊN {pos, cf_a, cf_b}):")
    print(f"   - Target Rank B1       : Mean = {report['triplet_counterfactual']['b1_mean_rank']:.3f} | Median = {report['triplet_counterfactual']['b1_median_rank']:.1f}")
    print(f"   - Target Rank B4-new   : Mean = {report['triplet_counterfactual']['b4_mean_rank']:.3f} | Median = {report['triplet_counterfactual']['b4_median_rank']:.1f}")
    print(f"   - Số query dịch chuyển : Improved = {triplet_improved} | Same = {triplet_same} | Worsened = {triplet_worsened}")
    print(f"   - Top-1 (Both-Win)     : B1 = {b1_triplet_r1:.2f}% | B4-new = {b4_triplet_r1:.2f}%")
    print(f"   - ΔR@1 Naive CI (IID)  : {delta_r1_triplet:+.2f}% (95% CI: [{ci_low_triplet_naive:+.2f}%, {ci_high_triplet_naive:+.2f}%], Width: {ci_high_triplet_naive - ci_low_triplet_naive:.2f}%)")
    print(f"   - ΔR@1 Cluster-Robust  : {delta_r1_triplet:+.2f}% (95% CI: [{ci_low_triplet_cluster:+.2f}%, {ci_high_triplet_cluster:+.2f}%], Width: {ci_high_triplet_cluster - ci_low_triplet_cluster:.2f}%)")
    
    print("\n2. BÀI THI GALLERY RETRIEVAL (TRUY XUẤT TRÊN TOÀN BỘ 908 ẢNH TARGET):")
    print(f"   - Target Rank B1       : Mean = {report['gallery_retrieval']['b1_mean_rank']:.2f} | Median = {report['gallery_retrieval']['b1_median_rank']:.1f}")
    print(f"   - Target Rank B4-new   : Mean = {report['gallery_retrieval']['b4_mean_rank']:.2f} | Median = {report['gallery_retrieval']['b4_median_rank']:.1f}")
    print(f"   - Số query dịch chuyển : Improved = {gallery_improved} | Same = {gallery_same} | Worsened = {gallery_worsened}")
    print(f"   - Top-1 Recall (R@1)   : B1 = {b1_gal_r1:.2f}% | B4-new = {b4_gal_r1:.2f}%")
    print(f"   - Top-5 Recall (R@5)   : B1 = {b1_gal_r5:.2f}% | B4-new = {b4_gal_r5:.2f}%")
    print(f"   - Top-10 Recall (R@10) : B1 = {b1_gal_r10:.2f}% | B4-new = {b4_gal_r10:.2f}%")
    print(f"   - ΔR@1 Naive CI        : {delta_gal_r1:+.2f}% (95% CI: [{ci_gal_r1_low_naive:+.2f}%, {ci_gal_r1_high_naive:+.2f}%], Width: {ci_gal_r1_high_naive - ci_gal_r1_low_naive:.2f}%)")
    print(f"   - ΔR@1 Cluster-Robust  : {delta_gal_r1:+.2f}% (95% CI: [{ci_gal_r1_low_cluster:+.2f}%, {ci_gal_r1_high_cluster:+.2f}%], Width: {ci_gal_r1_high_cluster - ci_gal_r1_low_cluster:.2f}%)")
    print(f"   - ΔR@5 Naive CI        : {delta_gal_r5:+.2f}% (95% CI: [{ci_gal_r5_low_naive:+.2f}%, {ci_gal_r5_high_naive:+.2f}%], Width: {ci_gal_r5_high_naive - ci_gal_r5_low_naive:.2f}%)")
    print(f"   - ΔR@5 Cluster-Robust  : {delta_gal_r5:+.2f}% (95% CI: [{ci_gal_r5_low_cluster:+.2f}%, {ci_gal_r5_high_cluster:+.2f}%], Width: {ci_gal_r5_high_cluster - ci_gal_r5_low_cluster:.2f}%)")
    print("=" * 80)
    print(f"Đã lưu kết quả chi tiết vào: {out_dir / 'comparison_metrics.json'}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
