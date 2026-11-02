"""Multi-Document Multi-Strategy Retrieval Evaluation for LocalRAG Phase 4.8.

Evaluates 7 retrieval strategies across the 5-document corpus (102 chunks):
1. Dense Retrieval (nomic-embed-text)
2. BM25 Retrieval
3. Hybrid Reciprocal Rank Fusion (Dense + BM25)
4. BM25 + Cross-Encoder Reranking
5. Hybrid RRF + Cross-Encoder Reranking
6. BM25 + Cross-Encoder with Score Threshold
7. Hybrid RRF + Cross-Encoder with Score Threshold

Evaluation Rules:
- 155-query multi-document benchmark.
- Stratified 50/50 split into Calibration (~79) and Held-out Test (~76) sets.
- Score threshold tuned ONLY on Calibration set.
- All final metrics reported on untouched Held-out Test set.
- Metrics: MRR, R@1, R@3, R@4, P@4, Unsupported Rejection, Hard-Negative Rejection, FRR, FAR, Latency (Mean, P50, P95).
- Output saved to data/evaluation/multidoc_results.json.
"""
from __future__ import annotations

import json
import logging
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config import settings
from src.embeddings.ollama import OllamaClient
from src.generation.contextualize import contextualize_query
from src.retrieval.bm25_search import BM25Index
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import CrossEncoderReranker, _sigmoid
from src.retrieval.search import search as dense_search
from src.storage.vector_index import VectorIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multidoc_eval")


def compute_ir_metrics(
    retrieved_cids: list[str],
    expected_cids: set[str],
    is_relevant: bool,
    actually_rejected: bool,
    k_values: list[int] = [1, 2, 3, 4],
) -> dict:
    if not is_relevant:
        return {
            "mrr": None,
            "p_at_k": {k: 0.0 if retrieved_cids[:k] else 1.0 for k in k_values},
            "r_at_k": {k: 1.0 for k in k_values},
            "is_correctly_rejected": actually_rejected,
        }

    rr = 0.0
    for rank_idx, cid in enumerate(retrieved_cids, start=1):
        if cid in expected_cids:
            rr = 1.0 / rank_idx
            break

    p_at_k = {}
    r_at_k = {}
    for k in k_values:
        top_k_set = set(retrieved_cids[:k])
        hits = len(top_k_set & expected_cids)
        p_at_k[k] = hits / k
        r_at_k[k] = hits / len(expected_cids) if expected_cids else 0.0

    return {
        "mrr": rr,
        "p_at_k": p_at_k,
        "r_at_k": r_at_k,
        "is_correctly_rejected": False,
    }


def aggregate_pipeline_metrics(results: list[dict], latencies: list[float] | None = None) -> dict:
    relevant_items = [r for r in results if r["is_relevant"]]
    unsupported_items = [r for r in results if r["category"] == "Unsupported"]
    hard_negative_items = [r for r in results if r["category"] == "Hard Negative"]
    negative_items = [r for r in results if not r["is_relevant"]]

    valid_mrrs = [r["metrics"]["mrr"] for r in relevant_items if r["metrics"]["mrr"] is not None]
    mean_mrr = statistics.mean(valid_mrrs) if valid_mrrs else 0.0

    p_at_k = {}
    r_at_k = {}
    for k in [1, 2, 3, 4]:
        p_vals = [r["metrics"]["p_at_k"][k] for r in relevant_items]
        r_vals = [r["metrics"]["r_at_k"][k] for r in relevant_items]
        p_at_k[k] = statistics.mean(p_vals) if p_vals else 0.0
        r_at_k[k] = statistics.mean(r_vals) if r_vals else 0.0

    unsupported_rejections = sum(1 for r in unsupported_items if r["actually_rejected"])
    unsupported_rate = unsupported_rejections / len(unsupported_items) if unsupported_items else 0.0

    hard_negative_rejections = sum(1 for r in hard_negative_items if r["actually_rejected"])
    hard_negative_rate = hard_negative_rejections / len(hard_negative_items) if hard_negative_items else 0.0

    # False Rejection Rate (FRR) on relevant queries
    false_rejections = sum(1 for r in relevant_items if r["actually_rejected"])
    frr = false_rejections / len(relevant_items) if relevant_items else 0.0

    # False Acceptance Rate (FAR) on negative queries
    false_acceptances = sum(1 for r in negative_items if not r["actually_rejected"])
    far = false_acceptances / len(negative_items) if negative_items else 0.0

    latency_stats = {}
    if latencies:
        sorted_lats = sorted(latencies)
        latency_stats = {
            "mean_ms": round(statistics.mean(latencies), 2),
            "p50_ms": round(statistics.median(latencies), 2),
            "p95_ms": round(float(np.percentile(sorted_lats, 95)), 2),
        }

    return {
        "total_queries": len(results),
        "relevant_queries": len(relevant_items),
        "negative_queries": len(negative_items),
        "mrr": mean_mrr,
        "p_at_1": p_at_k[1],
        "p_at_2": p_at_k[2],
        "p_at_3": p_at_k[3],
        "p_at_4": p_at_k[4],
        "r_at_1": r_at_k[1],
        "r_at_2": r_at_k[2],
        "r_at_3": r_at_k[3],
        "r_at_4": r_at_k[4],
        "unsupported_rejections": unsupported_rejections,
        "unsupported_total": len(unsupported_items),
        "unsupported_rate": unsupported_rate,
        "hard_negative_rejections": hard_negative_rejections,
        "hard_negative_total": len(hard_negative_items),
        "hard_negative_rate": hard_negative_rate,
        "false_rejections": false_rejections,
        "frr": frr,
        "false_acceptances": false_acceptances,
        "far": far,
        "latency": latency_stats,
    }


def split_benchmark(benchmark: list[dict]) -> tuple[list[dict], list[dict]]:
    by_cat = defaultdict(list)
    for q in benchmark:
        by_cat[q["category"]].append(q)

    calibration_set = []
    test_set = []

    for cat, items in sorted(by_cat.items()):
        sorted_items = sorted(items, key=lambda x: x["question_id"])
        for i, item in enumerate(sorted_items):
            if i % 2 == 0:
                calibration_set.append(item)
            else:
                test_set.append(item)

    return calibration_set, test_set


def run_pipeline_on_dataset(
    dataset: list[dict],
    pipeline_name: str,
    index: VectorIndex,
    bm25_idx: BM25Index,
    reranker: CrossEncoderReranker,
    client: OllamaClient,
    score_threshold: float | None = None,
    precomputed_embeddings: dict[str, list[float]] | None = None,
) -> tuple[dict, list[dict], list[dict], list[float]]:
    results = []
    raw_records = []
    latencies = []
    top_k = settings.top_k  # 4
    first_stage_depth = 10
    dense_threshold = settings.similarity_threshold  # 0.55
    bm25_min_score = 3.0
    rrf_k = 60

    for item in dataset:
        qid = item["question_id"]
        category = item["category"]
        question = item["question"]
        expected_cids = set(item["relevant_chunk_ids"])
        should_reject = item["should_reject"]
        is_relevant = len(expected_cids) > 0 and not should_reject

        search_query = question
        if category == "Conversational follow-ups" and item.get("history"):
            search_query = contextualize_query(question, item["history"], client, settings.chat_model_default)

        # In Document-specific, only filter if category is Document-specific
        doc_scope = item.get("relevant_doc_id") if category == "Document-specific" else None

        t0 = time.perf_counter()

        q_vec = None
        if pipeline_name in ("dense", "hybrid", "hybrid_rerank"):
            if precomputed_embeddings and search_query in precomputed_embeddings:
                q_vec = precomputed_embeddings[search_query]
            else:
                q_vec = client.embed(search_query, settings.embedding_model)
                if precomputed_embeddings is not None:
                    precomputed_embeddings[search_query] = q_vec

        retrieved_cids = []
        rejected = False
        candidates_info = []

        if pipeline_name == "dense":
            try:
                ret = dense_search(index, q_vec, top_k=top_k, threshold=dense_threshold, doc_id=doc_scope)
                retrieved_cids = [c.chunk_id for c, _ in ret]
            except ValueError:
                pass
            rejected = len(retrieved_cids) == 0

        elif pipeline_name == "bm25":
            try:
                ret = bm25_idx.search(search_query, top_k=top_k, min_score=bm25_min_score, doc_id=doc_scope)
                retrieved_cids = [c.chunk_id for c, _ in ret]
            except ValueError:
                pass
            rejected = len(retrieved_cids) == 0

        elif pipeline_name == "hybrid":
            try:
                ret, _ = hybrid_search(
                    index=index,
                    query=search_query,
                    query_vector=q_vec,
                    bm25_index=bm25_idx,
                    top_k=top_k,
                    dense_threshold=dense_threshold,
                    bm25_min_score=bm25_min_score,
                    doc_id=doc_scope,
                    rrf_k=rrf_k,
                )
                retrieved_cids = [c.chunk_id for c, _ in ret]
            except ValueError:
                pass
            rejected = len(retrieved_cids) == 0

        elif pipeline_name == "bm25_rerank":
            bm25_cands = []
            try:
                bm25_cands = bm25_idx.search(search_query, top_k=first_stage_depth, min_score=bm25_min_score, doc_id=doc_scope)
            except ValueError:
                pass
            rep = reranker.rerank_with_report(
                query=search_query,
                candidates=bm25_cands,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            retrieved_cids = [c.chunk_id for c, _ in rep.results]
            rejected = len(retrieved_cids) == 0
            candidates_info = [
                {
                    "chunk_id": d.chunk_id,
                    "original_rank": d.original_rank,
                    "original_score": d.original_score,
                    "rerank_score": d.rerank_score,
                    "sigmoid_score": d.sigmoid_score,
                }
                for d in rep.details
            ]

        elif pipeline_name == "hybrid_rerank":
            hybrid_cands = []
            try:
                hybrid_cands, _ = hybrid_search(
                    index=index,
                    query=search_query,
                    query_vector=q_vec,
                    bm25_index=bm25_idx,
                    top_k=first_stage_depth,
                    dense_threshold=dense_threshold,
                    bm25_min_score=bm25_min_score,
                    doc_id=doc_scope,
                    rrf_k=rrf_k,
                )
            except ValueError:
                pass
            rep = reranker.rerank_with_report(
                query=search_query,
                candidates=hybrid_cands,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            retrieved_cids = [c.chunk_id for c, _ in rep.results]
            rejected = len(retrieved_cids) == 0
            candidates_info = [
                {
                    "chunk_id": d.chunk_id,
                    "original_rank": d.original_rank,
                    "original_score": d.original_score,
                    "rerank_score": d.rerank_score,
                    "sigmoid_score": d.sigmoid_score,
                }
                for d in rep.details
            ]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        metrics = compute_ir_metrics(retrieved_cids, expected_cids, is_relevant, rejected)
        results.append({
            "question_id": qid,
            "category": category,
            "is_relevant": is_relevant,
            "should_reject": should_reject,
            "retrieved_cids": retrieved_cids,
            "actually_rejected": rejected,
            "metrics": metrics,
            "latency_ms": round(latency_ms, 2),
        })
        raw_records.append({
            "question_id": qid,
            "category": category,
            "is_relevant": is_relevant,
            "should_reject": should_reject,
            "expected_cids": list(expected_cids),
            "candidates": candidates_info,
        })

    agg = aggregate_pipeline_metrics(results, latencies)
    return agg, results, raw_records, latencies


def main():
    print("=" * 85)
    print("LOCALRAG PHASE 4.8 — MULTI-DOCUMENT MULTI-STRATEGY RETRIEVAL EVALUATION")
    print("=" * 85)

    benchmark_path = PROJECT_ROOT / "data" / "evaluation" / "multidoc_benchmark.json"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(benchmark)} total benchmark questions.")

    calibration_set, test_set = split_benchmark(benchmark)
    print(f"Calibration Set: {len(calibration_set)} queries | Held-Out Test Set: {len(test_set)} queries")

    from collections import Counter
    cal_counts = Counter(x["category"] for x in calibration_set)
    test_counts = Counter(x["category"] for x in test_set)
    print("\nStratification Breakdown:")
    print(f"{'Category':<28} | {'Calibration':<12} | {'Held-Out Test':<14} | {'Total':<6}")
    print("-" * 65)
    for cat in sorted(cal_counts.keys() | test_counts.keys()):
        print(f"{cat:<28} | {cal_counts[cat]:<12} | {test_counts[cat]:<14} | {cal_counts[cat]+test_counts[cat]:<6}")

    # Load resources
    index = VectorIndex.load(PROJECT_ROOT / "data" / "index", settings.embedding_model, settings.chunk_size, settings.chunk_overlap)
    dim = index.vectors.shape[1] if index.vectors.ndim == 2 else 0
    print(f"\nLoaded VectorIndex: {len(index.chunks)} chunks, dimension {dim}")
    bm25_idx = BM25Index(index.chunks, k1=1.5, b=0.75)
    reranker = CrossEncoderReranker(device="cpu", lazy_load=False)
    client = OllamaClient(settings.ollama_base_url, timeout=60.0)

    # Embedding cache for benchmark queries to ensure fair latency comparison
    embedding_cache: dict[str, list[float]] = {}
    print("\nPre-computing query embeddings for benchmark...")
    for item in benchmark:
        q = item["question"]
        if item["category"] == "Conversational follow-ups" and item.get("history"):
            q = contextualize_query(q, item["history"], client, settings.chat_model_default)
        if q not in embedding_cache:
            embedding_cache[q] = client.embed(q, settings.embedding_model)
    print(f"Cached {len(embedding_cache)} query embeddings.")

    # =========================================================================
    # STEP 1: CALIBRATION SWEEP (Tune threshold tau on Calibration Set ONLY)
    # =========================================================================
    print("\n" + "=" * 85)
    print("STEP 1: THRESHOLD CALIBRATION SWEEP (Calibration Set ONLY)")
    print("=" * 85)

    _, _, cal_raw_records, _ = run_pipeline_on_dataset(
        calibration_set,
        "hybrid_rerank",
        index,
        bm25_idx,
        reranker,
        client,
        score_threshold=None,
        precomputed_embeddings=embedding_cache,
    )

    threshold_candidates = [-6.0, -5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 1.0, 2.0]
    calibration_sweep = []

    for thresh in threshold_candidates:
        sim_results = []
        for rec in cal_raw_records:
            is_relevant = rec["is_relevant"]
            expected_cids = set(rec["expected_cids"])
            surviving = [c for c in rec["candidates"] if c["rerank_score"] >= thresh]
            top4 = surviving[:4]
            cids = [c["chunk_id"] for c in top4]
            rejected = len(cids) == 0
            metrics = compute_ir_metrics(cids, expected_cids, is_relevant, rejected)
            sim_results.append({
                "question_id": rec["question_id"],
                "category": rec["category"],
                "is_relevant": is_relevant,
                "should_reject": rec["should_reject"],
                "retrieved_cids": cids,
                "actually_rejected": rejected,
                "metrics": metrics,
            })
        agg = aggregate_pipeline_metrics(sim_results)
        agg["threshold"] = thresh
        agg["sigmoid"] = _sigmoid(thresh)
        calibration_sweep.append(agg)

    print(f"{'Threshold':<10} | {'Sigmoid':<8} | {'MRR':<7} | {'R@4':<7} | {'Unsupp Rej':<11} | {'Hard Neg Rej':<13} | {'FRR':<7} | {'FAR':<7}")
    print("-" * 85)
    for s in calibration_sweep:
        print(
            f"{s['threshold']:<10.1f} | "
            f"{s['sigmoid']:<8.4f} | "
            f"{s['mrr']:<7.4f} | "
            f"{s['r_at_4']:<7.2%} | "
            f"{s['unsupported_rate']:<11.1%} | "
            f"{s['hard_negative_rate']:<13.1%} | "
            f"{s['frr']:<7.2%} | "
            f"{s['far']:<7.2%}"
        )

    # Threshold Selection:
    # Rule: FRR <= 6.0% on relevant queries, maximize (HardNegRej + UnsuppRej, R@4)
    admissible = [s for s in calibration_sweep if s["frr"] <= 0.06]
    selected_sweep = max(
        admissible,
        key=lambda s: (s["hard_negative_rate"] + s["unsupported_rate"], s["r_at_4"]),
    )
    selected_threshold = selected_sweep["threshold"]
    print(f"\n--> Selected Calibration Threshold: tau = {selected_threshold:.1f} (Sigmoid = {selected_sweep['sigmoid']:.4f})")
    print(f"    Calibration FRR: {selected_sweep['frr']:.2%} ({selected_sweep['false_rejections']}/{selected_sweep['relevant_queries']} false rejections)")
    print(f"    Calibration FAR: {selected_sweep['far']:.2%} ({selected_sweep['false_acceptances']}/{selected_sweep['negative_queries']} false acceptances)")
    print(f"    Calibration Unsupported Rejection: {selected_sweep['unsupported_rate']:.1%}")
    print(f"    Calibration Hard Negative Rejection: {selected_sweep['hard_negative_rate']:.1%}")

    # =========================================================================
    # STEP 2: EVALUATION ON UNTOUCHED HELD-OUT TEST SET (76 queries)
    # =========================================================================
    print("\n" + "=" * 85)
    print(f"STEP 2: EVALUATION OF 7 STRATEGIES ON HELD-OUT TEST SET ({len(test_set)} queries)")
    print("=" * 85)

    strategies = [
        ("1. Dense (nomic-embed-text)", "dense", None),
        ("2. BM25", "bm25", None),
        ("3. Hybrid RRF", "hybrid", None),
        ("4. BM25 + Cross-Encoder", "bm25_rerank", None),
        ("5. Hybrid RRF + Cross-Encoder", "hybrid_rerank", None),
        (f"6. BM25 + CE (tau={selected_threshold:.1f})", "bm25_rerank", selected_threshold),
        (f"7. Hybrid + CE (tau={selected_threshold:.1f})", "hybrid_rerank", selected_threshold),
    ]

    strategy_results = {}
    strategy_details = {}

    print(f"{'STRATEGY':<35} | {'MRR':<6} | {'R@1':<6} | {'R@3':<6} | {'R@4':<6} | {'P@4':<6} | {'FRR':<6} | {'FAR':<6} | {'UNSUPP':<7} | {'HARD NEG':<8} | {'LAT(p50)':<8}")
    print("-" * 115)

    for name, pipe, thresh in strategies:
        agg, det, _, _ = run_pipeline_on_dataset(
            test_set,
            pipe,
            index,
            bm25_idx,
            reranker,
            client,
            score_threshold=thresh,
            precomputed_embeddings=embedding_cache,
        )
        strategy_results[name] = agg
        strategy_details[name] = det

        print(
            f"{name:<35} | "
            f"{agg['mrr']:<6.4f} | "
            f"{agg['r_at_1']:<6.2%} | "
            f"{agg['r_at_3']:<6.2%} | "
            f"{agg['r_at_4']:<6.2%} | "
            f"{agg['p_at_4']:<6.2%} | "
            f"{agg['frr']:<6.2%} | "
            f"{agg['far']:<6.2%} | "
            f"{agg['unsupported_rate']:<7.1%} | "
            f"{agg['hard_negative_rate']:<8.1%} | "
            f"{agg['latency']['p50_ms']:<8.1f} ms"
        )
    print("=" * 115)

    # Output detailed report and serialize results
    output_payload = {
        "dataset_split": {
            "total_benchmark": len(benchmark),
            "calibration_size": len(calibration_set),
            "test_size": len(test_set),
            "stratification": {cat: {"cal": cal_counts[cat], "test": test_counts[cat]} for cat in cal_counts},
        },
        "threshold_calibration": {
            "method": "Constrained Pareto (FRR <= 6.0%, Maximize HardNegRej + UnsuppRej)",
            "selected_threshold": selected_threshold,
            "selected_sigmoid": selected_sweep["sigmoid"],
            "sweep": calibration_sweep,
        },
        "held_out_test_results": strategy_results,
    }

    out_file = PROJECT_ROOT / "data" / "evaluation" / "multidoc_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"\nSuccessfully serialized multi-document evaluation results to: {out_file}")


if __name__ == "__main__":
    main()
