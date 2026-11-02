"""Split calibration and held-out evaluation for LocalRAG Cross-Encoder Reranker.

Implements Task 2:
- Creates a reproducible 50/50 split (Calibration vs Held-out Test) stratified by category.
- Tunes rejection threshold ONLY on the Calibration set.
- Evaluates the chosen threshold on the untouched Held-Out Test set.
- Calculates MRR, Recall@K, Precision@K, False Acceptance Rate (FAR), False Rejection Rate (FRR).
- Serializes data/evaluation/reranker_validation.json and data/evaluation/reranker_test_results.json.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config import settings
from src.embeddings.ollama import OllamaClient
from src.generation.contextualize import contextualize_query
from src.retrieval.bm25_search import BM25Index
from src.retrieval.hybrid_search import hybrid_search, reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker, _sigmoid
from src.retrieval.search import search as dense_search
from src.storage.vector_index import VectorIndex


def compute_ir_metrics(
    retrieved_cids: list[str],
    expected_cids: set[str],
    is_relevant: bool,
    actually_rejected: bool,
    k_values: list[int] = [1, 2, 4],
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


def aggregate_pipeline_metrics(results: list[dict]) -> dict:
    relevant_items = [r for r in results if r["is_relevant"]]
    unsupported_items = [r for r in results if r["category"] in ("Unsupported", "Unsupported questions")]
    hard_negative_items = [r for r in results if r["category"] in ("Hard Negative", "Hard negatives")]
    negative_items = [r for r in results if not r["is_relevant"]]

    valid_mrrs = [r["metrics"]["mrr"] for r in relevant_items if r["metrics"]["mrr"] is not None]
    mean_mrr = statistics.mean(valid_mrrs) if valid_mrrs else 0.0

    p_at_k = {}
    r_at_k = {}
    for k in [1, 2, 4]:
        p_vals = [r["metrics"]["p_at_k"][k] for r in relevant_items]
        r_vals = [r["metrics"]["r_at_k"][k] for r in relevant_items]
        p_at_k[k] = statistics.mean(p_vals) if p_vals else 0.0
        r_at_k[k] = statistics.mean(r_vals) if r_vals else 0.0

    unsupported_rejections = sum(1 for r in unsupported_items if r["actually_rejected"])
    unsupported_rate = unsupported_rejections / len(unsupported_items) if unsupported_items else 0.0

    hard_negative_rejections = sum(1 for r in hard_negative_items if r["actually_rejected"])
    hard_negative_rate = hard_negative_rejections / len(hard_negative_items) if hard_negative_items else 0.0

    # False Rejection Rate (FRR) on relevant queries:
    false_rejections = sum(1 for r in relevant_items if r["actually_rejected"])
    frr = false_rejections / len(relevant_items) if relevant_items else 0.0

    # False Acceptance Rate (FAR) on negative queries:
    # A negative query is falsely accepted if it was NOT actually rejected (i.e. passed candidates to user)
    false_acceptances = sum(1 for r in negative_items if not r["actually_rejected"])
    far = false_acceptances / len(negative_items) if negative_items else 0.0

    return {
        "total_queries": len(results),
        "relevant_queries": len(relevant_items),
        "negative_queries": len(negative_items),
        "mrr": mean_mrr,
        "p_at_1": p_at_k[1],
        "p_at_2": p_at_k[2],
        "p_at_4": p_at_k[4],
        "r_at_1": r_at_k[1],
        "r_at_2": r_at_k[2],
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
    }


def split_benchmark(benchmark: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split the 100-query benchmark into 50 calibration and 50 test queries.
    Stratified by category using alternating assignment.
    """
    from collections import defaultdict
    by_cat = defaultdict(list)
    for q in benchmark:
        by_cat[q["category"]].append(q)

    calibration_set = []
    test_set = []

    for cat, items in sorted(by_cat.items()):
        # Deterministic sort by question_id
        sorted_items = sorted(items, key=lambda x: x["question_id"])
        # Alternating split: even index -> calibration, odd index -> test
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
) -> tuple[dict, list[dict], list[dict]]:
    """
    Run a specified pipeline over a dataset.
    Returns: (aggregate_metrics, detailed_results, raw_candidate_records)
    """
    results = []
    raw_records = []
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

        doc_scope = item.get("relevant_doc_id") if category == "Document-specific" else None

        q_vec = None
        if pipeline_name in ("dense", "hybrid", "hybrid_rerank"):
            q_vec = client.embed(search_query, settings.embedding_model)

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

        metrics = compute_ir_metrics(retrieved_cids, expected_cids, is_relevant, rejected)
        results.append({
            "question_id": qid,
            "category": category,
            "is_relevant": is_relevant,
            "should_reject": should_reject,
            "retrieved_cids": retrieved_cids,
            "actually_rejected": rejected,
            "metrics": metrics,
        })
        raw_records.append({
            "question_id": qid,
            "category": category,
            "is_relevant": is_relevant,
            "should_reject": should_reject,
            "expected_cids": list(expected_cids),
            "candidates": candidates_info,
        })

    agg = aggregate_pipeline_metrics(results)
    return agg, results, raw_records


def run_split_validation():
    print("=" * 80)
    print("LOCALRAG PHASE 4.7 — SEPARATE CALIBRATION & HELD-OUT VALIDATION")
    print("=" * 80)

    benchmark_path = PROJECT_ROOT / "data" / "evaluation" / "retrieval_benchmark_100.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))

    calibration_set, test_set = split_benchmark(benchmark)
    print(f"Total benchmark queries: {len(benchmark)}")
    print(f"Calibration set size:   {len(calibration_set)}")
    print(f"Held-out test set size: {len(test_set)}")

    # Category breakdown of split
    from collections import Counter
    cal_counts = Counter(x["category"] for x in calibration_set)
    test_counts = Counter(x["category"] for x in test_set)
    print("\nSplit Stratification by Category:")
    print(f"{'Category':<28} | {'Calibration':<12} | {'Held-Out Test':<14} | {'Total':<6}")
    print("-" * 65)
    for cat in sorted(cal_counts.keys() | test_counts.keys()):
        print(f"{cat:<28} | {cal_counts[cat]:<12} | {test_counts[cat]:<14} | {cal_counts[cat]+test_counts[cat]:<6}")

    # Load resources
    index = VectorIndex.load(PROJECT_ROOT / "data" / "index", settings.embedding_model, settings.chunk_size, settings.chunk_overlap)
    bm25_idx = BM25Index(index.chunks, k1=1.5, b=0.75)
    reranker = CrossEncoderReranker(device="cpu", lazy_load=False)
    client = OllamaClient(settings.ollama_base_url, timeout=60.0)

    # =========================================================================
    # STEP 1: Threshold Tuning on CALIBRATION SET ONLY (in-memory simulation)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: CALIBRATION SWEEP (Tuning threshold on Calibration Set ONLY)")
    print("=" * 80)

    # Run inference once on calibration set to collect candidates and scores
    print("Generating hybrid candidate pool and cross-encoder scores for calibration set...")
    _, _, cal_raw_records = run_pipeline_on_dataset(
        calibration_set,
        "hybrid_rerank",
        index,
        bm25_idx,
        reranker,
        client,
        score_threshold=None,
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
    print("-" * 80)
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

    # Threshold Selection Criteria:
    # 1. False Rejection Rate (FRR) on relevant queries <= 6.0% (at most 2 false rejections out of 34 relevant)
    # 2. Maximize negative rejection (harmonic mean of unsupported_rate and hard_negative_rate)
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

    # Save calibration results
    cal_payload = {
        "dataset_split": {
            "calibration_size": len(calibration_set),
            "test_size": len(test_set),
            "stratification": {cat: {"cal": cal_counts[cat], "test": test_counts[cat]} for cat in cal_counts},
        },
        "threshold_selection_method": "Constrained Pareto: Maximize (HardNegRej + UnsuppRej) subject to FRR <= 6.0%",
        "selected_threshold": selected_threshold,
        "selected_sigmoid": selected_sweep["sigmoid"],
        "calibration_sweep": calibration_sweep,
        "selected_calibration_metrics": selected_sweep,
    }
    cal_out = PROJECT_ROOT / "data" / "evaluation" / "reranker_validation.json"
    cal_out.write_text(json.dumps(cal_payload, indent=2), encoding="utf-8")
    print(f"\nSaved calibration results to: {cal_out}")

    # =========================================================================
    # STEP 2: Evaluation on UNTOUCHED HELD-OUT TEST SET
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: FINAL EVALUATION ON UNTOUCHED HELD-OUT TEST SET (50 queries)")
    print("=" * 80)

    # Evaluate 5 pipelines on the Held-Out Test Set:
    # 1. Dense Only
    test_dense, _, _ = run_pipeline_on_dataset(test_set, "dense", index, bm25_idx, reranker, client)
    # 2. BM25 Only
    test_bm25, _, _ = run_pipeline_on_dataset(test_set, "bm25", index, bm25_idx, reranker, client)
    # 3. Hybrid RRF
    test_hybrid, _, _ = run_pipeline_on_dataset(test_set, "hybrid", index, bm25_idx, reranker, client)
    # 4. BM25 + Cross-Encoder (unthresholded)
    test_bm25_ce_base, _, _ = run_pipeline_on_dataset(test_set, "bm25_rerank", index, bm25_idx, reranker, client, score_threshold=None)
    # 5. Hybrid RRF + Cross-Encoder (unthresholded)
    test_hybrid_ce_base, _, _ = run_pipeline_on_dataset(test_set, "hybrid_rerank", index, bm25_idx, reranker, client, score_threshold=None)
    # 6. BM25 + Cross-Encoder (with frozen calibrated threshold)
    test_bm25_ce_cal, _, _ = run_pipeline_on_dataset(test_set, "bm25_rerank", index, bm25_idx, reranker, client, score_threshold=selected_threshold)
    # 7. Hybrid RRF + Cross-Encoder (with frozen calibrated threshold)
    test_hybrid_ce_cal, _, _ = run_pipeline_on_dataset(test_set, "hybrid_rerank", index, bm25_idx, reranker, client, score_threshold=selected_threshold)

    print(f"{'PIPELINE (Held-Out Test)':<32} | {'MRR':<7} | {'R@1':<7} | {'R@4':<7} | {'P@4':<7} | {'FRR':<7} | {'FAR':<7} | {'UNSUPP':<7} | {'HARD NEG':<8}")
    print("-" * 105)

    test_table = [
        ("A. Dense Baseline", test_dense),
        ("B. BM25 Baseline", test_bm25),
        ("C. Hybrid RRF Baseline", test_hybrid),
        ("D. BM25 + Cross-Encoder (base)", test_bm25_ce_base),
        ("E. Hybrid + Cross-Encoder (base)", test_hybrid_ce_base),
        (f"D*. BM25 + CE (tau={selected_threshold:.1f})", test_bm25_ce_cal),
        (f"E*. Hybrid + CE (tau={selected_threshold:.1f})", test_hybrid_ce_cal),
    ]

    for name, m in test_table:
        print(
            f"{name:<32} | "
            f"{m['mrr']:<7.4f} | "
            f"{m['r_at_1']:<7.2%} | "
            f"{m['r_at_4']:<7.2%} | "
            f"{m['p_at_4']:<7.2%} | "
            f"{m['frr']:<7.2%} | "
            f"{m['far']:<7.2%} | "
            f"{m['unsupported_rate']:<7.1%} | "
            f"{m['hard_negative_rate']:<8.1%}"
        )
    print("=" * 105)

    # Save test results
    test_payload = {
        "dataset_split": {
            "test_size": len(test_set),
            "relevant_queries": test_dense["relevant_queries"],
            "negative_queries": test_dense["negative_queries"],
        },
        "applied_threshold": selected_threshold,
        "test_results": {
            "dense_baseline": test_dense,
            "bm25_baseline": test_bm25,
            "hybrid_rrf_baseline": test_hybrid,
            "bm25_cross_encoder_base": test_bm25_ce_base,
            "hybrid_cross_encoder_base": test_hybrid_ce_base,
            "bm25_cross_encoder_calibrated": test_bm25_ce_cal,
            "hybrid_cross_encoder_calibrated": test_hybrid_ce_cal,
        },
    }
    test_out = PROJECT_ROOT / "data" / "evaluation" / "reranker_test_results.json"
    test_out.write_text(json.dumps(test_payload, indent=2), encoding="utf-8")
    print(f"\nSaved held-out test evaluation results to: {test_out}")


if __name__ == "__main__":
    run_split_validation()
