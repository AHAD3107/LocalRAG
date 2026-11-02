"""Five-way benchmark comparison: Dense vs. BM25 vs. Hybrid vs. BM25+Reranker vs. Hybrid+Reranker.

Evaluates the exact same 100-query benchmark across:
A. Existing Dense retrieval (nomic-embed-text + cosine similarity >= 0.55)
B. BM25-only sparse retrieval (Okapi BM25, min_score >= 3.0)
C. Hybrid BM25 + Dense Reciprocal Rank Fusion (RRF, k=60)
D. BM25 (top 10) + Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2 -> top 4)
E. Hybrid RRF (top 10) + Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2 -> top 4)

Outputs:
- Global metrics (MRR, P@1/2/4, R@1/2/4, Rejection rates)
- Calibrated rejection threshold analysis (logits & sigmoid cutoffs)
- Category-wise comparisons across all 8 query categories
- Latency and resource benchmarks (CPU inference, RAM footprint, load time)
- Serializes full data to data/evaluation/reranker_comparison.json
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
import psutil

from config import settings
from src.embeddings.ollama import OllamaClient
from src.generation.contextualize import contextualize_query
from src.retrieval.bm25_search import BM25Index
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import CrossEncoderReranker, _sigmoid
from src.retrieval.search import search as dense_search
from src.storage.vector_index import VectorIndex


def get_process_ram_mb() -> float:
    """Return resident set size (RSS) in megabytes."""
    process = psutil.Process()
    return process.memory_info().rss / (1024.0 * 1024.0)


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

    false_rejections = sum(1 for r in relevant_items if r["actually_rejected"])

    return {
        "total_queries": len(results),
        "relevant_queries": len(relevant_items),
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
        "false_rejections_on_relevant": false_rejections,
    }


def evaluate_reranker_thresholds(
    raw_rerank_records: list[dict],
    threshold_values: list[float],
) -> list[dict]:
    """
    Simulate different score_threshold cutoffs on the reranked candidate lists.
    """
    threshold_evals = []
    for thresh in threshold_values:
        simulated_results = []
        for rec in raw_rerank_records:
            is_relevant = rec["is_relevant"]
            expected_cids = rec["expected_cids"]
            # Filter candidates above threshold
            surviving = [
                cand for cand in rec["candidates"]
                if cand["rerank_score"] >= thresh
            ]
            top_surviving = surviving[:4]
            cids = [c["chunk_id"] for c in top_surviving]
            actually_rejected = len(cids) == 0

            metrics = compute_ir_metrics(cids, expected_cids, is_relevant, actually_rejected)
            simulated_results.append({
                "question_id": rec["question_id"],
                "category": rec["category"],
                "is_relevant": is_relevant,
                "should_reject": rec["should_reject"],
                "retrieved_cids": cids,
                "actually_rejected": actually_rejected,
                "metrics": metrics,
            })

        agg = aggregate_pipeline_metrics(simulated_results)
        agg["threshold"] = thresh
        agg["sigmoid_threshold"] = _sigmoid(thresh)
        threshold_evals.append(agg)

    return threshold_evals


def run_reranker_evaluation():
    print("=" * 80)
    print("LOCALRAG PHASE 4.6 — CROSS-ENCODER RERANKING EMPIRICAL EVALUATION")
    print("=" * 80)

    # 1. Environment and resource baselines
    initial_ram = get_process_ram_mb()
    print(f"Initial process RAM: {initial_ram:.2f} MB")

    benchmark_path = PROJECT_ROOT / "data" / "evaluation" / "retrieval_benchmark_100.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    print(f"Loaded benchmark queries: {len(benchmark)} from {benchmark_path}")

    # Load persistent VectorIndex
    index_path = PROJECT_ROOT / "data" / "index"
    index = VectorIndex.load(index_path, settings.embedding_model, settings.chunk_size, settings.chunk_overlap)
    print(f"Loaded VectorIndex with {len(index.chunks)} chunks from {index_path}")

    # Build BM25 Index
    t_bm25_0 = time.perf_counter()
    bm25_idx = BM25Index(index.chunks, k1=1.5, b=0.75)
    t_bm25_1 = time.perf_counter()
    bm25_indexing_ms = (t_bm25_1 - t_bm25_0) * 1000.0
    print(f"Built BM25Index in {bm25_indexing_ms:.2f} ms")

    # Connect Ollama Client
    client = OllamaClient(settings.ollama_base_url, timeout=60.0)
    if not client.is_available():
        raise RuntimeError(f"Ollama is not available on {settings.ollama_base_url}")
    print(f"Connected to Ollama on {settings.ollama_base_url}")

    # Initialize Reranker Model
    print("\nInitializing CrossEncoderReranker ('cross-encoder/ms-marco-MiniLM-L-6-v2')...")
    ram_before_model = get_process_ram_mb()
    t_model_0 = time.perf_counter()
    reranker = CrossEncoderReranker(device="cpu", lazy_load=False)
    t_model_1 = time.perf_counter()
    model_load_ms = (t_model_1 - t_model_0) * 1000.0
    ram_after_model = get_process_ram_mb()
    model_ram_delta = ram_after_model - ram_before_model
    print(f"Model loaded in: {model_load_ms:.2f} ms")
    print(f"Model RAM delta: {model_ram_delta:.2f} MB (Total process RAM: {ram_after_model:.2f} MB)")
    print(f"Reranker device: {reranker.device}")

    # Retrieval configuration
    top_k = settings.top_k  # 4
    first_stage_depth = 10  # candidate depth fed into cross-encoder
    dense_threshold = settings.similarity_threshold  # 0.55
    bm25_min_score = 3.0  # minimal score to prevent pure noise
    rrf_k = 60

    results_dense = []
    results_bm25 = []
    results_hybrid = []
    results_bm25_rerank = []
    results_hybrid_rerank = []

    raw_bm25_rerank_records = []
    raw_hybrid_rerank_records = []

    latencies = {
        "dense_embed": [],
        "dense_search": [],
        "dense_total": [],
        "bm25_search": [],
        "hybrid_fusion": [],
        "hybrid_total": [],
        "rerank_bm25_stage": [],
        "rerank_bm25_total": [],
        "rerank_hybrid_stage": [],
        "rerank_hybrid_total": [],
    }

    print("\nExecuting 5-way evaluation across 100 queries...")

    for item_idx, item in enumerate(benchmark, start=1):
        qid = item["question_id"]
        category = item["category"]
        question = item["question"]
        expected_cids = set(item["relevant_chunk_ids"])
        should_reject = item["should_reject"]
        is_relevant = len(expected_cids) > 0 and not should_reject

        # Contextualize if conversational
        search_query = question
        if category == "Conversational follow-ups" and item.get("history"):
            search_query = contextualize_query(question, item["history"], client, settings.chat_model_default)

        doc_scope = item.get("relevant_doc_id") if category == "Document-specific" else None

        # 1. Embed query
        t0 = time.perf_counter()
        q_vec = client.embed(search_query, settings.embedding_model)
        t1 = time.perf_counter()
        embed_ms = (t1 - t0) * 1000.0
        latencies["dense_embed"].append(embed_ms)

        # ----------------------------------------------------
        # Pipeline A: Dense Only (top 4, threshold 0.55)
        # ----------------------------------------------------
        t2 = time.perf_counter()
        dense_retrieved = []
        try:
            dense_retrieved = dense_search(
                index,
                q_vec,
                top_k=top_k,
                threshold=dense_threshold,
                doc_id=doc_scope,
            )
        except ValueError:
            pass
        t3 = time.perf_counter()
        dense_search_ms = (t3 - t2) * 1000.0
        latencies["dense_search"].append(dense_search_ms)
        latencies["dense_total"].append(embed_ms + dense_search_ms)

        dense_cids = [c.chunk_id for c, _ in dense_retrieved]
        dense_rejected = len(dense_cids) == 0
        dense_metrics = compute_ir_metrics(dense_cids, expected_cids, is_relevant, dense_rejected)
        results_dense.append({
            "question_id": qid,
            "category": category,
            "is_relevant": is_relevant,
            "should_reject": should_reject,
            "retrieved_cids": dense_cids,
            "actually_rejected": dense_rejected,
            "metrics": dense_metrics,
        })

        # ----------------------------------------------------
        # Pipeline B: BM25 Only (top 4, min_score 3.0)
        # ----------------------------------------------------
        t4 = time.perf_counter()
        bm25_retrieved = []
        try:
            bm25_retrieved = bm25_idx.search(
                query=search_query,
                top_k=top_k,
                min_score=bm25_min_score,
                doc_id=doc_scope,
            )
        except ValueError:
            pass
        t5 = time.perf_counter()
        bm25_search_ms = (t5 - t4) * 1000.0
        latencies["bm25_search"].append(bm25_search_ms)

        bm25_cids = [c.chunk_id for c, _ in bm25_retrieved]
        bm25_rejected = len(bm25_cids) == 0
        bm25_metrics = compute_ir_metrics(bm25_cids, expected_cids, is_relevant, bm25_rejected)
        results_bm25.append({
            "question_id": qid,
            "category": category,
            "is_relevant": is_relevant,
            "should_reject": should_reject,
            "retrieved_cids": bm25_cids,
            "actually_rejected": bm25_rejected,
            "metrics": bm25_metrics,
        })

        # ----------------------------------------------------
        # Pipeline C: Hybrid RRF (top 4)
        # ----------------------------------------------------
        t6 = time.perf_counter()
        hybrid_retrieved = []
        try:
            hybrid_retrieved, _ = hybrid_search(
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
        except ValueError:
            pass
        t7 = time.perf_counter()
        hybrid_search_ms = (t7 - t6) * 1000.0
        latencies["hybrid_fusion"].append(hybrid_search_ms)
        latencies["hybrid_total"].append(embed_ms + hybrid_search_ms)

        hybrid_cids = [c.chunk_id for c, _ in hybrid_retrieved]
        hybrid_rejected = len(hybrid_cids) == 0
        hybrid_metrics = compute_ir_metrics(hybrid_cids, expected_cids, is_relevant, hybrid_rejected)
        results_hybrid.append({
            "question_id": qid,
            "category": category,
            "is_relevant": is_relevant,
            "should_reject": should_reject,
            "retrieved_cids": hybrid_cids,
            "actually_rejected": hybrid_rejected,
            "metrics": hybrid_metrics,
        })

        # ----------------------------------------------------
        # First-stage Candidates for Reranker: Depth = 10
        # ----------------------------------------------------
        # Stage 1 (BM25 candidates depth 10)
        bm25_candidates_10 = []
        try:
            bm25_candidates_10 = bm25_idx.search(
                query=search_query,
                top_k=first_stage_depth,
                min_score=bm25_min_score,
                doc_id=doc_scope,
            )
        except ValueError:
            pass

        # Stage 1 (Hybrid candidates depth 10)
        hybrid_candidates_10 = []
        try:
            hybrid_candidates_10, _ = hybrid_search(
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

        # ----------------------------------------------------
        # Pipeline D: BM25 (top 10) + Cross-Encoder Reranker
        # ----------------------------------------------------
        t8 = time.perf_counter()
        bm25_rerank_report = reranker.rerank_with_report(
            query=search_query,
            candidates=bm25_candidates_10,
            top_k=top_k,
            score_threshold=None,  # unthresholded base
        )
        t9 = time.perf_counter()
        rerank_bm25_ms = (t9 - t8) * 1000.0
        latencies["rerank_bm25_stage"].append(rerank_bm25_ms)
        latencies["rerank_bm25_total"].append(bm25_search_ms + rerank_bm25_ms)

        d_cids = [c.chunk_id for c, _ in bm25_rerank_report.results]
        d_rejected = len(d_cids) == 0
        d_metrics = compute_ir_metrics(d_cids, expected_cids, is_relevant, d_rejected)
        results_bm25_rerank.append({
            "question_id": qid,
            "category": category,
            "is_relevant": is_relevant,
            "should_reject": should_reject,
            "retrieved_cids": d_cids,
            "actually_rejected": d_rejected,
            "metrics": d_metrics,
        })

        raw_bm25_rerank_records.append({
            "question_id": qid,
            "category": category,
            "is_relevant": is_relevant,
            "should_reject": should_reject,
            "expected_cids": expected_cids,
            "candidates": [
                {
                    "chunk_id": det.chunk_id,
                    "original_rank": det.original_rank,
                    "original_score": det.original_score,
                    "rerank_score": det.rerank_score,
                    "sigmoid_score": det.sigmoid_score,
                }
                for det in bm25_rerank_report.details
            ],
        })

        # ----------------------------------------------------
        # Pipeline E: Hybrid RRF (top 10) + Cross-Encoder Reranker
        # ----------------------------------------------------
        t10 = time.perf_counter()
        hybrid_rerank_report = reranker.rerank_with_report(
            query=search_query,
            candidates=hybrid_candidates_10,
            top_k=top_k,
            score_threshold=None,  # unthresholded base
        )
        t11 = time.perf_counter()
        rerank_hybrid_ms = (t11 - t10) * 1000.0
        latencies["rerank_hybrid_stage"].append(rerank_hybrid_ms)
        latencies["rerank_hybrid_total"].append(embed_ms + hybrid_search_ms + rerank_hybrid_ms)

        e_cids = [c.chunk_id for c, _ in hybrid_rerank_report.results]
        e_rejected = len(e_cids) == 0
        e_metrics = compute_ir_metrics(e_cids, expected_cids, is_relevant, e_rejected)
        results_hybrid_rerank.append({
            "question_id": qid,
            "category": category,
            "is_relevant": is_relevant,
            "should_reject": should_reject,
            "retrieved_cids": e_cids,
            "actually_rejected": e_rejected,
            "metrics": e_metrics,
        })

        raw_hybrid_rerank_records.append({
            "question_id": qid,
            "category": category,
            "is_relevant": is_relevant,
            "should_reject": should_reject,
            "expected_cids": expected_cids,
            "candidates": [
                {
                    "chunk_id": det.chunk_id,
                    "original_rank": det.original_rank,
                    "original_score": det.original_score,
                    "rerank_score": det.rerank_score,
                    "sigmoid_score": det.sigmoid_score,
                }
                for det in hybrid_rerank_report.details
            ],
        })

        if item_idx % 20 == 0 or item_idx == len(benchmark):
            print(f"Evaluated {item_idx}/{len(benchmark)} queries...")

    # Peak RAM measurement
    peak_ram = get_process_ram_mb()

    # Aggregate global metrics
    agg_dense = aggregate_pipeline_metrics(results_dense)
    agg_bm25 = aggregate_pipeline_metrics(results_bm25)
    agg_hybrid = aggregate_pipeline_metrics(results_hybrid)
    agg_bm25_rerank = aggregate_pipeline_metrics(results_bm25_rerank)
    agg_hybrid_rerank = aggregate_pipeline_metrics(results_hybrid_rerank)

    # ----------------------------------------------------
    # Calibrated Rejection Threshold Grid Search
    # ----------------------------------------------------
    threshold_grid = [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    thresh_eval_bm25_rerank = evaluate_reranker_thresholds(raw_bm25_rerank_records, threshold_grid)
    thresh_eval_hybrid_rerank = evaluate_reranker_thresholds(raw_hybrid_rerank_records, threshold_grid)

    # Find optimal threshold for BM25+Reranker and Hybrid+Reranker:
    # Maximize (hard_neg_rejection + unsupp_rejection) while keeping false_rejections <= 2
    def select_best_threshold(eval_list: list[dict]) -> dict:
        candidates = [e for e in eval_list if e["false_rejections_on_relevant"] <= 3]
        if not candidates:
            candidates = eval_list
        # Primary: harmonic mean or sum of rejections; tie-breaker: Recall@4
        return max(candidates, key=lambda x: (x["hard_negative_rate"] + x["unsupported_rate"], x["r_at_4"]))

    best_thresh_bm25 = select_best_threshold(thresh_eval_bm25_rerank)
    best_thresh_hybrid = select_best_threshold(thresh_eval_hybrid_rerank)

    # ----------------------------------------------------
    # Category Breakdown
    # ----------------------------------------------------
    categories = sorted(list({item["category"] for item in benchmark}))
    category_breakdown = {}
    for cat in categories:
        cat_d = [r for r in results_dense if r["category"] == cat]
        cat_b = [r for r in results_bm25 if r["category"] == cat]
        cat_h = [r for r in results_hybrid if r["category"] == cat]
        cat_br = [r for r in results_bm25_rerank if r["category"] == cat]
        cat_hr = [r for r in results_hybrid_rerank if r["category"] == cat]

        category_breakdown[cat] = {
            "query_count": len(cat_d),
            "dense": aggregate_pipeline_metrics(cat_d),
            "bm25": aggregate_pipeline_metrics(cat_b),
            "hybrid": aggregate_pipeline_metrics(cat_h),
            "bm25_rerank": aggregate_pipeline_metrics(cat_br),
            "hybrid_rerank": aggregate_pipeline_metrics(cat_hr),
        }

    # ----------------------------------------------------
    # Latency Summaries
    # ----------------------------------------------------
    latency_summary = {}
    for key, vals in latencies.items():
        if vals:
            latency_summary[key] = {
                "mean_ms": round(statistics.mean(vals), 2),
                "p50_ms": round(statistics.median(vals), 2),
                "p95_ms": round(np.percentile(vals, 95), 2),
            }

    # Print Formatted Results
    print("\n" + "=" * 90)
    print(f"{'PIPELINE':<30} | {'MRR':<8} | {'R@1':<8} | {'R@4':<8} | {'P@4':<8} | {'UNSUPP REJ':<11} | {'HARD NEG REJ':<12}")
    print("-" * 90)

    pipelines = [
        ("A. Dense Only", agg_dense),
        ("B. BM25 Only", agg_bm25),
        ("C. Hybrid RRF", agg_hybrid),
        ("D. BM25 + CrossEncoder (base)", agg_bm25_rerank),
        ("E. Hybrid + CrossEncoder (base)", agg_hybrid_rerank),
        (f"D*. BM25+CE (thresh={best_thresh_bm25['threshold']:.1f})", best_thresh_bm25),
        (f"E*. Hybrid+CE (thresh={best_thresh_hybrid['threshold']:.1f})", best_thresh_hybrid),
    ]

    for name, p in pipelines:
        print(
            f"{name:<30} | "
            f"{p['mrr']:<8.4f} | "
            f"{p['r_at_1']:<8.2%} | "
            f"{p['r_at_4']:<8.2%} | "
            f"{p['p_at_4']:<8.2%} | "
            f"{p['unsupported_rate']:<11.1%} | "
            f"{p['hard_negative_rate']:<12.1%}"
        )
    print("=" * 90)

    print("\nREJECTION THRESHOLD CALIBRATION GRID (BM25 + Cross-Encoder):")
    print(f"{'Logit':<7} | {'Sigmoid':<8} | {'MRR':<7} | {'R@4':<7} | {'Unsupp Rej':<11} | {'Hard Neg Rej':<13} | {'False Rejections':<16}")
    print("-" * 80)
    for g in thresh_eval_bm25_rerank:
        print(
            f"{g['threshold']:<7.1f} | "
            f"{g['sigmoid_threshold']:<8.4f} | "
            f"{g['mrr']:<7.4f} | "
            f"{g['r_at_4']:<7.2%} | "
            f"{g['unsupported_rate']:<11.1%} | "
            f"{g['hard_negative_rate']:<13.1%} | "
            f"{g['false_rejections_on_relevant']}/{g['relevant_queries']}"
        )

    print("\nREJECTION THRESHOLD CALIBRATION GRID (Hybrid + Cross-Encoder):")
    print(f"{'Logit':<7} | {'Sigmoid':<8} | {'MRR':<7} | {'R@4':<7} | {'Unsupp Rej':<11} | {'Hard Neg Rej':<13} | {'False Rejections':<16}")
    print("-" * 80)
    for g in thresh_eval_hybrid_rerank:
        print(
            f"{g['threshold']:<7.1f} | "
            f"{g['sigmoid_threshold']:<8.4f} | "
            f"{g['mrr']:<7.4f} | "
            f"{g['r_at_4']:<7.2%} | "
            f"{g['unsupported_rate']:<11.1%} | "
            f"{g['hard_negative_rate']:<13.1%} | "
            f"{g['false_rejections_on_relevant']}/{g['relevant_queries']}"
        )

    print("\nLATENCY & RESOURCE PROFILE:")
    print(f"Model Load Time: {model_load_ms:.2f} ms")
    print(f"Model RAM Footprint: {model_ram_delta:.2f} MB")
    print(f"Process RAM Peak: {peak_ram:.2f} MB")
    print(f"Dense embed mean: {latency_summary.get('dense_embed', {}).get('mean_ms')} ms")
    print(f"Dense search mean: {latency_summary.get('dense_search', {}).get('mean_ms')} ms")
    print(f"BM25 search mean: {latency_summary.get('bm25_search', {}).get('mean_ms')} ms")
    print(f"Hybrid total mean: {latency_summary.get('hybrid_total', {}).get('mean_ms')} ms")
    print(f"Reranker scoring stage mean: {latency_summary.get('rerank_bm25_stage', {}).get('mean_ms')} ms")
    print(f"BM25 + Reranker total mean: {latency_summary.get('rerank_bm25_total', {}).get('mean_ms')} ms")
    print(f"Hybrid + Reranker total mean: {latency_summary.get('rerank_hybrid_total', {}).get('mean_ms')} ms")

    # Serialize complete payload
    comparison_payload = {
        "benchmark_query_count": len(benchmark),
        "model_name": reranker.model_name,
        "device": reranker.device,
        "model_load_ms": model_load_ms,
        "model_ram_delta_mb": model_ram_delta,
        "process_peak_ram_mb": peak_ram,
        "global_metrics": {
            "dense": agg_dense,
            "bm25": agg_bm25,
            "hybrid": agg_hybrid,
            "bm25_rerank_base": agg_bm25_rerank,
            "hybrid_rerank_base": agg_hybrid_rerank,
            "bm25_rerank_calibrated": best_thresh_bm25,
            "hybrid_rerank_calibrated": best_thresh_hybrid,
        },
        "threshold_calibration": {
            "bm25_rerank": thresh_eval_bm25_rerank,
            "hybrid_rerank": thresh_eval_hybrid_rerank,
        },
        "category_breakdown": category_breakdown,
        "latency_summary": latency_summary,
    }

    out_file = PROJECT_ROOT / "data" / "evaluation" / "reranker_comparison.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")
    print(f"\nSaved complete benchmark data to: {out_file}")


if __name__ == "__main__":
    run_reranker_evaluation()
