"""Three-way benchmark comparison: Dense vs. BM25 vs. Hybrid RRF.

Evaluates the exact same 100-query benchmark across:
A. Existing Dense retrieval (nomic-embed-text + cosine similarity >= 0.55)
B. BM25-only sparse retrieval (custom technical tokenizer + Okapi BM25)
C. Hybrid BM25 + Dense Reciprocal Rank Fusion (RRF, k=60)

Outputs:
- Global metrics (MRR, P@1/2/4, R@1/2/4, Rejection rates)
- Category-wise comparisons
- Latency and memory benchmarks
- Query-by-query improvement and regression tracking
- Serializes full data to data/evaluation/hybrid_comparison.json
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config import settings
from src.embeddings.ollama import OllamaClient
from src.generation.contextualize import contextualize_query
from src.retrieval.bm25_search import BM25Index
from src.retrieval.hybrid_search import hybrid_search, reciprocal_rank_fusion
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


def run_evaluation():
    # Load 100-query benchmark dataset
    benchmark_path = PROJECT_ROOT / "data" / "evaluation" / "retrieval_benchmark_100.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(benchmark)} benchmark queries from {benchmark_path}")

    # Load persistent VectorIndex
    index_path = PROJECT_ROOT / "data" / "index"
    index = VectorIndex.load(index_path, settings.embedding_model, settings.chunk_size, settings.chunk_overlap)
    print(f"Loaded VectorIndex with {len(index.chunks)} chunks from {index_path}")

    # Build and benchmark BM25Index
    t_bm25_build_0 = time.perf_counter()
    bm25_idx = BM25Index(index.chunks, k1=1.5, b=0.75)
    t_bm25_build_1 = time.perf_counter()
    bm25_indexing_time_ms = (t_bm25_build_1 - t_bm25_build_0) * 1000.0
    print(f"Built in-memory BM25 index in {bm25_indexing_time_ms:.2f} ms")

    # Connect Ollama Client
    client = OllamaClient(settings.ollama_base_url, timeout=60.0)
    if not client.is_available():
        raise RuntimeError(f"Ollama is not available on {settings.ollama_base_url}")

    top_k = settings.top_k  # 4
    dense_threshold = settings.similarity_threshold  # 0.55
    bm25_min_score = 3.0  # Calibrated threshold for BM25 to avoid single accidental stop-word false positive
    rrf_k = 60

    # Containers for results
    results_dense = []
    results_bm25 = []
    results_hybrid = []

    latencies = {
        "dense_embed": [],
        "dense_search": [],
        "dense_total": [],
        "bm25_search": [],
        "hybrid_fusion": [],
        "hybrid_total": [],
    }

    print("\nExecuting three-way evaluation over 100 queries...")

    for item in benchmark:
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
        # Method A: Dense Retrieval
        # ----------------------------------------------------
        t2 = time.perf_counter()
        dense_retrieved = []
        dense_error = None
        try:
            dense_retrieved = dense_search(
                index,
                q_vec,
                top_k=top_k,
                threshold=dense_threshold,
                doc_id=doc_scope,
            )
        except ValueError as exc:
            dense_error = str(exc)
        t3 = time.perf_counter()
        dense_search_ms = (t3 - t2) * 1000.0
        latencies["dense_search"].append(dense_search_ms)
        latencies["dense_total"].append(embed_ms + dense_search_ms)

        dense_cids = [c.chunk_id for c, score in dense_retrieved]
        dense_rejected = len(dense_cids) == 0 or (dense_error is not None)
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
        # Method B: BM25-Only Retrieval
        # ----------------------------------------------------
        t4 = time.perf_counter()
        bm25_retrieved = []
        bm25_error = None
        try:
            bm25_retrieved = bm25_idx.search(
                query=search_query,
                top_k=top_k,
                min_score=bm25_min_score,
                doc_id=doc_scope,
            )
        except ValueError as exc:
            bm25_error = str(exc)
        t5 = time.perf_counter()
        bm25_ms = (t5 - t4) * 1000.0
        latencies["bm25_search"].append(bm25_ms)

        bm25_cids = [c.chunk_id for c, score in bm25_retrieved]
        bm25_rejected = len(bm25_cids) == 0 or (bm25_error is not None)
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
        # Method C: Hybrid BM25 + Dense RRF
        # ----------------------------------------------------
        t6 = time.perf_counter()
        hybrid_retrieved = []
        hybrid_error = None
        try:
            hybrid_retrieved, hybrid_report = hybrid_search(
                index=index,
                query=search_query,
                query_vector=q_vec,
                bm25_index=bm25_idx,
                top_k=top_k,
                dense_threshold=dense_threshold,
                bm25_min_score=bm25_min_score,
                dense_candidate_k=20,
                bm25_candidate_k=20,
                rrf_k=rrf_k,
                dense_weight=1.0,
                bm25_weight=1.0,
                doc_id=doc_scope,
            )
        except ValueError as exc:
            hybrid_error = str(exc)
        t7 = time.perf_counter()
        hybrid_pure_ms = (t7 - t6) * 1000.0
        latencies["hybrid_fusion"].append(hybrid_pure_ms)
        latencies["hybrid_total"].append(embed_ms + hybrid_pure_ms)

        hybrid_cids = [c.chunk_id for c, score in hybrid_retrieved]
        hybrid_rejected = len(hybrid_cids) == 0 or (hybrid_error is not None)
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

    # Aggregation function
    def aggregate_metrics(results_list: list[dict]) -> dict:
        rel_entries = [r for r in results_list if r["is_relevant"]]
        irrel_entries = [r for r in results_list if not r["is_relevant"]]
        unsupp_entries = [r for r in results_list if r["category"] == "Unsupported"]
        hardneg_entries = [r for r in results_list if r["category"] == "Hard Negative"]

        mrr_vals = [r["metrics"]["mrr"] for r in rel_entries if r["metrics"]["mrr"] is not None]
        p1_vals = [r["metrics"]["p_at_k"][1] for r in rel_entries]
        p2_vals = [r["metrics"]["p_at_k"][2] for r in rel_entries]
        p4_vals = [r["metrics"]["p_at_k"][4] for r in rel_entries]
        r1_vals = [r["metrics"]["r_at_k"][1] for r in rel_entries]
        r2_vals = [r["metrics"]["r_at_k"][2] for r in rel_entries]
        r4_vals = [r["metrics"]["r_at_k"][4] for r in rel_entries]

        correct_rej_total = sum(1 for r in irrel_entries if r["actually_rejected"])
        correct_rej_unsupp = sum(1 for r in unsupp_entries if r["actually_rejected"])
        correct_rej_hardneg = sum(1 for r in hardneg_entries if r["actually_rejected"])

        # Category breakdowns
        categories = sorted(list({r["category"] for r in results_list}))
        cat_stats = {}
        for cat in categories:
            c_rel = [r for r in rel_entries if r["category"] == cat]
            c_irrel = [r for r in irrel_entries if r["category"] == cat]
            c_mrr = statistics.mean([r["metrics"]["mrr"] for r in c_rel]) if c_rel else None
            c_r1 = statistics.mean([r["metrics"]["r_at_k"][1] for r in c_rel]) if c_rel else None
            c_p1 = statistics.mean([r["metrics"]["p_at_k"][1] for r in c_rel]) if c_rel else None
            c_r4 = statistics.mean([r["metrics"]["r_at_k"][4] for r in c_rel]) if c_rel else None
            c_rej = (sum(1 for r in c_irrel if r["actually_rejected"]) / len(c_irrel)) if c_irrel else None

            cat_stats[cat] = {
                "count": len([r for r in results_list if r["category"] == cat]),
                "mrr": round(c_mrr, 4) if c_mrr is not None else None,
                "recall_at_1": round(c_r1, 4) if c_r1 is not None else None,
                "precision_at_1": round(c_p1, 4) if c_p1 is not None else None,
                "recall_at_4": round(c_r4, 4) if c_r4 is not None else None,
                "rejection_rate": round(c_rej, 4) if c_rej is not None else None,
            }

        return {
            "mrr": round(statistics.mean(mrr_vals), 4) if mrr_vals else 0.0,
            "precision_at_1": round(statistics.mean(p1_vals), 4) if p1_vals else 0.0,
            "precision_at_2": round(statistics.mean(p2_vals), 4) if p2_vals else 0.0,
            "precision_at_4": round(statistics.mean(p4_vals), 4) if p4_vals else 0.0,
            "recall_at_1": round(statistics.mean(r1_vals), 4) if r1_vals else 0.0,
            "recall_at_2": round(statistics.mean(r2_vals), 4) if r2_vals else 0.0,
            "recall_at_4": round(statistics.mean(r4_vals), 4) if r4_vals else 0.0,
            "overall_rejection_rate": round(correct_rej_total / len(irrel_entries), 4) if irrel_entries else 1.0,
            "unsupported_rejection_rate": round(correct_rej_unsupp / len(unsupp_entries), 4) if unsupp_entries else 1.0,
            "hard_negative_rejection_rate": round(correct_rej_hardneg / len(hardneg_entries), 4) if hardneg_entries else 1.0,
            "categories": cat_stats,
        }

    agg_dense = aggregate_metrics(results_dense)
    agg_bm25 = aggregate_metrics(results_bm25)
    agg_hybrid = aggregate_metrics(results_hybrid)

    # Latency aggregates
    lat_summary = {
        "dense_embed_ms": round(statistics.mean(latencies["dense_embed"]), 2),
        "dense_search_ms": round(statistics.mean(latencies["dense_search"]), 2),
        "dense_total_ms": round(statistics.mean(latencies["dense_total"]), 2),
        "bm25_search_ms": round(statistics.mean(latencies["bm25_search"]), 2),
        "hybrid_fusion_ms": round(statistics.mean(latencies["hybrid_fusion"]), 2),
        "hybrid_total_ms": round(statistics.mean(latencies["hybrid_total"]), 2),
    }

    # Tracking improvements and regressions relative to Dense baseline
    improved_queries = []
    regressed_queries = []
    unchanged_queries = []

    for i in range(len(benchmark)):
        qid = benchmark[i]["question_id"]
        qtext = benchmark[i]["question"]
        cat = benchmark[i]["category"]
        is_rel = benchmark[i]["should_reject"] is False

        d_m = results_dense[i]["metrics"]
        h_m = results_hybrid[i]["metrics"]

        if is_rel:
            d_rr = d_m["mrr"] or 0.0
            h_rr = h_m["mrr"] or 0.0
            d_r4 = d_m["r_at_k"][4]
            h_r4 = h_m["r_at_k"][4]

            if (h_rr > d_rr) or (h_r4 > d_r4):
                improved_queries.append({
                    "question_id": qid,
                    "question": qtext,
                    "category": cat,
                    "dense_mrr": round(d_rr, 4),
                    "hybrid_mrr": round(h_rr, 4),
                    "dense_r4": round(d_r4, 4),
                    "hybrid_r4": round(h_r4, 4),
                    "dense_retrieved": results_dense[i]["retrieved_cids"],
                    "hybrid_retrieved": results_hybrid[i]["retrieved_cids"],
                })
            elif (h_rr < d_rr) or (h_r4 < d_r4):
                regressed_queries.append({
                    "question_id": qid,
                    "question": qtext,
                    "category": cat,
                    "dense_mrr": round(d_rr, 4),
                    "hybrid_mrr": round(h_rr, 4),
                    "dense_r4": round(d_r4, 4),
                    "hybrid_r4": round(h_r4, 4),
                    "dense_retrieved": results_dense[i]["retrieved_cids"],
                    "hybrid_retrieved": results_hybrid[i]["retrieved_cids"],
                })
            else:
                unchanged_queries.append(qid)
        else:
            d_rej = results_dense[i]["actually_rejected"]
            h_rej = results_hybrid[i]["actually_rejected"]
            if not d_rej and h_rej:
                improved_queries.append({
                    "question_id": qid,
                    "question": qtext,
                    "category": cat,
                    "improvement": "False acceptance in Dense successfully rejected in Hybrid",
                })
            elif d_rej and not h_rej:
                regressed_queries.append({
                    "question_id": qid,
                    "question": qtext,
                    "category": cat,
                    "regression": "Correct rejection in Dense falsely accepted in Hybrid",
                })

    # Print Comparison Table
    print("\n" + "=" * 90)
    print("               THREE-WAY RETRIEVAL COMPARISON BENCHMARK (100 QUERIES)")
    print("=" * 90)
    print(f"{'Metric':<30} {'A. Dense (nomic)':<20} {'B. BM25-Only':<20} {'C. Hybrid (RRF)':<20}")
    print("-" * 90)
    print(f"{'Mean Reciprocal Rank (MRR)':<30} {agg_dense['mrr']:<20.4f} {agg_bm25['mrr']:<20.4f} {agg_hybrid['mrr']:<20.4f}")
    print(f"{'Recall@1':<30} {agg_dense['recall_at_1']:<20.4f} {agg_bm25['recall_at_1']:<20.4f} {agg_hybrid['recall_at_1']:<20.4f}")
    print(f"{'Precision@1':<30} {agg_dense['precision_at_1']:<20.4f} {agg_bm25['precision_at_1']:<20.4f} {agg_hybrid['precision_at_1']:<20.4f}")
    print(f"{'Recall@2':<30} {agg_dense['recall_at_2']:<20.4f} {agg_bm25['recall_at_2']:<20.4f} {agg_hybrid['recall_at_2']:<20.4f}")
    print(f"{'Precision@2':<30} {agg_dense['precision_at_2']:<20.4f} {agg_bm25['precision_at_2']:<20.4f} {agg_hybrid['precision_at_2']:<20.4f}")
    print(f"{'Recall@4':<30} {agg_dense['recall_at_4']:<20.4f} {agg_bm25['recall_at_4']:<20.4f} {agg_hybrid['recall_at_4']:<20.4f}")
    print(f"{'Precision@4':<30} {agg_dense['precision_at_4']:<20.4f} {agg_bm25['precision_at_4']:<20.4f} {agg_hybrid['precision_at_4']:<20.4f}")
    print("-" * 90)
    print(f"{'Overall Rejection Rate':<30} {agg_dense['overall_rejection_rate']*100:<19.1f}% {agg_bm25['overall_rejection_rate']*100:<19.1f}% {agg_hybrid['overall_rejection_rate']*100:<19.1f}%")
    print(f"{'Pure Unsupported Rejection':<30} {agg_dense['unsupported_rejection_rate']*100:<19.1f}% {agg_bm25['unsupported_rejection_rate']*100:<19.1f}% {agg_hybrid['unsupported_rejection_rate']*100:<19.1f}%")
    print(f"{'Hard Negative Rejection':<30} {agg_dense['hard_negative_rejection_rate']*100:<19.1f}% {agg_bm25['hard_negative_rejection_rate']*100:<19.1f}% {agg_hybrid['hard_negative_rejection_rate']*100:<19.1f}%")
    print("-" * 90)
    print(f"{'Mean Search Latency':<30} {lat_summary['dense_search_ms']:<17.2f} ms {lat_summary['bm25_search_ms']:<17.2f} ms {lat_summary['hybrid_fusion_ms']:<17.2f} ms")
    print(f"{'Total Retrieval Latency':<30} {lat_summary['dense_total_ms']:<17.2f} ms {lat_summary['bm25_search_ms']:<17.2f} ms {lat_summary['hybrid_total_ms']:<17.2f} ms")
    print("=" * 90)

    print(f"\nQUERY MOVEMENT ANALYSIS (Hybrid vs Dense Baseline):")
    print(f"  - Improved Queries:  {len(improved_queries)}")
    print(f"  - Regressed Queries: {len(regressed_queries)}")
    print(f"  - Unchanged Queries: {len(unchanged_queries)}")

    # Save to hybrid_comparison.json
    output_path = PROJECT_ROOT / "data" / "evaluation" / "hybrid_comparison.json"
    comparison_data = {
        "metadata": {
            "benchmark_query_count": len(benchmark),
            "relevant_count": 68,
            "irrelevant_count": 32,
            "rrf_k": rrf_k,
            "dense_threshold": dense_threshold,
            "bm25_min_score": bm25_min_score,
            "bm25_indexing_time_ms": round(bm25_indexing_time_ms, 2),
        },
        "summary": {
            "dense": agg_dense,
            "bm25": agg_bm25,
            "hybrid": agg_hybrid,
        },
        "latency": lat_summary,
        "improvements": {
            "improved_count": len(improved_queries),
            "regressed_count": len(regressed_queries),
            "unchanged_count": len(unchanged_queries),
            "improved_queries": improved_queries,
            "regressed_queries": regressed_queries,
        },
    }
    output_path.write_text(json.dumps(comparison_data, indent=2), encoding="utf-8")
    print(f"\nSaved full three-way comparison to {output_path}")

    return comparison_data


if __name__ == "__main__":
    run_evaluation()

