"""Accurate, non-overlapping component-wise latency benchmarking for LocalRAG.

Measures cold vs warm latency across isolated retrieval components:
1. Query Embedding (Ollama nomic-embed-text HTTP round-trip)
2. BM25 Search (sparse inverted index traversal & scoring)
3. Dense Search (NumPy cosine similarity matrix multiplication)
4. RRF Fusion (rank merging & candidate deduplication)
5. Cross-Encoder Scoring (ms-marco-MiniLM-L-6-v2 PyTorch inference on CPU)
6. Evidence Threshold Filtering (score cutoff & sorting)
7. End-to-End Pipeline (single wall-clock timer for full retrieval)

Reconciles component sums against end-to-end elapsed time.
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
from src.retrieval.bm25_search import BM25Index
from src.retrieval.hybrid_search import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker, _sigmoid
from src.retrieval.search import search as dense_search
from src.storage.vector_index import VectorIndex


def run_latency_benchmark(num_repetitions: int = 30) -> dict:
    print("=" * 80)
    print("LOCALRAG PHASE 4.7 — RETRIEVAL LATENCY BENCHMARK")
    print("=" * 80)

    # 1. Cold loading measurements
    t_start_cold_index = time.perf_counter()
    index = VectorIndex.load(
        PROJECT_ROOT / "data" / "index",
        settings.embedding_model,
        settings.chunk_size,
        settings.chunk_overlap,
    )
    t_end_cold_index = time.perf_counter()
    cold_index_load_ms = (t_end_cold_index - t_start_cold_index) * 1000.0

    t_start_cold_bm25 = time.perf_counter()
    bm25_idx = BM25Index(index.chunks, k1=1.5, b=0.75)
    t_end_cold_bm25 = time.perf_counter()
    cold_bm25_build_ms = (t_end_cold_bm25 - t_start_cold_bm25) * 1000.0

    t_start_cold_reranker = time.perf_counter()
    reranker = CrossEncoderReranker(device="cpu", lazy_load=False)
    t_end_cold_reranker = time.perf_counter()
    cold_reranker_load_ms = (t_end_cold_reranker - t_start_cold_reranker) * 1000.0

    client = OllamaClient(settings.ollama_base_url, timeout=60.0)

    # Cold first query inference
    test_query = "What are the core foundations and Q-Matrix formulation of the DINA model?"
    t_cold_embed_0 = time.perf_counter()
    q_vec_cold = client.embed(test_query, settings.embedding_model)
    t_cold_embed_1 = time.perf_counter()
    cold_embed_ms = (t_cold_embed_1 - t_cold_embed_0) * 1000.0

    cands_bm25_cold = bm25_idx.search(test_query, top_k=10, min_score=3.0)
    t_cold_rerank_0 = time.perf_counter()
    _ = reranker.score_pairs([(test_query, c.text) for c, _ in cands_bm25_cold])
    t_cold_rerank_1 = time.perf_counter()
    cold_rerank_first_inference_ms = (t_cold_rerank_1 - t_cold_rerank_0) * 1000.0

    print(f"Cold Index Load:        {cold_index_load_ms:8.2f} ms")
    print(f"Cold BM25 Build:        {cold_bm25_build_ms:8.2f} ms")
    print(f"Cold Reranker Load:     {cold_reranker_load_ms:8.2f} ms")
    print(f"Cold First Embedding:   {cold_embed_ms:8.2f} ms")
    print(f"Cold First CE Inference:{cold_rerank_first_inference_ms:8.2f} ms")

    # 2. Warm-up runs
    print(f"\nPerforming 5 warmup iterations...")
    for _ in range(5):
        qv = client.embed(test_query, settings.embedding_model)
        b_cands = bm25_idx.search(test_query, top_k=10, min_score=3.0)
        d_cands = dense_search(index, qv, top_k=10, threshold=0.55)
        f_cands, _ = reciprocal_rank_fusion(d_cands, b_cands, rrf_k=60, top_k=10)
        _ = reranker.score_pairs([(test_query, c.text) for c, _ in f_cands])

    # 3. Isolated Component Timing over repetitions
    print(f"Running {num_repetitions} repetitions across representative queries...")
    benchmark_queries = [
        "What are the core foundations and Q-Matrix formulation of the DINA model?",
        "What is the target inference speed and graph validation library in KnowledgeTrace MCQ?",
        "What are the four BKT parameters and their default values?",
        "What does Kahn's algorithm validate in the prerequisite graph?",
        "How is the confidence score calculated when student mastery uncertainty is maximum?",
    ]

    timings = {
        "embed": [],
        "bm25_search": [],
        "dense_search": [],
        "rrf_fusion": [],
        "cross_encoder_10": [],
        "threshold_filter": [],
        "end_to_end_hybrid_rerank": [],
        "end_to_end_bm25_rerank": [],
    }

    for rep in range(num_repetitions):
        q = benchmark_queries[rep % len(benchmark_queries)]

        # --- Component 1: Embedding ---
        t0 = time.perf_counter()
        qv = client.embed(q, settings.embedding_model)
        t1 = time.perf_counter()
        timings["embed"].append((t1 - t0) * 1000.0)

        # --- Component 2: BM25 Search ---
        t2 = time.perf_counter()
        bm25_cands = bm25_idx.search(q, top_k=10, min_score=3.0)
        t3 = time.perf_counter()
        timings["bm25_search"].append((t3 - t2) * 1000.0)

        # --- Component 3: Dense Search ---
        t4 = time.perf_counter()
        dense_cands = dense_search(index, qv, top_k=10, threshold=0.55)
        t5 = time.perf_counter()
        timings["dense_search"].append((t5 - t4) * 1000.0)

        # --- Component 4: RRF Fusion ---
        t6 = time.perf_counter()
        hybrid_cands, _ = reciprocal_rank_fusion(dense_cands, bm25_cands, rrf_k=60, top_k=10)
        t7 = time.perf_counter()
        timings["rrf_fusion"].append((t7 - t6) * 1000.0)

        # Ensure exactly 10 pairs for clean normalization
        pairs_to_score = [(q, chunk.text) for chunk, _ in hybrid_cands[:10]]
        # --- Component 5: Cross-Encoder Scoring (10 pairs) ---
        t8 = time.perf_counter()
        scores = reranker.score_pairs(pairs_to_score)
        t9 = time.perf_counter()
        timings["cross_encoder_10"].append((t9 - t8) * 1000.0)

        # --- Component 6: Evidence Threshold Filtering & Sorting ---
        t10 = time.perf_counter()
        scored = [
            (chunk, float(score))
            for (chunk, _), score in zip(hybrid_cands[:10], scores)
            if float(score) >= -3.0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        final_top4 = scored[:4]
        t11 = time.perf_counter()
        timings["threshold_filter"].append((t11 - t10) * 1000.0)

        # --- Complete End-to-End Hybrid + Reranker pipeline (single stopwatch) ---
        t_e2e_0 = time.perf_counter()
        # 1. embed
        qv_e2e = client.embed(q, settings.embedding_model)
        # 2. bm25 & dense
        b_cands_e2e = bm25_idx.search(q, top_k=10, min_score=3.0)
        d_cands_e2e = dense_search(index, qv_e2e, top_k=10, threshold=0.55)
        # 3. fusion
        h_cands_e2e, _ = reciprocal_rank_fusion(d_cands_e2e, b_cands_e2e, rrf_k=60, top_k=10)
        # 4. rerank & filter
        rep_e2e = reranker.rerank(q, h_cands_e2e, top_k=4, score_threshold=-3.0)
        t_e2e_1 = time.perf_counter()
        timings["end_to_end_hybrid_rerank"].append((t_e2e_1 - t_e2e_0) * 1000.0)

        # --- Complete End-to-End BM25 + Reranker pipeline (single stopwatch) ---
        t_b_0 = time.perf_counter()
        b_cands_only = bm25_idx.search(q, top_k=10, min_score=3.0)
        rep_b_only = reranker.rerank(q, b_cands_only, top_k=4, score_threshold=-3.0)
        t_b_1 = time.perf_counter()
        timings["end_to_end_bm25_rerank"].append((t_b_1 - t_b_0) * 1000.0)

    # 4. Compute Statistics
    stats = {}
    for name, vals in timings.items():
        stats[name] = {
            "mean_ms": round(statistics.mean(vals), 2),
            "median_ms": round(statistics.median(vals), 2),
            "p95_ms": round(np.percentile(vals, 95), 2),
            "std_dev_ms": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
        }

    # Sum of individual components
    component_sum_hybrid_mean = (
        stats["embed"]["mean_ms"]
        + stats["bm25_search"]["mean_ms"]
        + stats["dense_search"]["mean_ms"]
        + stats["rrf_fusion"]["mean_ms"]
        + stats["cross_encoder_10"]["mean_ms"]
        + stats["threshold_filter"]["mean_ms"]
    )
    discrepancy_hybrid_ms = stats["end_to_end_hybrid_rerank"]["mean_ms"] - component_sum_hybrid_mean

    component_sum_bm25_mean = (
        stats["bm25_search"]["mean_ms"]
        + stats["cross_encoder_10"]["mean_ms"]
        + stats["threshold_filter"]["mean_ms"]
    )
    discrepancy_bm25_ms = stats["end_to_end_bm25_rerank"]["mean_ms"] - component_sum_bm25_mean

    print("\n" + "=" * 80)
    print(f"{'COMPONENT':<35} | {'MEAN (ms)':<10} | {'MEDIAN (ms)':<12} | {'P95 (ms)':<10}")
    print("-" * 80)
    for name, data in stats.items():
        print(f"{name:<35} | {data['mean_ms']:<10.2f} | {data['median_ms']:<12.2f} | {data['p95_ms']:<10.2f}")
    print("=" * 80)

    print(f"\nRECONCILIATION AUDIT:")
    print(f"Hybrid + Reranker Component Sum:  {component_sum_hybrid_mean:8.2f} ms")
    print(f"Hybrid + Reranker Measured E2E:  {stats['end_to_end_hybrid_rerank']['mean_ms']:8.2f} ms")
    print(f"Discrepancy (dispatch/overhead): {discrepancy_hybrid_ms:8.2f} ms ({(discrepancy_hybrid_ms/stats['end_to_end_hybrid_rerank']['mean_ms'])*100:.2f}%)")
    print("-" * 50)
    print(f"BM25 + Reranker Component Sum:    {component_sum_bm25_mean:8.2f} ms")
    print(f"BM25 + Reranker Measured E2E:    {stats['end_to_end_bm25_rerank']['mean_ms']:8.2f} ms")
    print(f"Discrepancy (dispatch/overhead): {discrepancy_bm25_ms:8.2f} ms ({(discrepancy_bm25_ms/stats['end_to_end_bm25_rerank']['mean_ms'])*100:.2f}%)")

    report = {
        "num_repetitions": num_repetitions,
        "cold_latencies_ms": {
            "index_load": cold_index_load_ms,
            "bm25_build": cold_bm25_build_ms,
            "reranker_load": cold_reranker_load_ms,
            "first_embed": cold_embed_ms,
            "first_reranker_inference": cold_rerank_first_inference_ms,
        },
        "warm_component_stats": stats,
        "reconciliation": {
            "hybrid_component_sum_ms": round(component_sum_hybrid_mean, 2),
            "hybrid_measured_e2e_ms": stats["end_to_end_hybrid_rerank"]["mean_ms"],
            "hybrid_discrepancy_ms": round(discrepancy_hybrid_ms, 2),
            "bm25_component_sum_ms": round(component_sum_bm25_mean, 2),
            "bm25_measured_e2e_ms": stats["end_to_end_bm25_rerank"]["mean_ms"],
            "bm25_discrepancy_ms": round(discrepancy_bm25_ms, 2),
        },
    }
    return report


if __name__ == "__main__":
    report = run_latency_benchmark(num_repetitions=30)
    out_file = PROJECT_ROOT / "data" / "evaluation" / "latency_reconciliation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote latency benchmark report to: {out_file}")

