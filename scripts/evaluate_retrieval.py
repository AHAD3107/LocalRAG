"""Comprehensive retrieval quality evaluation for LocalRAG Phase 4.

Evaluates:
- Exact-match queries
- Paraphrased queries
- Multi-chunk queries
- Unsupported questions
- Hard negatives / semantically similar irrelevant queries

Computes:
- Recall@K (K=1, 2, 4)
- Precision@K (K=1, 2, 4)
- Mean Reciprocal Rank (MRR)
- Retrieval latency (mean, median, p95 in ms)
- Unsupported query rejection rate
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import settings
from src.embeddings.ollama import OllamaClient
from src.ingestion.models import Chunk
from src.retrieval.search import search
from src.storage.vector_index import VectorIndex


def run_comprehensive_evaluation():
    client = OllamaClient(settings.ollama_base_url, timeout=60.0)
    if not client.is_available():
        raise RuntimeError(f"Ollama is not available on {settings.ollama_base_url}")

    # Define corpus chunks with realistic specs and document content
    corpus = [
        Chunk(
            chunk_id="chunk_1",
            doc_id="doc_specs",
            source_filename="specs.pdf",
            page_number=1,
            chunk_index=0,
            text="Project name: LocalRAG. Project purpose: Offline document intelligence and private neural search.",
        ),
        Chunk(
            chunk_id="chunk_2",
            doc_id="doc_specs",
            source_filename="specs.pdf",
            page_number=2,
            chunk_index=1,
            text="Project deadline: 25 October 2026. The backend framework is built entirely with FastAPI and Python 3.12.",
        ),
        Chunk(
            chunk_id="chunk_3",
            doc_id="doc_specs",
            source_filename="specs.pdf",
            page_number=3,
            chunk_index=2,
            text="The default embedding model is nomic-embed-text producing 768-dimensional vectors. The default chat generation model is llama3.2:1b running locally via Ollama.",
        ),
        Chunk(
            chunk_id="chunk_4",
            doc_id="doc_specs",
            source_filename="specs.pdf",
            page_number=4,
            chunk_index=3,
            text="Vector storage is handled using persistent NumPy arrays with atomic replacement and cosine similarity search.",
        ),
        Chunk(
            chunk_id="chunk_5",
            doc_id="doc_specs",
            source_filename="specs.pdf",
            page_number=5,
            chunk_index=4,
            text="Security and privacy: all data is processed strictly on-premise on local GPU hardware without external cloud API dependencies.",
        ),
    ]

    print(f"Generating real embeddings for {len(corpus)} corpus chunks via {settings.embedding_model}...")
    corpus_vectors = client.embed_batch([c.text for c in corpus], settings.embedding_model)

    temp_dir = Path("data/eval_index_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    index = VectorIndex(temp_dir, settings.embedding_model, settings.chunk_size, settings.chunk_overlap)
    index.add(corpus, corpus_vectors)

    # Test queries covering all required categories
    test_cases = [
        # 1. Exact-match questions
        {
            "category": "Exact-match",
            "query": "When is the project deadline?",
            "expected_chunk_ids": ["chunk_2"],
        },
        {
            "category": "Exact-match",
            "query": "What is the project name and purpose?",
            "expected_chunk_ids": ["chunk_1"],
        },
        {
            "category": "Exact-match",
            "query": "How is vector storage handled?",
            "expected_chunk_ids": ["chunk_4"],
        },
        # 2. Paraphrased questions
        {
            "category": "Paraphrased",
            "query": "By which date does this system need to be completed?",
            "expected_chunk_ids": ["chunk_2"],
        },
        {
            "category": "Paraphrased",
            "query": "What offline technology does this tool provide?",
            "expected_chunk_ids": ["chunk_1"],
        },
        {
            "category": "Paraphrased",
            "query": "Which AI models are used for vector representation and chat answers?",
            "expected_chunk_ids": ["chunk_3"],
        },
        {
            "category": "Paraphrased",
            "query": "Does the system maintain confidentiality without cloud access?",
            "expected_chunk_ids": ["chunk_5"],
        },
        # 3. Multi-chunk questions
        {
            "category": "Multi-chunk",
            "query": "What backend framework and local AI models are used in LocalRAG?",
            "expected_chunk_ids": ["chunk_2", "chunk_3"],
        },
        {
            "category": "Multi-chunk",
            "query": "How does LocalRAG maintain privacy and how does it store vector embeddings?",
            "expected_chunk_ids": ["chunk_4", "chunk_5"],
        },
        # 4. Questions with no supporting document evidence (Unsupported)
        {
            "category": "Unsupported",
            "query": "Who is the lead project manager and architect?",
            "expected_chunk_ids": [],
        },
        {
            "category": "Unsupported",
            "query": "What is the current weather in Tokyo?",
            "expected_chunk_ids": [],
        },
        {
            "category": "Unsupported",
            "query": "How much venture capital funding was raised?",
            "expected_chunk_ids": [],
        },
        # 5. Semantically similar but factually irrelevant questions (Hard Negatives)
        {
            "category": "Hard Negative",
            "query": "What is the deadline for filing corporate taxes?",
            "expected_chunk_ids": [],
        },
        {
            "category": "Hard Negative",
            "query": "Which AWS cloud provider hosts serverless Lambda functions?",
            "expected_chunk_ids": [],
        },
        {
            "category": "Hard Negative",
            "query": "How do I configure my OpenAI GPT-4 API key in the cloud?",
            "expected_chunk_ids": [],
        },
        {
            "category": "Hard Negative",
            "query": "What is the pricing model for ChromaDB and Pinecone vector databases?",
            "expected_chunk_ids": [],
        },
    ]

    threshold = settings.similarity_threshold
    k_values = [1, 2, 4]

    detailed_results = []
    latencies_embed_ms = []
    latencies_search_ms = []
    latencies_total_ms = []

    reciprocal_ranks = []
    precisions_at_k = {k: [] for k in k_values}
    recalls_at_k = {k: [] for k in k_values}

    unsupported_total = 0
    unsupported_rejected = 0

    category_stats = {}

    print(f"\nEvaluating {len(test_cases)} queries with similarity threshold = {threshold:.2f}...\n")

    for tc in test_cases:
        category = tc["category"]
        query = tc["query"]
        expected = set(tc["expected_chunk_ids"])
        is_relevant_query = len(expected) > 0

        # Benchmark embedding latency
        t0 = time.perf_counter()
        q_vec = client.embed(query, settings.embedding_model)
        t1 = time.perf_counter()

        # Benchmark search latency
        t2 = time.perf_counter()
        # Retrieve up to top_k=4 above threshold
        matches = search(index, q_vec, top_k=4, threshold=threshold)
        t3 = time.perf_counter()

        # Also get all unconstrained matches to compute rank and raw similarities
        all_matches = search(index, q_vec, top_k=len(corpus), threshold=0.0)
        all_ranked_cids = [m[0].chunk_id for m in all_matches]
        scores_map = {m[0].chunk_id: round(m[1], 4) for m in all_matches}

        embed_ms = (t1 - t0) * 1000.0
        search_ms = (t3 - t2) * 1000.0
        total_ms = embed_ms + search_ms

        latencies_embed_ms.append(embed_ms)
        latencies_search_ms.append(search_ms)
        latencies_total_ms.append(total_ms)

        retrieved_cids = [m[0].chunk_id for m in matches]

        # Calculate metrics
        rr = 0.0
        p_at_k = {}
        r_at_k = {}

        if is_relevant_query:
            # First rank in all_ranked_cids where score >= threshold and cid in expected
            for rank_idx, cid in enumerate(retrieved_cids, start=1):
                if cid in expected:
                    rr = 1.0 / rank_idx
                    break
            reciprocal_ranks.append(rr)

            for k in k_values:
                top_k_retrieved = set(retrieved_cids[:k])
                hits = len(top_k_retrieved & expected)
                p_k = hits / k
                r_k = hits / len(expected)
                p_at_k[k] = round(p_k, 4)
                r_at_k[k] = round(r_k, 4)
                precisions_at_k[k].append(p_k)
                recalls_at_k[k].append(r_k)
        else:
            unsupported_total += 1
            if len(retrieved_cids) == 0:
                unsupported_rejected += 1
            # For non-relevant queries: P@K has hits=0 (or 0 false positives if 0 retrieved)
            for k in k_values:
                top_k_retrieved = set(retrieved_cids[:k])
                p_k = 0.0 if top_k_retrieved else 1.0
                p_at_k[k] = round(p_k, 4)
                r_at_k[k] = 1.0  # None expected, none missing

        # Accumulate per category stats
        if category not in category_stats:
            category_stats[category] = {
                "queries": 0,
                "latencies_ms": [],
                "mrr_list": [],
                "p_at_1": [],
                "r_at_1": [],
                "p_at_2": [],
                "r_at_2": [],
                "p_at_4": [],
                "r_at_4": [],
                "rejected": 0,
            }

        cat = category_stats[category]
        cat["queries"] += 1
        cat["latencies_ms"].append(total_ms)
        if is_relevant_query:
            cat["mrr_list"].append(rr)
            cat["p_at_1"].append(p_at_k[1])
            cat["r_at_1"].append(r_at_k[1])
            cat["p_at_2"].append(p_at_k[2])
            cat["r_at_2"].append(r_at_k[2])
            cat["p_at_4"].append(p_at_k[4])
            cat["r_at_4"].append(r_at_k[4])
        else:
            if len(retrieved_cids) == 0:
                cat["rejected"] += 1

        entry = {
            "category": category,
            "query": query,
            "expected_chunk_ids": list(expected),
            "retrieved_chunk_ids": retrieved_cids,
            "all_scores": scores_map,
            "reciprocal_rank": round(rr, 4) if is_relevant_query else None,
            "precision_at_k": p_at_k if is_relevant_query else None,
            "recall_at_k": r_at_k if is_relevant_query else None,
            "rejected_below_threshold": len(retrieved_cids) == 0,
            "latency_ms": {
                "embed": round(embed_ms, 2),
                "search": round(search_ms, 2),
                "total": round(total_ms, 2),
            },
        }
        detailed_results.append(entry)

    # Cleanup temp index
    for f in temp_dir.iterdir():
        f.unlink(missing_ok=True)
    temp_dir.rmdir()

    # Overall Summary Calculations
    mean_mrr = statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    mean_p_at_1 = statistics.mean(precisions_at_k[1]) if precisions_at_k[1] else 0.0
    mean_r_at_1 = statistics.mean(recalls_at_k[1]) if recalls_at_k[1] else 0.0
    mean_p_at_2 = statistics.mean(precisions_at_k[2]) if precisions_at_k[2] else 0.0
    mean_r_at_2 = statistics.mean(recalls_at_k[2]) if recalls_at_k[2] else 0.0
    mean_p_at_4 = statistics.mean(precisions_at_k[4]) if precisions_at_k[4] else 0.0
    mean_r_at_4 = statistics.mean(recalls_at_k[4]) if recalls_at_k[4] else 0.0

    rejection_rate = (unsupported_rejected / unsupported_total) if unsupported_total > 0 else 1.0

    latency_summary = {
        "mean_embed_ms": round(statistics.mean(latencies_embed_ms), 2),
        "mean_search_ms": round(statistics.mean(latencies_search_ms), 2),
        "mean_total_ms": round(statistics.mean(latencies_total_ms), 2),
        "median_total_ms": round(statistics.median(latencies_total_ms), 2),
        "min_total_ms": round(min(latencies_total_ms), 2),
        "max_total_ms": round(max(latencies_total_ms), 2),
        "p95_total_ms": round(np.percentile(latencies_total_ms, 95), 2),
    }

    # Print Report
    print("=" * 80)
    print("               LOCALRAG RETRIEVAL QUALITY EVALUATION REPORT")
    print("=" * 80)
    print(f"Similarity Threshold:           {threshold:.2f}")
    print(f"Embedding Model:                {settings.embedding_model}")
    print(f"Total Test Queries:             {len(test_cases)}")
    print(f"Relevant Queries:               {len(reciprocal_ranks)}")
    print(f"Irrelevant / Unsupported:       {unsupported_total}")
    print("-" * 80)
    print("KEY RETRIEVAL METRICS:")
    print(f"  Mean Reciprocal Rank (MRR):   {mean_mrr:.4f}")
    print(f"  Recall@1:                     {mean_r_at_1:.4f}   | Precision@1: {mean_p_at_1:.4f}")
    print(f"  Recall@2:                     {mean_r_at_2:.4f}   | Precision@2: {mean_p_at_2:.4f}")
    print(f"  Recall@4:                     {mean_r_at_4:.4f}   | Precision@4: {mean_p_at_4:.4f}")
    print(f"  Unsupported Rejection Rate:   {rejection_rate * 100:.1f}% ({unsupported_rejected}/{unsupported_total} correctly rejected)")
    print("-" * 80)
    print("RETRIEVAL LATENCY:")
    print(f"  Embedding Latency (mean):     {latency_summary['mean_embed_ms']} ms")
    print(f"  Vector Search Latency (mean): {latency_summary['mean_search_ms']} ms")
    print(f"  Total Retrieval Latency (mean):{latency_summary['mean_total_ms']} ms")
    print(f"  Median Latency:               {latency_summary['median_total_ms']} ms")
    print(f"  95th Percentile Latency:      {latency_summary['p95_total_ms']} ms")
    print("=" * 80)

    print("\nBREAKDOWN BY QUERY CATEGORY:")
    print(f"{'Category':<15} {'Count':<7} {'MRR':<8} {'Rec@1':<8} {'Prec@1':<8} {'Rec@2':<8} {'Prec@2':<8} {'Rec@4':<8} {'Rejection':<10} {'Latency(ms)':<10}")
    print("-" * 95)
    category_summary = {}
    for cat_name, stats in category_stats.items():
        cnt = stats["queries"]
        avg_lat = round(statistics.mean(stats["latencies_ms"]), 1)
        if stats["mrr_list"]:
            cat_mrr = f"{statistics.mean(stats['mrr_list']):.3f}"
            c_r1 = f"{statistics.mean(stats['r_at_1']):.3f}"
            c_p1 = f"{statistics.mean(stats['p_at_1']):.3f}"
            c_r2 = f"{statistics.mean(stats['r_at_2']):.3f}"
            c_p2 = f"{statistics.mean(stats['p_at_2']):.3f}"
            c_r4 = f"{statistics.mean(stats['r_at_4']):.3f}"
            rej = "N/A"
        else:
            cat_mrr = "N/A"
            c_r1 = "N/A"
            c_p1 = "N/A"
            c_r2 = "N/A"
            c_p2 = "N/A"
            c_r4 = "N/A"
            rej = f"{(stats['rejected'] / cnt) * 100:.0f}%"

        print(f"{cat_name:<15} {cnt:<7} {cat_mrr:<8} {c_r1:<8} {c_p1:<8} {c_r2:<8} {c_p2:<8} {c_r4:<8} {rej:<10} {avg_lat:<10}")
        category_summary[cat_name] = {
            "count": cnt,
            "mrr": float(cat_mrr) if cat_mrr != "N/A" else None,
            "recall_at_1": float(c_r1) if c_r1 != "N/A" else None,
            "precision_at_1": float(c_p1) if c_p1 != "N/A" else None,
            "recall_at_2": float(c_r2) if c_r2 != "N/A" else None,
            "precision_at_2": float(c_p2) if c_p2 != "N/A" else None,
            "recall_at_4": float(c_r4) if c_r4 != "N/A" else None,
            "rejection_rate": stats["rejected"] / cnt if not stats["mrr_list"] else None,
            "mean_latency_ms": avg_lat,
        }

    # Save to JSON
    output_file = Path("data/retrieval_quality_evaluation.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "configuration": {
            "embedding_model": settings.embedding_model,
            "similarity_threshold": threshold,
            "corpus_chunk_count": len(corpus),
            "test_query_count": len(test_cases),
        },
        "overall_metrics": {
            "mean_reciprocal_rank_mrr": round(mean_mrr, 4),
            "recall_at_1": round(mean_r_at_1, 4),
            "precision_at_1": round(mean_p_at_1, 4),
            "recall_at_2": round(mean_r_at_2, 4),
            "precision_at_2": round(mean_p_at_2, 4),
            "recall_at_4": round(mean_r_at_4, 4),
            "precision_at_4": round(mean_p_at_4, 4),
            "unsupported_rejection_rate": round(rejection_rate, 4),
            "latency": latency_summary,
        },
        "category_metrics": category_summary,
        "query_results": detailed_results,
    }
    output_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"\nFull evaluation metrics saved to {output_file.resolve()}\n")
    return report_data


if __name__ == "__main__":
    run_comprehensive_evaluation()
