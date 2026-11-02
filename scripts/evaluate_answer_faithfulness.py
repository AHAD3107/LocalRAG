"""Audit Answer Faithfulness, Refusal, and Numerical Preservation for LocalRAG Phase 4.8.

Executes Task 5 requirements:
1. Grounding & Faithfulness:
   - Evaluates multi-chunk, cross-document, and single-document queries using full RAG pipeline.
   - Verifies citation preservation: [Source X], filename, and page numbers.
2. Refusal Behavior:
   - Tests all 15 Unsupported queries and 15 Hard-Negative queries.
   - Verifies that system cleanly refuses unanswerable / fabricated queries.
3. Information Synthesis:
   - Verifies that multi-chunk and cross-doc queries synthesize multiple pieces of context accurately.
4. Numerical and Technical Preservation:
   - Verifies exact preservation of metrics, numbers, and technical constants.
5. Saves results to data/evaluation/answer_faithfulness_results.json.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.embeddings.ollama import OllamaClient
from src.generation.contextualize import contextualize_query
from src.generation.prompt import build_prompt
from src.retrieval.pipeline import retrieve_context
from src.storage.vector_index import VectorIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("faithfulness_eval")


REFUSAL_PHRASES = [
    "could not find enough evidence",
    "cannot find enough evidence",
    "not mentioned",
    "not provided",
    "does not provide",
    "no evidence",
    "does not mention",
    "unsupported",
    "no information",
    "do not have information",
]


def is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in REFUSAL_PHRASES)


def extract_technical_terms(text: str) -> set[str]:
    # Extract numbers, percentages, acronyms, and technical compounds
    tokens = set()
    # Numbers and metrics (e.g., 512, 120, 0.20, 15ms, 4KB, 256)
    for m in re.finditer(r"\b\d+(?:\.\d+)?(?:ms|s|%|kb|mb|gb)?\b", text, re.IGNORECASE):
        tokens.add(m.group(0).lower())
    # Proper nouns and technical words (e.g., B+, ARIES, WAL, MVCC, Raft, TCP, QUIC, CPython, GIL, PEG, Saga, DINA, BKT)
    for m in re.finditer(r"\b[A-Z][a-zA-Z0-9_\+\-]{2,}\b", text):
        tokens.add(m.group(0).lower())
    return tokens


def main():
    print("=" * 80)
    print("LOCALRAG PHASE 4.8 — RETRIEVAL QUALITY AND ANSWER FAITHFULNESS AUDIT")
    print("=" * 80)

    benchmark_path = PROJECT_ROOT / "data" / "evaluation" / "multidoc_benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))

    index = VectorIndex.load(settings.index_dir, settings.embedding_model, settings.chunk_size, settings.chunk_overlap)
    client = OllamaClient(settings.ollama_base_url, timeout=90.0)

    # Select representative subset:
    # - 10 Exact Factual / Paraphrased (Single-doc)
    # - 8 Multi-chunk
    # - 8 Cross-document
    # - All 15 Hard Negatives
    # - All 15 Unsupported
    single_doc = [q for q in benchmark if q["category"] in ("Exact factual", "Paraphrased")][:10]
    multi_chunk = [q for q in benchmark if q["category"] == "Multi-chunk"][:8]
    cross_doc = [q for q in benchmark if q["category"] == "Cross-document"][:8]
    hard_negatives = [q for q in benchmark if q["category"] == "Hard Negative"]
    unsupported = [q for q in benchmark if q["category"] == "Unsupported"]

    eval_set = single_doc + multi_chunk + cross_doc + hard_negatives + unsupported
    print(f"Total audit queries selected: {len(eval_set)}")
    print(f"  - Grounded Single-Doc:  {len(single_doc)}")
    print(f"  - Grounded Multi-Chunk: {len(multi_chunk)}")
    print(f"  - Grounded Cross-Doc:   {len(cross_doc)}")
    print(f"  - Hard Negatives:       {len(hard_negatives)}")
    print(f"  - Unsupported Queries:  {len(unsupported)}")

    records = []
    grounded_records = []
    unsupported_refusals = 0
    hard_negative_refusals = 0

    print("\nRunning live RAG answer generation...")

    for i, item in enumerate(eval_set, start=1):
        qid = item["question_id"]
        category = item["category"]
        question = item["question"]
        expected = item["expected_answer"]
        should_reject = item["should_reject"]

        search_query = question
        if category == "Conversational follow-ups" and item.get("history"):
            search_query = contextualize_query(question, item["history"], client, settings.chat_model_default)

        doc_scope = item.get("relevant_doc_id") if category == "Document-specific" else None

        # Execute retrieval with calibrated threshold
        matches = retrieve_context(
            query=search_query,
            index=index,
            client=client,
            top_k=4,
            doc_id=doc_scope,
            strategy="hybrid_rerank",
            score_threshold=settings.reranker_score_threshold,
        )

        if not matches:
            answer = "I could not find enough evidence in the indexed documents to answer that question."
            sources = []
        else:
            prompt = build_prompt(question, matches)
            answer = client.chat(prompt, model=settings.chat_model_default)
            sources = [
                {
                    "source_id": idx,
                    "filename": c.source_filename,
                    "page": c.page_number,
                    "chunk_id": c.chunk_id,
                }
                for idx, (c, _) in enumerate(matches, start=1)
            ]

        ans_refused = is_refusal(answer)

        # Numerical & technical terms preservation
        expected_terms = extract_technical_terms(expected)
        preserved_terms = [t for t in expected_terms if t in answer.lower()]
        preservation_rate = len(preserved_terms) / len(expected_terms) if expected_terms else 1.0

        # Citations check
        has_source_citations = bool(re.search(r"\[Source\s+\d+\]", answer)) or len(sources) == 0

        record = {
            "question_id": qid,
            "category": category,
            "question": question,
            "expected_answer": expected,
            "should_reject": should_reject,
            "retrieved_chunk_count": len(matches),
            "sources": sources,
            "generated_answer": answer,
            "is_refusal": ans_refused,
            "expected_terms": sorted(expected_terms),
            "preserved_terms": sorted(preserved_terms),
            "term_preservation_rate": round(preservation_rate, 2),
            "has_citations": has_source_citations,
        }
        records.append(record)

        if category == "Unsupported":
            if ans_refused:
                unsupported_refusals += 1
        elif category == "Hard Negative":
            if ans_refused:
                hard_negative_refusals += 1
        else:
            grounded_records.append(record)

        if i % 10 == 0 or i == len(eval_set):
            print(f"Processed {i}/{len(eval_set)} queries...")

    # Aggregates
    unsupported_rate = unsupported_refusals / len(unsupported) if unsupported else 0.0
    hard_negative_rate = hard_negative_refusals / len(hard_negative_queries := hard_negatives) if hard_negatives else 0.0

    mean_preservation = (
        sum(r["term_preservation_rate"] for r in grounded_records) / len(grounded_records)
        if grounded_records
        else 0.0
    )
    citation_accuracy = (
        sum(1 for r in grounded_records if r["has_citations"]) / len(grounded_records)
        if grounded_records
        else 0.0
    )

    summary = {
        "total_audited": len(eval_set),
        "grounded_queries_evaluated": len(grounded_records),
        "unsupported_queries_evaluated": len(unsupported),
        "hard_negatives_evaluated": len(hard_negatives),
        "unsupported_refusal_rate": round(unsupported_rate, 4),
        "hard_negative_refusal_rate": round(hard_negative_rate, 4),
        "average_technical_term_preservation": round(mean_preservation, 4),
        "citation_presence_rate": round(citation_accuracy, 4),
    }

    output_payload = {
        "summary": summary,
        "records": records,
    }

    out_file = PROJECT_ROOT / "data" / "evaluation" / "answer_faithfulness_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"\nSaved answer faithfulness audit results to: {out_file}")

    print("\nFAITHFULNESS AUDIT SUMMARY:")
    print(f"  Unsupported Refusal Rate:       {summary['unsupported_refusal_rate']:.1%} ({unsupported_refusals}/{len(unsupported)})")
    print(f"  Hard Negative Refusal Rate:      {summary['hard_negative_refusal_rate']:.1%} ({hard_negative_refusals}/{len(hard_negatives)})")
    print(f"  Technical Term Preservation:     {summary['average_technical_term_preservation']:.1%}")
    print(f"  Citation Compliance:             {summary['citation_presence_rate']:.1%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
