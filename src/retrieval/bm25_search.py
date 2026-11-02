"""Experimental sparse BM25 retrieval for LocalRAG.

Implements Okapi BM25 ranking over document chunks with technical-token awareness:
- Preserves compound technical identifiers (e.g., P(L0), Q-Matrix, networkx.DiGraph, 0.20, 15ms)
- Provides document-specific filtering (doc_id)
- Zero external search service dependencies (pure Python & math)
- Maintains strict compatibility with VectorIndex chunk metadata
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from src.ingestion.models import Chunk


# Standard English stop words to reduce noise while preserving variables and technical symbols
STOP_WORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "you", "your", "yours", "yourself", "yourselves",
})

# Regex pattern for technical compound tokens:
# 1. Parenthesized notation: P(L0), P(T), P(S), P(G), P(Lt), P(Lt+1), xi_j(alpha)
# 2. Dotted identifiers: networkx.DiGraph, nx.is_directed_acyclic_graph, routers.student
# 3. Hyphenated technical compounds: Q-Matrix, Log-Sum-Exp, Rule-Space, non-blocking
# 4. Numbers and metrics: 0.20, 0.15, 0.8855, 15ms, 25%, 2^30
# 5. Standard alphanumeric words: [a-zA-Z0-9_]+
TOKEN_PATTERN = re.compile(
    r"[a-zA-Z]+(?:\([a-zA-Z0-9_+\-]+\))+|"             # P(L0), P(Lt+1)
    r"[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)+|"             # networkx.DiGraph, routers.student
    r"[a-zA-Z0-9]+-[a-zA-Z0-9]+|"                     # Q-Matrix, Log-Sum-Exp
    r"\d+(?:\.\d+)?(?:ms|s|%|k)?|"                    # 0.20, 15ms, 25%
    r"[a-zA-Z0-9_]+",                                 # alphanumeric words
    re.IGNORECASE,
)

SUBTOKEN_SPLIT = re.compile(r"[\(\)\.\-_+\s]+")


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Tokenize input text preserving technical compounds and decomposing them into sub-tokens.
    
    Examples:
        'P(L0)' -> ['p(l0)', 'p', 'l0']
        'Q-Matrix' -> ['q-matrix', 'q', 'matrix']
        'networkx.DiGraph' -> ['networkx.digraph', 'networkx', 'digraph']
        '0.20' -> ['0.20']
    """
    if not text:
        return []

    tokens: list[str] = []
    matches = TOKEN_PATTERN.findall(text)

    for match in matches:
        lowered = match.lower().strip()
        if not lowered:
            continue

        # Check if it's a compound token
        is_compound = any(char in lowered for char in "().-_+") and not lowered.replace(".", "").isdigit()

        if is_compound:
            tokens.append(lowered)
            # Emit sub-tokens for fine-grained matching
            parts = [p.strip() for p in SUBTOKEN_SPLIT.split(lowered) if p.strip()]
            for part in parts:
                if remove_stopwords and part in STOP_WORDS and len(part) > 1:
                    continue
                tokens.append(part)
        else:
            if remove_stopwords and lowered in STOP_WORDS and len(lowered) > 1:
                continue
            tokens.append(lowered)

    return tokens


@dataclass
class BM25Index:
    """
    In-memory Okapi BM25 index built over a sequence of Chunk objects.
    """
    chunks: list[Chunk]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self):
        self.doc_count = len(self.chunks)
        self.doc_lengths: list[int] = []
        self.doc_tokens: list[list[str]] = []
        self.inverted_index: dict[str, dict[int, int]] = {}
        self.doc_frequencies: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avg_doc_len = 0.0

        if self.doc_count > 0:
            total_len = 0
            for idx, chunk in enumerate(self.chunks):
                tokens = tokenize(chunk.text)
                self.doc_tokens.append(tokens)
                length = len(tokens)
                self.doc_lengths.append(length)
                total_len += length

                counts = Counter(tokens)
                for term, freq in counts.items():
                    if term not in self.inverted_index:
                        self.inverted_index[term] = {}
                        self.doc_frequencies[term] = 0
                    self.inverted_index[term][idx] = freq
                    self.doc_frequencies[term] += 1

            self.avg_doc_len = total_len / self.doc_count

            # Robertson-Spärck Jones IDF with +1 floor (BM25+ / Lucene formulation)
            # Guarantees non-negative IDF for common terms: ln(1 + (N - df + 0.5) / (df + 0.5))
            for term, df in self.doc_frequencies.items():
                self.idf[term] = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 4,
        min_score: float = 0.0,
        doc_id: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Search indexed chunks using Okapi BM25.
        
        Args:
            query: User search question or query string.
            top_k: Maximum number of ranked chunks to return.
            min_score: Minimum BM25 score required for returned chunks.
            doc_id: Optional document ID filter.
            
        Returns:
            List of (Chunk, bm25_score) tuples sorted in descending order of score.
            
        Raises:
            ValueError: If doc_id is provided but not found in the index.
        """
        if top_k <= 0 or not self.chunks:
            if doc_id is not None:
                raise ValueError(f"Document '{doc_id}' was not found in the index.")
            return []

        # Validate doc_id filter if supplied
        candidate_indices: set[int] | None = None
        if doc_id is not None:
            matching = [i for i, c in enumerate(self.chunks) if c.doc_id == doc_id]
            if not matching:
                raise ValueError(f"Document '{doc_id}' was not found in the index.")
            candidate_indices = set(matching)

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = Counter()
        query_counts = Counter(query_tokens)

        for term, q_freq in query_counts.items():
            if term not in self.inverted_index:
                continue

            term_idf = self.idf[term]
            postings = self.inverted_index[term]

            for doc_idx, tf in postings.items():
                if candidate_indices is not None and doc_idx not in candidate_indices:
                    continue

                d_len = self.doc_lengths[doc_idx]
                # Standard Okapi BM25 document term score
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (d_len / self.avg_doc_len))
                scores[doc_idx] += term_idf * (numerator / denominator)

        if not scores:
            return []

        # Sort descending by score
        ranked_items = sorted(scores.items(), key=lambda item: -item[1])

        results: list[tuple[Chunk, float]] = []
        for doc_idx, score in ranked_items:
            if score < min_score or score <= 0.0:
                continue
            results.append((self.chunks[doc_idx], float(score)))
            if len(results) >= top_k:
                break

        return results


def bm25_search(
    chunks: Sequence[Chunk],
    query: str,
    top_k: int = 4,
    min_score: float = 0.0,
    doc_id: str | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[Chunk, float]]:
    """Convenience helper to index and search chunks in a single call."""
    index = BM25Index(list(chunks), k1=k1, b=b)
    return index.search(query, top_k=top_k, min_score=min_score, doc_id=doc_id)

