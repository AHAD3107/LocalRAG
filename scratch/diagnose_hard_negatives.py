"""Diagnose hard negative query scores and failure modes."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.storage.vector_index import VectorIndex
from src.retrieval.bm25_search import BM25Index
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import CrossEncoderReranker
from src.embeddings.ollama import OllamaClient

idx = VectorIndex.load(Path("data/index"), settings.embedding_model, settings.chunk_size, settings.chunk_overlap)
bm = BM25Index(idx.chunks)
reranker = CrossEncoderReranker(device="cpu")
client = OllamaClient(settings.ollama_base_url)

benchmark = json.loads(Path("data/evaluation/retrieval_benchmark_100.json").read_text(encoding="utf-8"))
hard_negatives = [x for x in benchmark if x["category"] == "Hard Negative"]

print(f"Loaded {len(hard_negatives)} hard negatives.")
print("=" * 100)

for q in hard_negatives:
    qid = q["question_id"]
    qtext = q["question"]
    qv = client.embed(qtext, settings.embedding_model)
    cands, _ = hybrid_search(idx, qtext, qv, bm, top_k=10)
    rep = reranker.rerank_with_report(qtext, cands, top_k=3)
    if not rep.details:
        print(f"[{qid}] REJECTED by stage 1: {qtext}")
    else:
        top = rep.details[0]
        # find matching chunk
        top_chunk = [c for c, _ in rep.results if c.chunk_id == top.chunk_id][0]
        preview = top_chunk.text.replace("\n", " ")[:120]
        print(f"[{qid}] Logit={top.rerank_score:6.2f} | Sig={top.sigmoid_score:.4f} | Chunk={top.chunk_id}")
        print(f"       Q: {qtext}")
        print(f"       Retrieved Text: {preview}...")
        print("-" * 100)
