# eval_survival_ranking.py
#
# Evaluate BM25 (and optional CrossEncoder reranker) on survival_eval_queries.jsonl
#
# Metrics:
#   - MRR@10
#   - nDCG@10
#   - Recall@5
#
# Assumes:
#   - retriever_bm25.py provides:
#       * load_index(index_path) -> (bm25, docs)
#       * retrieve(query, bm25, docs, top_k)
#   - reranker_ce.py provides RerankerCE (your trained CrossEncoder wrapper)
#   - data/survival_eval_queries.jsonl exists with:
#       {"query": "...", "relevant_ids": ["HOME/...pdf#123", ...]}

import os
import json
import math
from pathlib import Path
from typing import List, Dict, Set

from retriever_bm25 import load_index, retrieve as bm25_retrieve


# Try to import the reranker wrapper
try:
    from reranker_ce import RerankerCE
    HAS_RERANKER = True
except ImportError:
    RerankerCE = None
    HAS_RERANKER = False


# ---- Paths / knobs ----

EVAL_FILE = Path("data/survival_eval_queries.jsonl")

# BM25 index path (you already set this correctly)
BM25_INDEX_PATH = Path("external/Survival-Data/bm25_survival_home.idx")

# Reranker checkpoint dir:
# - default: checkpoints/survival_ce
# - can override with env var: RERANKER_CKPT=/path/to/checkpoint_dir
#RERANKER_CKPT = os.environ.get("RERANKER_CKPT", "checkpoints/survival_ce")
#CKPT_DIR = Path(RERANKER_CKPT)
RERANKER_CKPT = os.environ.get("RERANKER_CKPT", "runs/modelA_miniLM_e2_bm25neg")
CKPT_DIR = Path(RERANKER_CKPT)

BM25_TOP_K = 50      # how many docs BM25 retrieves
RERANK_TOP_K = 20    # how many top BM25 docs we rerank (and then evaluate)


# ---- Metrics helpers ----

def mrr_at_k(ranked_ids: List[str], relevant: Set[str], k: int = 10) -> float:
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked_ids: List[str], relevant: Set[str], k: int = 10) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if doc_id in relevant else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_ids: List[str], relevant: Set[str], k: int = 10) -> float:
    dcg = dcg_at_k(ranked_ids, relevant, k)
    ideal_ranked = [1.0] * min(len(relevant), k)
    ideal_dcg = 0.0
    for i, rel in enumerate(ideal_ranked, start=1):
        ideal_dcg += rel / math.log2(i + 1)
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def recall_at_k(ranked_ids: List[str], relevant: Set[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    retrieved = set(ranked_ids[:k])
    hit = len(retrieved & relevant)
    return hit / len(relevant)


# ---- Load eval queries ----

def load_eval_queries(path: Path) -> List[Dict]:
    queries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("query") or "").strip()
            rel = obj.get("relevant_ids") or []
            if not q or not rel:
                continue
            queries.append({"query": q, "relevant_ids": list({str(r) for r in rel})})
    return queries


# ---- Evaluation runners ----

def eval_bm25_only(eval_queries: List[Dict], bm25, docs: List[Dict]):
    print("Evaluating BM25 baseline...")

    mrrs, ndcgs, recalls = [], [], []

    for ex in eval_queries:
        q = ex["query"]
        relevant = set(ex["relevant_ids"])

        results = bm25_retrieve(q, bm25, docs, top_k=BM25_TOP_K)
        ranked_ids = [r["id"] for r in results]

        mrrs.append(mrr_at_k(ranked_ids, relevant, k=10))
        ndcgs.append(ndcg_at_k(ranked_ids, relevant, k=10))
        recalls.append(recall_at_k(ranked_ids, relevant, k=5))

    avg_mrr = sum(mrrs) / len(mrrs)
    avg_ndcg = sum(ndcgs) / len(ndcgs)
    avg_recall = sum(recalls) / len(recalls)

    print(f"BM25 baseline over {len(eval_queries)} queries:")
    print(f"  MRR@10   = {avg_mrr:.4f}")
    print(f"  nDCG@10  = {avg_ndcg:.4f}")
    print(f"  Recall@5 = {avg_recall:.4f}")
    print()
    return avg_mrr, avg_ndcg, avg_recall


def eval_bm25_plus_reranker(eval_queries: List[Dict], bm25, docs: List[Dict], ckpt_dir: Path):
    if not HAS_RERANKER:
        print("Cannot import reranker_ce.RerankerCE — install it or check PYTHONPATH.")
        return None
    if not ckpt_dir.exists():
        print(f"Reranker checkpoint directory does not exist: {ckpt_dir}")
        return None

    print(f"Loading reranker from {ckpt_dir} ...")
    reranker = RerankerCE(str(ckpt_dir))

    print("Evaluating BM25 + CrossEncoder reranker...")

    mrrs, ndcgs, recalls = [], [], []

    for ex in eval_queries:
        q = ex["query"]
        relevant = set(ex["relevant_ids"])

        results = bm25_retrieve(q, bm25, docs, top_k=BM25_TOP_K)
        if not results:
            mrrs.append(0.0)
            ndcgs.append(0.0)
            recalls.append(0.0)
            continue

        candidates = results[:RERANK_TOP_K]
        texts = [c["text"] for c in candidates]
        ids = [c["id"] for c in candidates]

        scores = reranker.score(q, texts)

        scored = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
        reranked_ids = [doc_id for doc_id, s in scored]

        mrrs.append(mrr_at_k(reranked_ids, relevant, k=10))
        ndcgs.append(ndcg_at_k(reranked_ids, relevant, k=10))
        recalls.append(recall_at_k(reranked_ids, relevant, k=5))

    avg_mrr = sum(mrrs) / len(mrrs)
    avg_ndcg = sum(ndcgs) / len(ndcgs)
    avg_recall = sum(recalls) / len(recalls)

    print(f"BM25 + reranker over {len(eval_queries)} queries:")
    print(f"  MRR@10   = {avg_mrr:.4f}")
    print(f"  nDCG@10  = {avg_ndcg:.4f}")
    print(f"  Recall@5 = {avg_recall:.4f}")
    print()
    return avg_mrr, avg_ndcg, avg_recall


# ---- Main ----

def main():
    if not EVAL_FILE.exists():
        raise SystemExit(f"Eval file not found: {EVAL_FILE}")

    eval_queries = load_eval_queries(EVAL_FILE)
    if not eval_queries:
        raise SystemExit("No valid eval queries found.")

    print(f"Loaded {len(eval_queries)} eval queries from {EVAL_FILE}")

    if not BM25_INDEX_PATH.exists():
        raise SystemExit(f"BM25 index not found at {BM25_INDEX_PATH}.")

    print(f"Loading BM25 index from {BM25_INDEX_PATH} ...")
    bm25, docs = load_index(str(BM25_INDEX_PATH))
    print(f"Loaded {len(docs)} chunks from BM25 index.")

    # BM25 baseline
    bm25_metrics = eval_bm25_only(eval_queries, bm25, docs)

    # BM25 + reranker
    rerank_metrics = eval_bm25_plus_reranker(eval_queries, bm25, docs, CKPT_DIR)

    print("Done.")
    if bm25_metrics and rerank_metrics:
        bm_mrr, bm_ndcg, bm_rec = bm25_metrics
        rr_mrr, rr_ndcg, rr_rec = rerank_metrics
        print("Summary (for your report):")
        print(f"  BM25:          MRR@10={bm_mrr:.2f}, nDCG@10={bm_ndcg:.2f}, Recall@5={bm_rec:.2f}")
        print(f"  BM25+Reranker: MRR@10={rr_mrr:.2f}, nDCG@10={rr_ndcg:.2f}, Recall@5={rr_rec:.2f}")
        print("  (ΔMRR, ΔnDCG, ΔRecall are the gains you can brag about.)")


if __name__ == "__main__":
    main()

