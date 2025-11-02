#!/usr/bin/env python3
import json, os, random, argparse
from typing import List, Dict
from rank_bm25 import BM25Okapi

def read_jsonl(path):
    out=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out

def contains_answer(text:str, answers:List[str]) -> bool:
    t = text.lower()
    for a in answers or []:
        a=a.strip().lower()
        if a and a in t: return True
    return False

def tokenize(s: str) -> List[str]:
    return s.lower().split()

def main(args):
    random.seed(42)
    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)

    corpus = read_jsonl(args.corpus)
    queries = read_jsonl(args.queries)

    # Build BM25 over corpus
    docs = [c["text"] for c in corpus]
    tokenized = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized)

    # Map pid -> text for quick lookup
    pid2text = {c["pid"]: c["text"] for c in corpus}

    # Retrieve topK per query, label with answer-matching if provided
    pointwise = []
    eval_candidates = []
    K = args.top_k

    for q in queries:
        qid = q["qid"]
        qtext = q["query"]
        answers = q.get("answers") or []

        scores = bm25.get_scores(tokenize(qtext))
        # topK indices
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:K]
        cand = []
        for i in idxs:
            passage = corpus[i]["text"]
            label = 1 if contains_answer(passage, answers) else 0
            # pointwise training lines
            pointwise.append({"query": qtext, "passage": passage, "label": label})
            cand.append({"passage": passage, "label": label})
        eval_candidates.append({"query": qtext, "candidates": cand})

    # Shuffle and split pointwise train/dev
    random.shuffle(pointwise)
    dev_size = max(1, int(len(pointwise) * args.dev_ratio))
    dev = pointwise[:dev_size]
    train = pointwise[dev_size:]

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(args.out_train, train)
    write_jsonl(args.out_dev, dev)
    with open(args.out_eval_cands, "w", encoding="utf-8") as f:
        json.dump(eval_candidates, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(train)} train, {len(dev)} dev, {len(eval_candidates)} eval queries.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--out_train", default="data/rerank_train.jsonl")
    ap.add_argument("--out_dev", default="data/rerank_dev.jsonl")
    ap.add_argument("--out_eval_cands", default="data/rerank_eval_candidates.json")
    args = ap.parse_args()
    main(args)

