#!/usr/bin/env python3
"""
Build hard negatives with BM25:
- Load corpus (pid,text)
- Load queries (qid,query) + positives file mapping qid->positive pid(s)
- For each query, retrieve top_k BM25 docs; treat non-positive hits as hard negatives.

Writes JSONL pairs: {"query":..., "passage":..., "label": 0/1}
"""
import argparse, json, re
from rank_bm25 import BM25Okapi
from pathlib import Path
from collections import defaultdict

def iter_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: yield json.loads(line)
            except: pass

def tokenize(t):
    return re.findall(r"[A-Za-z0-9]+", t.lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="JSONL with {'pid','text'}")
    ap.add_argument("--queries", required=True, help="JSONL with {'qid','query'}")
    ap.add_argument("--qpos", required=True, help="JSONL mapping {'qid','pid'} of known positives (one per line)")
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--k_neg", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Load corpus
    docs, pids, tok = [], [], []
    for row in iter_jsonl(args.corpus):
        pid = row.get("pid")
        text = row.get("text","").strip()
        if not pid or not text: continue
        pids.append(pid); docs.append(text); tok.append(tokenize(text))

    bm25 = BM25Okapi(tok)

    # positives per qid
    q2pos = defaultdict(set)
    for row in iter_jsonl(args.qpos):
        qid = row.get("qid"); pid=row.get("pid")
        if qid and pid: q2pos[qid].add(pid)

    out = []
    # Build pairs
    for row in iter_jsonl(args.queries):
        qid=row.get("qid"); q=row.get("query","").strip()
        if not qid or not q: continue
        pos_pids = q2pos.get(qid,set())
        # pos pairs
        for pid in pos_pids:
            # fetch text
            try:
                idx = pids.index(pid)
                out.append({"query": q, "passage": docs[idx], "label": 1})
            except ValueError:
                pass
        # retrieve candidates
        scores = bm25.get_scores(tokenize(q))
        # top_k indices
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.top_k]
        neg_added = 0
        for i in top_idx:
            if pids[i] in pos_pids: 
                continue
            out.append({"query": q, "passage": docs[i], "label": 0})
            neg_added += 1
            if neg_added >= args.k_neg: break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w",encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} pairs → {args.out}")

if __name__ == "__main__":
    main()

