#!/usr/bin/env python3
"""
Make query_positives.jsonl by mining likely positives from the corpus with BM25
+ a simple keyword-overlap heuristic.

Inputs:
  --corpus  data/corpus_clean.jsonl     (each line: {"pid": "...", "text": "...", ...})
  --queries data/queries.jsonl          (each line: {"qid": "...", "query": "..."})
Outputs:
  --out     data/query_positives.jsonl  (each line: {"qid": "...", "pid": "..."})

Heuristics:
- BM25 rank top_k candidates for each query
- Pick the first whose content-word overlap >= min_overlap (default 3)
- If none meet overlap, still take BM25 rank #1 (optional via --fallback_top1)

You can increase strictness (min_overlap 4–5) for higher precision but fewer matches.
"""

import argparse, json, re
from pathlib import Path
from rank_bm25 import BM25Okapi

def iter_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: yield json.loads(line)
            except: pass

WORD_RE = re.compile(r"[A-Za-z0-9]+")

def words(t):
    return [w for w in WORD_RE.findall(t.lower()) if len(w) > 2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--min_overlap", type=int, default=3)
    ap.add_argument("--fallback_top1", action="store_true",
                    help="If no candidate meets overlap, still take BM25 #1.")
    args = ap.parse_args()

    # Load corpus
    pids, docs, tok = [], [], []
    for row in iter_jsonl(args.corpus):
        pid = row.get("pid")
        text = (row.get("text") or "").strip()
        if not pid or not text: continue
        pids.append(pid); docs.append(text); tok.append(words(text))

    bm25 = BM25Okapi(tok)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_q, n_hit, n_fb = 0, 0, 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for row in iter_jsonl(args.queries):
            qid = row.get("qid"); q = (row.get("query") or "").strip()
            if not qid or not q: continue
            n_q += 1

            qtok = words(q)
            scores = bm25.get_scores(qtok)
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.top_k]

            # choose first that meets overlap
            picked = None
            qset = set(qtok)
            for i in top_idx:
                overlap = len(qset.intersection(tok[i]))
                if overlap >= args.min_overlap:
                    picked = i
                    break

            if picked is None and args.fallback_top1 and top_idx:
                picked = top_idx[0]
                n_fb += 1

            if picked is not None:
                n_hit += 1
                rec = {"qid": qid, "pid": pids[picked]}
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[positives] queries={n_q} with_positive={n_hit} fallback_top1={n_fb} → {args.out}")

if __name__ == "__main__":
    main()

