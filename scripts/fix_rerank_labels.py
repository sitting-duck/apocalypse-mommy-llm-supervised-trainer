#!/usr/bin/env python3
"""
Rename 'passage' -> 'doc' and ensure each query has at least one positive.
Weak labeling heuristic:
- Tokenize query to keywords (minus short/common stopwords).
- Score each candidate doc by keyword overlap ratio.
- If a query has zero positives, flip the best-scoring doc to label=1.

Usage:
  python scripts/fix_rerank_labels.py \
    --in data/rerank_train.jsonl \
    --out data/rerank_train.fixed.jsonl

  python scripts/fix_rerank_labels.py \
    --in data/rerank_dev.jsonl \
    --out data/rerank_dev.fixed.jsonl
"""
import argparse, json, re, sys
from collections import defaultdict

STOP = {
    "the","a","an","and","or","of","to","for","with","in","on","at","by","from",
    "is","are","was","were","be","as","that","this","these","those","it","its",
    "how","what","when","where","which","who","whom","why","do","does","did",
    "can","could","should","would","you","your","me","my","we","our","they","their"
}

def toks(s):
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2 and t not in STOP]

def score_overlap(query, doc):
    q = set(toks(query))
    if not q: return 0.0
    d = set(toks(doc))
    if not d: return 0.0
    inter = len(q & d)
    return inter / len(q)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_path",  required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()

    # Load
    rows = []
    with open(args.in_path, "rb") as f:
        for ln in f:
            s = ln.replace(b"\x00", b"").replace(b"\xef\xbb\xbf", b"").decode("utf-8","ignore").strip()
            if not s: continue
            try:
                r = json.loads(s)
            except Exception:
                continue
            # Normalize keys
            if "doc" not in r:
                if "passage" in r: r["doc"] = r.pop("passage")
                elif "context" in r: r["doc"] = r.pop("context")
                elif "text" in r: r["doc"] = r.pop("text")
            if "query" not in r:
                if "q" in r: r["query"] = r.pop("q")
                elif "question" in r: r["query"] = r.pop("question")

            if not r.get("query") or not r.get("doc"):
                continue

            # Coerce label -> float 0/1 (default to 0)
            y = r.get("label", 0)
            try:
                y = float(y)
            except Exception:
                y = 1.0 if str(y).strip().lower() in {"1","true","yes"} else 0.0
            r["label"] = 1.0 if y > 0 else 0.0

            rows.append(r)

    # Group by query
    by_q = defaultdict(list)
    for i, r in enumerate(rows):
        by_q[r["query"]].append((i, r))

    # For any query with no positives, flip the best-overlap doc to positive
    flips = 0
    for q, pairs in by_q.items():
        if any(r["label"] > 0 for _, r in pairs):
            continue
        # pick best scoring doc
        best_i, best_s = None, -1.0
        for i, r in pairs:
            s = score_overlap(q, r["doc"])
            if s > best_s:
                best_s, best_i = s, i
        if best_i is not None:
            rows[best_i]["label"] = 1.0
            flips += 1

    # Write out
    wrote = 0
    with open(args.out_path, "w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps({"query": r["query"], "doc": r["doc"], "label": r["label"]}, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"Loaded {len(rows)} rows; flipped {flips} queries to have a positive; wrote {wrote} -> {args.out_path}")

if __name__ == "__main__":
    main()

