i#!/usr/bin/env python3
"""
make_queries_from_corpus.py
- Read a cleaned corpus JSONL (each line: {"pid": "...", "text": "...", ...})
- Synthesize diverse, realistic search queries from the text
- Deduplicate, shuffle, and cap to a target size
- Write JSONL lines: {"qid": "q000001", "query": "..."}

Usage:
  python scripts/make_queries_from_corpus.py \
    --corpus data/corpus_clean.jsonl \
    --out data/queries.jsonl \
    --max_per_doc 5 \
    --target_total 3000
"""

import argparse
import json
import random
import re
from pathlib import Path

HDR_RE = re.compile(r"^\s*(chapter|section|appendix|figure|table)\b[^A-Za-z0-9]*", re.I)
HOWTO_RE = re.compile(r"\b(how to|how do i|how can i|what is|what are|when should|where can|best way to)\b", re.I)

def iter_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip malformed
                continue

def first_sentence(txt, max_words=14):
    # Very simple first-sentence extractor
    sent = re.split(r"(?<=[.!?])\s+", txt.strip(), 1)[0]
    words = sent.split()
    return " ".join(words[:max_words])

def normalize_text(t: str) -> str:
    # Keep common punctuation; collapse whitespace
    t = re.sub(r"[^A-Za-z0-9 ,:;?!'\"/\-\(\)]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def make_candidates(text: str):
    qs = set()
    lines = [l.strip() for l in text.splitlines() if l and l.strip()]

    # 1) Lines that already look like questions/“how to …”
    for l in lines:
        if HOWTO_RE.search(l) and len(l.split()) >= 4:
            q = normalize_text(l)
            if q:
                qs.add(q.lower())

    # 2) Short, header-ish lines → turn into queries
    for l in lines:
        if (len(l) <= 100 and not l.endswith(":")
            and not l.isupper()
            and not HDR_RE.match(l)
            and 3 <= len(l.split()) <= 12
            and l[0].isalpha()):
            base = l.lower().strip()
            qs.add(normalize_text(f"what to know about {base}"))
            qs.add(normalize_text(f"how to {base}"))

    # 3) First sentence prompt
    if text:
        fs = first_sentence(text, 12).lower()
        if len(fs.split()) >= 4:
            qs.add(normalize_text(f"explain {fs}"))

    # Final length filter
    out = []
    for q in qs:
        q = q.strip(" -:;,.")
        if 4 <= len(q.split()) <= 16:
            out.append(q)
    return out

def main(corpus_path, out_path, max_per_doc=5, target_total=5000, seed=13):
    random.seed(seed)
    seen = set()
    all_qs = []

    for row in iter_jsonl(corpus_path):
        text = row.get("text", "")
        if not text:
            continue
        cands = make_candidates(text)
        random.shuffle(cands)
        kept = 0
        for q in cands:
            if q in seen:
                continue
            seen.add(q)
            all_qs.append(q)
            kept += 1
            if kept >= max_per_doc:
                break

    random.shuffle(all_qs)
    if target_total and len(all_qs) > target_total:
        all_qs = all_qs[:target_total]

    # Write with qid field
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, q in enumerate(all_qs, 1):
            row = {"qid": f"q{i:06d}", "query": q}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_qs)} queries with qid → {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_clean.jsonl")
    ap.add_argument("--out", default="data/queries.jsonl")
    ap.add_argument("--max_per_doc", type=int, default=5)
    ap.add_argument("--target_total", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    main(args.corpus, args.out, args.max_per_doc, args.target_total, args.seed)

