#!/usr/bin/env python3
"""
make_rerank_pairs_from_pid.py
- Guarantees 1 positive per query, plus k random negatives
- Splits into train/dev

Writes JSONL lines like:
{"query": "...", "passage": "...", "label": 1 or 0}

eg:
    python scripts/make_rerank_pairs_from_pid.py \
  --corpus data/corpus_clean.jsonl \
  --queries data/queries.jsonl \
  --out_train data/rerank_train.jsonl \
  --out_dev   data/rerank_dev.jsonl \
  --dev_ratio 0.10 \
  --k_neg 5

"""

import argparse, json, random
from pathlib import Path

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except:
                continue

def main(corpus_path, queries_path, out_train, out_dev, dev_ratio=0.1, k_neg=5, seed=13, max_passage_len=None):
    random.seed(seed)

    # 1) Load corpus into pid -> text
    pid2text = {}
    corpus = list(read_jsonl(corpus_path))
    for r in corpus:
        pid = r.get("pid"); txt = r.get("text","")
        if pid and txt:
            pid2text[pid] = txt
    pids = list(pid2text.keys())
    if not pids:
        raise SystemExit("No corpus loaded (missing pid/text).")

    # 2) Load queries with pid
    queries = [r for r in read_jsonl(queries_path) if r.get("pid") and r.get("query")]
    if not queries:
        raise SystemExit("No queries with pid found. Did you rerun make_queries_from_corpus.py?")

    # 3) Build pairs
    pairs = []
    for q in queries:
        qtxt = q["query"]
        qpid = q["pid"]
        # positive
        pos_txt = pid2text.get(qpid)
        if not pos_txt:
            continue
        if max_passage_len:
            pos_txt = pos_txt[:max_passage_len]
        pairs.append({"query": qtxt, "passage": pos_txt, "label": 1})

        # negatives
        need = k_neg
        while need > 0:
            neg_pid = random.choice(pids)
            if neg_pid == qpid:
                continue
            ntxt = pid2text[neg_pid]
            if max_passage_len:
                ntxt = ntxt[:max_passage_len]
            pairs.append({"query": qtxt, "passage": ntxt, "label": 0})
            need -= 1

    random.shuffle(pairs)

    # 4) Split
    n = len(pairs)
    n_dev = int(round(n * dev_ratio))
    dev = pairs[:n_dev]
    train = pairs[n_dev:]

    # 5) Write
    Path(out_train).parent.mkdir(parents=True, exist_ok=True)
    with open(out_train, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dev, "w", encoding="utf-8") as f:
        for r in dev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 6) Quick stats
    def stats(path):
        from collections import Counter
        c=Counter()
        n=0
        for r in read_jsonl(path):
            c[r["label"]]+=1; n+=1
        return n, dict(c)

    ntr, ctr = stats(out_train)
    ndv, cdv = stats(out_dev)
    print(f"[done] wrote {out_train}: n={ntr} counts={ctr}")
    print(f"[done] wrote {out_dev}:   n={ndv} counts={cdv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_clean.jsonl")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--out_train", default="data/rerank_train.jsonl")
    ap.add_argument("--out_dev", default="data/rerank_dev.jsonl")
    ap.add_argument("--dev_ratio", type=float, default=0.10)
    ap.add_argument("--k_neg", type=int, default=5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_passage_len", type=int, default=None, help="optional char cap")
    args = ap.parse_args()
    main(args.corpus, args.queries, args.out_train, args.out_dev,
         args.dev_ratio, args.k_neg, args.seed, args.max_passage_len)

