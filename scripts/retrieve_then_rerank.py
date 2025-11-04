#!/usr/bin/env python3
import argparse, json, sys, re
from pathlib import Path
from collections import namedtuple
from sentence_transformers import CrossEncoder

# ---- tiny BM25 (rank_bm25) fallback impl using a simple scoring ----
# For quick smoke tests; good enough to produce candidates for reranking.
def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]

def bm25_topk(query, docs, k=50):
    # super-light TF scoring (idf-ish dampening); replace with rank_bm25 later if you like
    qt = set(simple_tokenize(query))
    scored = []
    for pid, text in docs:
        toks = set(simple_tokenize(text))
        overlap = len(qt & toks)
        if overlap > 0:
            scored.append((overlap / max(4, len(qt)), pid, text))
    scored.sort(reverse=True)
    return [(pid, text, score) for score, pid, text in scored[:k]]

def load_corpus(corpus_path, max_docs=None):
    rows = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                d = json.loads(line)
            except Exception:
                continue
            pid = d.get("pid") or f"doc{i}"
            txt = d.get("text") or ""
            if not txt.strip():
                continue
            rows.append((pid, txt))
            if max_docs and len(rows) >= max_docs:
                break
    return rows

def pick_profile(cfg, name=None):
    if isinstance(cfg, dict) and "profiles" in cfg and "default" in cfg:
        profiles = cfg["profiles"]
        if name is None:
            name = cfg["default"]
        if name not in profiles:
            raise KeyError(f"Profile '{name}' not found. Available: {', '.join(profiles.keys())}")
        prof = profiles[name]
        return name, prof["model_path"], float(prof["threshold"])
    else:
        # legacy single-profile file
        return "default", cfg["model_path"], float(cfg["threshold"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/inference.json")
    ap.add_argument("--profile", default=None, help="profile name in inference.json (defaults to 'default')")
    ap.add_argument("--corpus", default="data/corpus_clean.jsonl")
    ap.add_argument("--max_docs", type=int, default=100000, help="cap corpus docs for quick tests")
    ap.add_argument("--top_k", type=int, default=50, help="BM25 candidate count")
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", default="-", help="write JSON output (or '-' for stdout)")
    args = ap.parse_args()

    cfg = json.load(open(args.cfg))
    prof_name, model_path, thr = pick_profile(cfg, args.profile)

    print(f"[pipeline] profile={prof_name} model={model_path} thr={thr}")
    print(f"[pipeline] loading corpus: {args.corpus} (max_docs={args.max_docs})")
    corpus = load_corpus(args.corpus, max_docs=args.max_docs)
    print(f"[pipeline] corpus docs: {len(corpus)}")

    # Retrieve candidates
    cands = bm25_topk(args.query, corpus, k=args.top_k)
    print(f"[pipeline] bm25 candidates: {len(cands)}")

    # Rerank with CrossEncoder
    model = CrossEncoder(model_path, local_files_only=True)
    pairs = [(args.query, text) for (_, text, _) in cands]
    logits = model.predict(pairs)

    import math
    probs = [1/(1+math.exp(-float(z))) for z in logits]

    scored = [
        {"pid": pid, "passage": text, "bm25": bm, "logit": float(z), "prob": float(p)}
        for (pid, text, bm), z, p in zip(cands, logits, probs)
    ]
    scored.sort(key=lambda r: r["prob"], reverse=True)

    kept = [r for r in scored if r["prob"] >= thr]

    out = {
        "profile": prof_name,
        "query": args.query,
        "threshold": thr,
        "top": scored[:10],
        "kept": kept[:10],  # clip for display
        "counts": {"retrieved": len(cands), "kept": len(kept)}
    }
    s = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out == "-":
        print(s)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(s, encoding="utf-8")
        print(f"[pipeline] wrote → {args.out}")

if __name__ == "__main__":
    main()

