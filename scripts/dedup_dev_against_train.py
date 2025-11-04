import json, sys, re, os
from pathlib import Path

TRAIN_PATH = "data/rerank_train_bm25.train.jsonl"
DEV_IN     = "data/rerank_train_bm25.dev.jsonl"
DEV_OUT    = "data/rerank_train_bm25.dev.clean.jsonl"

def norm_text(t: str) -> str:
    if not isinstance(t, str): t = "" if t is None else str(t)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def load_pairs(path):
    seen = set()
    bad, miss = 0, 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            q = d.get("query"); p = d.get("passage")
            if q is None or p is None:
                miss += 1
                continue
            seen.add( (norm_text(q), norm_text(p)) )
    return seen, bad, miss

def main():
    for p in [TRAIN_PATH, DEV_IN]:
        if not Path(p).exists():
            print(f"ERROR: missing file: {p}", file=sys.stderr); sys.exit(1)

    print(f"[dedup] loading train pairs from: {TRAIN_PATH}")
    train_pairs, bad_t, miss_t = load_pairs(TRAIN_PATH)
    print(f"[dedup] train unique pairs: {len(train_pairs)} | bad_json={bad_t} | missing_fields={miss_t}")

    kept = removed = bad = miss = 0
    Path(DEV_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(DEV_IN, "r", encoding="utf-8") as fin, open(DEV_OUT, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            s = line.strip()
            if not s: continue
            try:
                d = json.loads(s)
            except json.JSONDecodeError:
                bad += 1
                continue
            q = d.get("query"); p = d.get("passage")
            if q is None or p is None:
                miss += 1
                continue
            key = (norm_text(q), norm_text(p))
            if key in train_pairs:
                removed += 1
                continue
            fout.write(line)
            kept += 1

    print(f"[dedup] dev in: {DEV_IN}")
    print(f"[dedup] wrote cleaned dev → {DEV_OUT}")
    print(f"[dedup] kept={kept}  removed_overlaps={removed}  bad_json={bad}  missing_fields={miss}")

if __name__ == "__main__":
    main()
