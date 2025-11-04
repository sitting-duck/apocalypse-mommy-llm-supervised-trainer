#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
from sentence_transformers import CrossEncoder

def load_cfg(p="configs/inference.json"):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def read_inputs(path):
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="-", help="JSON with {'query': str, 'candidates': [str,...]}. Use '-' for stdin.")
    ap.add_argument("--cfg", default="configs/inference.json")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    model = CrossEncoder(cfg["model_path"], local_files_only=True)

    data = read_inputs(args.input)
    q = data["query"].strip()
    cands = [c.strip() for c in data["candidates"] if c and c.strip()]
    pairs = [(q, c) for c in cands]

    logits = model.predict(pairs)          # raw scores (logits)
    # Convert logits → probability via sigmoid
    import math
    probs = [1/(1+math.exp(-float(z))) for z in logits]

    scored = sorted(
        [{"passage": c, "logit": float(z), "prob": float(p)} for c, z, p in zip(cands, logits, probs)],
        key=lambda r: r["prob"],
        reverse=True
    )
    # Filter with your chosen operating threshold
    keep = [r for r in scored if r["prob"] >= float(cfg["threshold"])]

    out = {
        "query": q,
        "threshold": cfg["threshold"],
        "top": scored[:10],      # top-10 preview
        "kept": keep
    }
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

