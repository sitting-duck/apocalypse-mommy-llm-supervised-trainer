#!/usr/bin/env python3
"""
crossencoder_dev_eval.py — Evaluate a saved CrossEncoder on a dev JSONL file.

Input JSONL lines must be:
  {"query": "...", "passage": "...", "label": 0 or 1}

Example:
  python eval/crossencoder_dev_eval.py \
    --model runs/modelA_miniLM_e3_bs8 \
    --dev data/rerank_dev.fixed.jsonl
or

python eval/crossencoder_dev_eval.py \
  --model runs/modelA_miniLM_e3_bs8 \
  --dev data/rerank_dev.fixed.jsonl \
  --out_csv eval/dev_preds.csv


Options:
  --threshold 0.5           Classification threshold on sigmoid(prob)
  --batch_size 64           Prediction batch size
  --out_csv preds.csv       (Optional) save per-example predictions
  --allow_hub               Allow downloading from HF Hub if --model isn't a local dir
"""

import argparse, json, math, os, sys, pathlib
from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import CrossEncoder

try:
    # Optional for extra metrics
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def read_jsonl(path: str) -> List[dict]:
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dev file not found: {p}")
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[warn] Could not parse JSON on line {i}: {e}", file=sys.stderr)
                continue
            if not {"query","passage","label"} <= set(obj.keys()):
                print(f"[warn] Missing keys on line {i}: have {list(obj.keys())}", file=sys.stderr)
                continue
            rows.append(obj)
    return rows


def batched(iterable, n):
    total = len(iterable)
    for i in range(0, total, n):
        yield iterable[i:i+n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to local model dir OR HF repo id")
    ap.add_argument("--dev",   required=True, help="Dev JSONL (query, passage, label)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out_csv", type=str, default=None, help="Optional: write predictions CSV")
    ap.add_argument("--allow_hub", action="store_true", help="Allow HF hub download if model path is not local")
    args = ap.parse_args()

    dev_rows = read_jsonl(args.dev)
    if not dev_rows:
        print("[error] No valid dev rows found.", file=sys.stderr)
        sys.exit(2)

    pairs: List[Tuple[str,str]] = [(r["query"], r["passage"]) for r in dev_rows]
    labels = np.array([int(r["label"]) for r in dev_rows], dtype=np.int64)

    model_path = args.model
    local_only = not args.allow_hub
    if os.path.isdir(model_path):
        local_only = True  # prefer local files if path exists

    print(f"[eval] device: {'cuda' if torch.cuda.is_available() else 'cpu'} | torch={torch.__version__}")
    print(f"[eval] loading model: {model_path} (local_files_only={local_only})")
    model = CrossEncoder(model_path, local_files_only=local_only)

    # Predict in batches to be RAM friendly
    all_logits: List[float] = []
    for chunk in batched(pairs, args.batch_size):
        logits = model.predict(chunk)
        all_logits.extend([float(x) for x in logits])

    logits = np.array(all_logits, dtype=np.float32)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= args.threshold).astype(np.int64)

    acc = float((preds == labels).mean())
    pos = int(labels.sum())
    neg = int((labels == 0).sum())

    print("\n[dev] ---- Summary ----")
    print(f"[dev] n={len(labels)} | pos={pos} | neg={neg}")
    print(f"[dev] accuracy={acc:.3f}")
    print(f"[dev] mean_logit={logits.mean():+.4f} | mean_prob={probs.mean():.3f} | threshold={args.threshold:.2f}")

    if HAVE_SK:
        try:
            auc = roc_auc_score(labels, probs)
            print(f"[dev] roc_auc={auc:.3f}")
        except Exception as e:
            print(f"[dev] roc_auc=NA ({e})")
        try:
            p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
            print(f"[dev] precision={p:.3f} recall={r:.3f} f1={f1:.3f}")
            cm = confusion_matrix(labels, preds)
            print(f"[dev] confusion_matrix=\n{cm}")
        except Exception as e:
            print(f"[dev] PRF/CM=NA ({e})")
    else:
        print("[dev] sklearn not installed; skipping ROC-AUC / PRF / confusion matrix.")

    # Show a few examples
    def preview(mask, title, k=5, reverse=True):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            print(f"\n[dev] {title}: none")
            return
        # sort by prob (descending for errors usually interesting)
        order = np.argsort(probs[idxs])
        if reverse:
            order = order[::-1]
        pick = idxs[order[:k]]
        print(f"\n[dev] {title}: showing up to {k} examples")
        for i in pick:
            q, psg, y, pr, lg = pairs[i][0], pairs[i][1], int(labels[i]), float(probs[i]), float(logits[i])
            print(f"  y={y} prob={pr:.3f} logit={lg:+.3f} | Q: {q[:60]} || P: {psg[:80]}")

    # False positives & false negatives
    preview((preds == 1) & (labels == 0), "Top false positives")
    preview((preds == 0) & (labels == 1), "Top false negatives")

    # Optional CSV
    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query","passage","label","logit","prob","pred"])
            for (q,p),y,l,pr,pe in zip(pairs, labels, logits, probs, preds):
                w.writerow([q, p, int(y), float(l), float(pr), int(pe)])
        print(f"\n[dev] wrote: {args.out_csv}")


if __name__ == "__main__":
    main()

