#!/usr/bin/env python3
"""
sweep_threshold.py
- Load a CSV of model predictions (from your dev eval)
- Sweep probability thresholds in [0,1] and report the best F1
- Also prints accuracy, precision, recall at that threshold
- Optionally write a CSV of the whole sweep

Expected CSV columns (case-insensitive; flexible):
  - label / y / y_true        → ground-truth label in {0,1}
  - prob / probability        → model score in [0,1]
    OR
  - logit                     → model score in R (will be sigmoid'ed)

Usage:
  python eval/sweep_threshold.py \
    --preds eval/dev_preds_moredata.csv \
    --out   eval/threshold_sweep.csv

Options:
  --label_col    Custom label column name
  --score_col    Custom score column (prob or logit)
"""

import argparse
import csv
import math
import numpy as np
from collections import Counter

def sigmoid(x):
    try:
        # stable sigmoid
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def pick_colname(fieldnames, candidates):
    lower = {f.lower(): f for f in fieldnames}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None

def load_preds(path, label_col=None, score_col=None):
    ys, scores = [], []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None:
            raise ValueError(f"No header found in {path}")

        # Resolve label column
        if label_col is None:
            label_col = pick_colname(
                rdr.fieldnames,
                ["label", "y", "y_true"]
            )
        if label_col is None:
            raise ValueError(f"Could not find label column in {rdr.fieldnames}")

        # Resolve score column: prefer prob; else logit
        used_logit = False
        if score_col is None:
            score_col = pick_colname(
                rdr.fieldnames,
                ["prob", "probability"]
            )
            if score_col is None:
                score_col = pick_colname(rdr.fieldnames, ["logit"])
                if score_col is None:
                    raise ValueError(
                        f"Could not find a score column (prob/probability/logit) in {rdr.fieldnames}"
                    )
                used_logit = True

        for row in rdr:
            try:
                y = float(row[label_col])
            except Exception:
                continue
            try:
                s = float(row[score_col])
            except Exception:
                continue

            if used_logit:
                s = sigmoid(s)
            # clamp just in case
            s = 0.0 if s < 0 else 1.0 if s > 1 else s

            # labels to {0,1}
            y = 1.0 if y >= 0.5 else 0.0

            ys.append(y)
            scores.append(s)

    if not ys:
        raise ValueError("No usable rows found.")
    return np.array(ys, dtype=np.float32), np.array(scores, dtype=np.float32)

def confusion(y_true, y_prob, thr):
    y_hat = (y_prob >= thr).astype(np.float32)
    tp = float(((y_hat == 1) & (y_true == 1)).sum())
    fp = float(((y_hat == 1) & (y_true == 0)).sum())
    tn = float(((y_hat == 0) & (y_true == 0)).sum())
    fn = float(((y_hat == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn

def safe_div(n, d):
    return n / d if d else 0.0

def metrics(tp, fp, tn, fn):
    precision = safe_div(tp, tp + fp)
    recall    = safe_div(tp, tp + fn)
    f1        = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    acc       = safe_div(tp + tn, tp + fp + tn + fn)
    return acc, precision, recall, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Path to predictions CSV")
    ap.add_argument("--out", default=None, help="Optional: write full sweep CSV here")
    ap.add_argument("--label_col", default=None, help="Override label column name")
    ap.add_argument("--score_col", default=None, help="Override score (prob/logit) column name")
    ap.add_argument("--steps", type=int, default=201, help="Threshold steps in [0,1]")
    args = ap.parse_args()

    y_true, y_prob = load_preds(args.preds, args.label_col, args.score_col)

    # Basic class balance
    counts = Counter(y_true.tolist())
    n = len(y_true)
    pos = int(counts.get(1.0, 0))
    neg = int(counts.get(0.0, 0))
    print(f"[sweep] loaded {n} rows | pos={pos} neg={neg} | mean_prob={y_prob.mean():.3f}")

    best = {"thr": None, "f1": -1.0, "acc": None, "prec": None, "rec": None}
    rows = []
    for thr in np.linspace(0.0, 1.0, args.steps):
        tp, fp, tn, fn = confusion(y_true, y_prob, thr)
        acc, prec, rec, f1 = metrics(tp, fp, tn, fn)
        rows.append((thr, acc, prec, rec, f1, tp, fp, tn, fn))
        if f1 > best["f1"]:
            best = {"thr": thr, "f1": f1, "acc": acc, "prec": prec, "rec": rec}

    print(f"[sweep] Best F1={best['f1']:.3f} at thr={best['thr']:.3f} | "
          f"P={best['prec']:.3f} R={best['rec']:.3f} Acc={best['acc']:.3f}")

    if args.out:
        import csv
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["threshold","accuracy","precision","recall","f1","tp","fp","tn","fn"])
            for r in rows:
                w.writerow(r)
        print(f"[sweep] wrote sweep → {args.out}")

if __name__ == "__main__":
    main()

