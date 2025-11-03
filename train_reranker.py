#!/usr/bin/env python3
import os, json, argparse, math
from typing import List, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, CrossEncoder

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                print(f"[read_jsonl] skip line {i} (bad JSON): {e}")
    return items

def load_ce_pairs(path: str) -> List[InputExample]:
    """Each JSONL must have: {"query": str, "passage": str, "label": 0/1}"""
    print(f"[load_ce_pairs] Reading JSONL from: {path}")
    data = read_jsonl(path)
    print(f"[load_ce_pairs] Lines in file: {len(data)}")
    samples: List[InputExample] = []
    missing = badlabel = 0
    label_hist = {}
    for idx, r in enumerate(data):
        q = r.get("query")
        p = r.get("passage") or r.get("doc")  # be tolerant: allow 'doc'
        y = r.get("label")
        if q is None or p is None or y is None:
            missing += 1
            if idx < 5:
                print(f"[load_ce_pairs] Skip line {idx} (missing key). Keys present: {list(r.keys())}")
            continue
        try:
            y = float(y)
        except Exception:
            badlabel += 1
            continue
        label_hist[y] = label_hist.get(y, 0) + 1
        samples.append(InputExample(texts=[str(q), str(p)], label=y))
    print("\n[load_ce_pairs] ==== Summary ====")
    print(f"[load_ce_pairs] Total lines:        {len(data)}")
    print(f"[load_ce_pairs] Valid examples:     {len(samples)}")
    print(f"[load_ce_pairs] Skipped (missing):  {missing}")
    print(f"[load_ce_pairs] Skipped (badlabel): {badlabel}")
    print(f"[load_ce_pairs] Label counts: {label_hist}\n")
    return samples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", default="")
    ap.add_argument("--base", required=True, help="CrossEncoder base, e.g. cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--fp16", type=str, default="true")
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()

    args.fp16 = str(args.fp16).lower() in {"1", "true", "yes", "y"}

    os.makedirs(args.output_path, exist_ok=True)
    print(f"[train] output_path = {os.path.abspath(args.output_path)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] torch.cuda.is_available = {torch.cuda.is_available()} | device = {device}")
    if torch.cuda.is_available():
        print(f"[train] GPU = {torch.cuda.get_device_name(0)}")

    print(f"[train] loading base model: {args.base}")
    model = CrossEncoder(args.base, num_labels=1, max_length=args.max_length, device=device)

    print("[train] loading datasets…")
    train_samples = load_ce_pairs(args.train)
    if not train_samples:
        raise ValueError(f"No valid train examples found in {args.train}")

    train_loader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)
    steps_per_epoch = math.ceil(len(train_samples) / args.batch_size)
    print(f"[train] batches/epoch = {steps_per_epoch}")
    print(f"[train] epochs={args.epochs} bs={args.batch_size} lr={args.lr} warmup={args.warmup_steps} fp16={args.fp16}")

    # Use standard PyTorch BCE-with-logits for binary 0/1 labels
    loss_fct = nn.BCEWithLogitsLoss()

    # Keep it simple: train without evaluator (we can add later when your dev set is ready)
    model.fit(
        train_dataloader=train_loader,
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        warmup_steps=args.warmup_steps,
        output_path=args.output_path,
        scheduler="WarmupLinear",
        show_progress_bar=True,
        use_amp=args.fp16,          # mixed precision if supported
        loss_fct=loss_fct,
        save_best_model=True,
    )

    # Ensure final weights are on disk even without eval
    model.save(args.output_path)
    print(f"[train] DONE. Saved model to: {args.output_path}")

if __name__ == "__main__":
    main()

