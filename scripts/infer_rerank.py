#python scripts/infer_rerank.py \
#  --cfg configs/inference.json \
#  --profile bm25neg \
#  --input - <<'JSON'
#{
#  "query": "how to purify water with household bleach",
#  "candidates": [
#    "Use 6–8 drops of unscented household bleach (5–6% sodium hypochlorite) per gallon...",
#    "This chapter discusses backpack repair and tent sewing patterns..."
#  ]
#}
#JSON




#!/usr/bin/env python3
import argparse, json, sys, math
from typing import Any, Dict, Tuple
from sentence_transformers import CrossEncoder

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_cfg(cfg_raw: Any) -> Dict[str, Dict[str, Any]]:
    """
    Normalize to: { "<profile>": {"model_path": "...", "threshold": float, ...}, ... }

    Supports:
      - { "profiles": {...}, "default": "<name>" }   <-- your file
      - { "profiles": {...}, "default": {...} }
      - flat dict-of-profiles
      - single dict (becomes {"default": ...})
      - list of {name: "...", ...}
    """
    # Wrapper with "profiles"
    if isinstance(cfg_raw, dict) and "profiles" in cfg_raw:
        profiles = cfg_raw.get("profiles") or {}
        if not isinstance(profiles, dict):
            raise ValueError("'profiles' must be an object mapping names to configs")
        flat = dict(profiles)  # copy
        default_val = cfg_raw.get("default")
        if isinstance(default_val, dict) and "model_path" in default_val:
            flat["default"] = default_val
        elif isinstance(default_val, str):
            if default_val not in profiles:
                avail = ", ".join(sorted(profiles.keys()))
                raise KeyError(f"default '{default_val}' not found. Available: {avail}")
            flat["default"] = profiles[default_val]
        return flat

    # Flat dict-of-profiles (no top-level model_path)
    if isinstance(cfg_raw, dict) and "model_path" not in cfg_raw:
        return cfg_raw

    # Single dict -> {"default": ...}
    if isinstance(cfg_raw, dict) and "model_path" in cfg_raw:
        return {"default": cfg_raw}

    # List of {name: "...", ...}
    if isinstance(cfg_raw, list):
        out = {}
        for item in cfg_raw:
            if isinstance(item, dict) and "name" in item:
                name = str(item["name"])
                out[name] = {k: v for k, v in item.items() if k != "name"}
        if out:
            return out

    raise ValueError("Unsupported configs/inference.json format.")

def pick_profile(cfg_map: Dict[str, Dict[str, Any]], requested: str | None) -> Tuple[str, Dict[str, Any]]:
    if requested:
        if requested not in cfg_map:
            available = ", ".join(sorted(cfg_map.keys()))
            raise KeyError(f"Profile '{requested}' not found. Available: {available}")
        return requested, cfg_map[requested]
    if "default" in cfg_map:
        return "default", cfg_map["default"]
    # fallback to first key
    name, cfg = next(iter(cfg_map.items()))
    return name, cfg

def read_inputs(path: str):
    if path == "-" or path == "/dev/stdin":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def main():
    ap = argparse.ArgumentParser(description="Cross-Encoder rerank inference with profiles.")
    ap.add_argument("--input", default="-", help="JSON: {'query': str, 'candidates': [str,...]}. '-' for stdin.")
    ap.add_argument("--cfg", default="configs/inference.json", help="Path to inference config JSON.")
    ap.add_argument("--profile", default=None, help="Profile name (e.g., 'bm25neg'). If omitted, uses 'default' from cfg.")
    ap.add_argument("--top_k", type=int, default=10, help="Show top-K in preview.")
    ap.add_argument("--local_only", action="store_true", help="Force local files only.")
    args = ap.parse_args()

    cfg_raw = load_json(args.cfg)
    cfg_map = normalize_cfg(cfg_raw)
    prof_name, prof = pick_profile(cfg_map, args.profile)

    model_path = prof.get("model_path")
    if not model_path:
        raise ValueError(f"Profile '{prof_name}' is missing 'model_path'.")
    threshold = float(prof.get("threshold", 0.5))

    model = CrossEncoder(model_path, local_files_only=(args.local_only or True))

    data = read_inputs(args.input)
    q = (data.get("query") or "").strip()
    if not q:
        raise ValueError("Input JSON missing non-empty 'query'.")
    cands_raw = data.get("candidates") or []
    cands = [c.strip() for c in cands_raw if isinstance(c, str) and c.strip()]
    if not cands:
        raise ValueError("Input JSON 'candidates' must be a non-empty list of strings.")

    pairs = [(q, c) for c in cands]
    logits = model.predict(pairs)
    probs = [sigmoid(float(z)) for z in logits]

    scored = sorted(
        [{"passage": c, "logit": float(z), "prob": float(p)} for c, z, p in zip(cands, logits, probs)],
        key=lambda r: r["prob"],
        reverse=True,
    )
    kept = [r for r in scored if r["prob"] >= threshold]

    out = {
        "profile": prof_name,
        "model_path": model_path,
        "threshold": threshold,
        "query": q,
        "top": scored[: args.top_k],
        "kept": kept,
    }
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

