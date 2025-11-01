import json, math

def load_pointwise_jsonl(path: str):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            data.append(json.loads(line))
    return data

def load_eval_candidates(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def mrr_at_k(labels, k):
    for i, lab in enumerate(labels[:k], start=1):
        if lab==1:
            return 1.0/i
    return 0.0

def _dcg(labels, k):
    s=0.0
    for i, lab in enumerate(labels[:k], start=1):
        gain = 1.0 if lab==1 else 0.0
        s += gain / math.log2(i+1)
    return s

def ndcg_at_k(labels, k):
    actual = _dcg(labels, k)
    ideal = _dcg(sorted(labels, reverse=True), k)
    return float(actual / (ideal if ideal>0 else 1e-9))

def recall_at_k(labels, k):
    return 1.0 if 1 in labels[:k] else 0.0
