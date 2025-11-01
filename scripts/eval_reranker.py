import os, argparse, json, yaml
import numpy as np, pandas as pd
from sentence_transformers import CrossEncoder
from src.data_utils import load_eval_candidates, mrr_at_k, ndcg_at_k, recall_at_k

def main(cfg):
    model_dir = cfg['train']['output_dir']
    model = CrossEncoder(model_dir)

    data = load_eval_candidates(cfg['eval']['candidates_path'])
    mrrs, ndcgs, recalls = [], [], []
    rows = []

    for ex_id, ex in enumerate(data):
        q = ex['query']
        labels = [int(c['label']) for c in ex['candidates']]
        pairs = [[q, c['passage']] for c in ex['candidates']]
        scores = model.predict(pairs).tolist()

        for i, c in enumerate(ex['candidates']):
            rows.append({'example_id': ex_id, 'query': q, 'passage': c['passage'], 'label': int(c['label']), 'score': float(scores[i])})

        ranked = [lab for _, lab in sorted(zip(scores, labels), reverse=True)]
        mrrs.append(mrr_at_k(ranked, int(cfg['eval']['mrr_k'])))
        ndcgs.append(ndcg_at_k(ranked, int(cfg['eval']['ndcg_k'])))
        recalls.append(recall_at_k(ranked, int(cfg['eval']['recall_k'])))

    metrics = {
        f"MRR@{cfg['eval']['mrr_k']}": float(np.mean(mrrs)),
        f"nDCG@{cfg['eval']['ndcg_k']}": float(np.mean(ndcgs)),
        f"Recall@{cfg['eval']['recall_k']}": float(np.mean(recalls)),
        "num_queries": len(data)
    }
    print(json.dumps(metrics, indent=2))

    os.makedirs(os.path.dirname(cfg['eval']['out_json']), exist_ok=True)
    with open(cfg['eval']['out_json'], 'w') as f: json.dump(metrics, f, indent=2)
    pd.DataFrame(rows).to_csv(cfg['eval']['out_scores_csv'], index=False)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    args = ap.parse_args()
    with open(args.config,'r') as f: cfg = yaml.safe_load(f)
    main(cfg)
