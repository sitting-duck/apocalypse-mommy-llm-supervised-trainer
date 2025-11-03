import argparse, json, yaml
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

def main(cfg):
    with open(cfg['eval']['out_json'],'r') as f:
        metrics = json.load(f)
    df = pd.read_csv(cfg['eval']['out_scores_csv'])

    # Plot 1: MRR bar
    plt.figure()
    plt.bar(['Reranked'], [metrics.get('MRR@10', 0.0)])
    plt.ylabel('MRR@10'); plt.title('MRR@10')
    plt.tight_layout(); plt.savefig(cfg['plots']['mrr_bar_path']); plt.close()

    # Plot 2: nDCG@k curve
    ks = list(range(1, 11)); ndcgs = []
    for K in ks:
        vals=[]
        for _, g in df.groupby('example_id'):
            g = g.sort_values('score', ascending=False)
            labels = g['label'].tolist()
            def dcg(labels, k):
                s=0.0
                for i, lab in enumerate(labels[:k], start=1):
                    gain=1.0 if lab==1 else 0.0
                    s += gain/np.log2(i+1)
                return s
            idcg = dcg(sorted(labels, reverse=True), K)
            vals.append(dcg(labels,K)/(idcg if idcg>0 else 1e-9))
        ndcgs.append(np.mean(vals) if vals else 0.0)

    plt.figure()
    plt.plot(ks, ndcgs, marker='o')
    plt.xlabel('k'); plt.ylabel('nDCG@k'); plt.title('nDCG@k Curve')
    plt.tight_layout(); plt.savefig(cfg['plots']['ndcg_curve_path']); plt.close()

    # Plot 3: score hist pos vs neg
    plt.figure()
    pos = df[df['label']==1]['score'].values
    neg = df[df['label']==0]['score'].values
    plt.hist(pos, bins=30, alpha=0.6, label='Positive')
    plt.hist(neg, bins=30, alpha=0.6, label='Negative')
    plt.xlabel('Score'); plt.ylabel('Count'); plt.title('Score Distribution')
    plt.legend(); plt.tight_layout(); plt.savefig(cfg['plots']['score_hist_path']); plt.close()

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    args = ap.parse_args()
    with open(args.config,'r') as f: cfg = yaml.safe_load(f)
    main(cfg)
