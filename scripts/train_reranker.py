import os, argparse, yaml, random, numpy as np
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from src.data_utils import load_pointwise_jsonl

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def build_dataset(path):
    data = load_pointwise_jsonl(path)
    return [InputExample(texts=[d['query'], d['passage']], label=float(d['label'])) for d in data]

def main(cfg):
    set_seed(cfg.get('seed',42))
    mname = cfg['model']['name']
    out_dir = cfg['train']['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    model = CrossEncoder(mname, num_labels=1)
    train_ds = build_dataset(cfg['train']['train_path'])
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True)
    warmup_steps = int(len(train_loader)*cfg['train']['epochs']*cfg['train']['warmup_ratio'])

    model.fit(
        train_dataloader=train_loader,
        epochs=cfg['train']['epochs'],
        warmup_steps=warmup_steps,
        optimizer_params={'lr': float(cfg['train']['lr'])},
        output_path=out_dir,
        save_best_model=bool(cfg['train'].get('save_best', True))
    )
    print(f'Saved model to: {out_dir}')

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    args = ap.parse_args()
    with open(args.config,'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
