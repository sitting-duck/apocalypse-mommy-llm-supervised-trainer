#!/usr/bin/env python3
# scripts/clean_rechunk_corpus.py
import json, re, argparse, hashlib
from pathlib import Path

BOM_RE = re.compile(r'^\ufeff|\ufeff')   # capture BOM characters
CTRL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')  # common nuisance control chars
MULTI_WS = re.compile(r'\s+')

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace('\r',' ')
    s = BOM_RE.sub('', s)
    s = CTRL_RE.sub(' ', s)
    s = MULTI_WS.sub(' ', s)
    return s.strip()

def sentence_split(text: str):
    # Simple sentence splitter (keeps punctuation). Good enough for chunking.
    pieces = re.split(r'(?<=[\.\!\?])\s+', text)
    out = [p.strip() for p in pieces if p and len(p.strip())>20]
    return out

def chunk_text(sentences, max_tokens=180, min_tokens=20):
    chunks = []
    cur = []
    cur_tokens = 0
    for s in sentences:
        toks = len(s.split())
        if cur_tokens + toks <= max_tokens:
            cur.append(s)
            cur_tokens += toks
        else:
            if cur and cur_tokens >= min_tokens:
                chunks.append(' '.join(cur))
            # if the single sentence is huge, split by commas as fallback
            if toks > max_tokens:
                parts = [p.strip() for p in re.split(r',\s*', s) if p.strip()]
                for p in parts:
                    if len(p.split()) <= max_tokens:
                        chunks.append(p)
                    else:
                        # hard-break large piece
                        words = p.split()
                        for i in range(0, len(words), max_tokens):
                            chunks.append(' '.join(words[i:i+max_tokens]))
                cur = []
                cur_tokens = 0
            else:
                cur = [s]
                cur_tokens = toks
    if cur and cur_tokens >= min_tokens:
        chunks.append(' '.join(cur))
    return chunks

def make_pid(source, idx, salt=0):
    return hashlib.md5(f"{source}#{idx}#{salt}".encode()).hexdigest()[:12]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data/corpus.jsonl")
    ap.add_argument("--outfile", default="data/corpus_clean.jsonl")
    ap.add_argument("--max_tokens", type=int, default=180)
    ap.add_argument("--min_tokens", type=int, default=25)
    args = ap.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    skipped = 0
    with infile.open('r', encoding='utf-8', errors='surrogateescape') as inf, outfile.open('w', encoding='utf-8') as outf:
        for line in inf:
            total_in += 1
            try:
                rec = json.loads(line)
            except Exception:
                skipped += 1
                continue
            text = normalize_text(rec.get("text",""))
            if len(text) < 30:
                skipped += 1
                continue
            # If the chunk looks list-like (many short items separated by commas or newlines),
            # use comma/newline split as sentence fallback.
            if text.count(',') > len(text.split()) * 0.1 or '\n' in text[:300]:
                # split on commas/newlines then group
                items = [i.strip() for i in re.split(r'[\n,;•\u2022]+', text) if len(i.strip())>10]
                # turn items into pseudo-sentences
                sentences = items
            else:
                sentences = sentence_split(text)
                if not sentences:
                    sentences = [text]

            chunks = chunk_text(sentences, max_tokens=args.max_tokens, min_tokens=args.min_tokens)
            if not chunks:
                # last resort: break into fixed token windows
                words = text.split()
                for i in range(0, len(words), args.max_tokens):
                    piece = ' '.join(words[i:i+args.max_tokens])
                    if len(piece.split()) >= args.min_tokens:
                        chunks.append(piece)

            for i,ch in enumerate(chunks):
                pid = make_pid(rec.get("source",""), i)
                outrec = {"pid": pid, "source": rec.get("source",""), "text": ch}
                outf.write(json.dumps(outrec, ensure_ascii=False) + "\n")
                total_out += 1

    print(f"Processed {total_in} input lines -> wrote {total_out} output chunks. skipped {skipped}.")

if __name__=="__main__":
    main()

