#!/usr/bin/env python3
import os, re, json, argparse, hashlib
from typing import Iterable, Tuple

# HTML parsing
from bs4 import BeautifulSoup

# PDF parsing (text-based PDFs only; OCR first if needed)
from pdfminer.high_level import extract_text

def clean_text(t: str) -> str:
    t = re.sub(r'\s+', ' ', t or '')
    return t.strip()

def chunk(text: str, max_tokens=180, min_tokens=25) -> Iterable[str]:
    words = text.split()
    cur = []
    for w in words:
        cur.append(w)
        if len(cur) >= max_tokens:
            c = ' '.join(cur).strip()
            if len(c.split()) >= min_tokens:
                yield c
            cur = []
    if cur:
        c = ' '.join(cur).strip()
        if len(c.split()) >= min_tokens:
            yield c

def html_to_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    return clean_text(soup.get_text(' '))

def pdf_to_text(path: str) -> str:
    try:
        txt = extract_text(path) or ''
        return clean_text(txt)
    except Exception as e:
        # If this hits, the PDF is likely scanned; OCR it, then retry.
        return ""

def is_html(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in {'.html', '.htm'}

def is_pdf(p: str) -> bool:
    return os.path.splitext(p)[1].lower() == '.pdf'

def iter_paths(src_root: str) -> Iterable[Tuple[str, str]]:
    """Yield (abs_path, rel_path) for html/pdf files."""
    for root, _, files in os.walk(src_root):
        for fn in files:
            p = os.path.join(root, fn)
            if is_pdf(p) or is_html(p):
                yield p, os.path.relpath(p, src_root)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Path to cloned Survival-Data repo')
    ap.add_argument('--out', default='data/corpus.jsonl')
    ap.add_argument('--max_tokens', type=int, default=180)
    ap.add_argument('--min_tokens', type=int, default=25)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    total_chunks = 0
    with open(args.out, 'w', encoding='utf-8') as out:
        for abs_p, rel_p in iter_paths(args.src):
            if is_pdf(abs_p):
                text = pdf_to_text(abs_p)
            else:
                text = html_to_text(abs_p)

            if not text:
                # Probably a scanned PDF without text
                # You can OCR and rerun if this happens often.
                continue

            for i, ck in enumerate(chunk(text, max_tokens=args.max_tokens, min_tokens=args.min_tokens)):
                pid = hashlib.md5(f'{rel_p}#{i}'.encode()).hexdigest()[:12]
                rec = {'pid': pid, 'source': rel_p, 'text': ck}
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                total_chunks += 1

    print(f'Wrote {total_chunks} chunks to {args.out}')

if __name__ == '__main__':
    main()

