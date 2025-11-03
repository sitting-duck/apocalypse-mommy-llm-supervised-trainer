#!/usr/bin/env python3
# scripts/inspect_corpus.py
import json, sys, argparse, re
from pathlib import Path

def clean_vis(s):
    # show control chars as hex so we can spot them
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', lambda m: f'<0x{ord(m.group(0)):02x}>', s[:600])

def token_count(s):
    return len(re.findall(r'\S+', s))

def main():
    p = Path(__file__).resolve().parents[1] / 'data' / 'corpus.jsonl'
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", default=str(p))
    parser.add_argument("--sample", "-n", type=int, default=5, help="how many examples to show")
    args = parser.parse_args()

    counts = []
    total = 0
    with open(args.file,'r',encoding='utf-8',errors='surrogateescape') as fh:
        lines = fh.readlines()
    print(f"Read {len(lines)} lines from {args.file}\n")
    for i,line in enumerate(lines):
        line=line.rstrip("\n")
        if not line: continue
        try:
            j=json.loads(line)
        except Exception as e:
            print(f"LINE {i+1}: JSON parse error: {e}")
            print("RAW:", clean_vis(line[:400]))
            continue
        txt=j.get("text","")
        tc = token_count(txt)
        counts.append(tc)
        total += tc
    if not counts:
        print("No usable entries found.")
        return
    import statistics as st
    print("Token stats (approx word tokens):")
    print(" count:", len(counts))
    print(" min:", min(counts))
    print(" 10p:", int(st.quantiles(counts, n=10)[0]))
    print(" median:", st.median(counts))
    print(" mean:", round(st.mean(counts),2))
    print(" 90p:", int(st.quantiles(counts, n=10)[-1]))
    print(" max:", max(counts))

    print("\nShow first", args.sample, "examples (with control chars highlighted):\n")
    for i,line in enumerate(lines[:args.sample]):
        try:
            j=json.loads(line)
        except:
            print(f"LINE {i+1} parse failed; raw:", clean_vis(line[:400]))
            continue
        print(f"--- example {i+1} --- pid={j.get('pid')} source={j.get('source')}")
        print(clean_vis(j.get("text","")))
        print(" tokens:", token_count(j.get("text","")))
        print()

if __name__=="__main__":
    main()

