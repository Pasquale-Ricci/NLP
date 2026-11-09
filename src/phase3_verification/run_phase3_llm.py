"""
Phase 3a entry point: LLM (Llama) + Chain-of-Thought verification over Phase 2 evidence.

  # evidence = top-20 triples ranked by the 'cross' scorer of a Phase 2 run
  python run_phase3_llm.py --phase2 phase2_outputs/ft_seed.jsonl --scorer cross --n 20 \
      --out phase3_outputs/llm_cross20.jsonl

  # ablation: claim only (no evidence)
  python run_phase3_llm.py --phase2 phase2_outputs/ft_seed.jsonl --n 0 --out phase3_outputs/llm_claim_only.jsonl

  # oracle-ish upper bound: evidence ranked from GOLD seeds
  python run_phase3_llm.py --phase2 phase2_outputs/ft_gold.jsonl --scorer cross --n 20 --out ...

  python run_phase3_llm.py --summarize phase3_outputs/llm_cross20.jsonl

Each result is appended as soon as it is produced (--resume continues an interrupted run).
Unparsable answers count as WRONG in every metric.
"""

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict

from llm_verifier import ask, build_messages, parse_verdict
from triple_verbalizer import triple_text

REASONING = ("num1", "num2", "multi claim", "existence", "multi hop", "negation")


def gold_of(row):
    lab = row.get("label")
    lab = lab[0] if isinstance(lab, list) and lab else lab
    return "SUPPORTED" if lab else "REFUTED"


def f1_for(rows, cls):
    tp = sum(1 for r in rows if r["gold"] == cls and r["pred"] == cls)
    fp = sum(1 for r in rows if r["gold"] != cls and r["pred"] == cls)
    fn = sum(1 for r in rows if r["gold"] == cls and r["pred"] != cls)
    p = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    return p, r_, (2 * p * r_ / (p + r_) if p + r_ else 0.0)


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def show_errors(rows, limit):
    wrong = [r for r in rows if r["pred"] != r["gold"]]
    print(f"\n=== {min(limit, len(wrong))} of {len(wrong)} errors ===")
    for r in wrong[:limit]:
        print(f"- {r['claim']}\n    gold={r['gold']} pred={r['pred']} types={r.get('types')}")
        for f in (r.get("facts") or [])[:8]:
            print(f"      fact: {f}")
        print("      LLM:", (r.get("response") or "").replace("\n", " ")[-400:])


def summarize(rows):
    n = len(rows)
    if not n:
        print("no results")
        return
    acc = sum(r["pred"] == r["gold"] for r in rows) / n
    fails = sum(r["pred"] is None for r in rows)
    majority = max(Counter(r["gold"] for r in rows).values()) / n
    ps, rs, fs = f1_for(rows, "SUPPORTED")
    pr, rr, fr = f1_for(rows, "REFUTED")
    conf = Counter((r["gold"], r["pred"]) for r in rows)
    print("=" * 70)
    print(f"Phase 3a - LLM verification ({n} claims)")
    print("=" * 70)
    lo, hi = wilson(sum(r["pred"] == r["gold"] for r in rows), n)
    print(f"Accuracy {acc:.4f} (95% CI {lo:.3f}-{hi:.3f}) | macro-F1 {(fs + fr) / 2:.4f} | majority baseline {majority:.4f} | "
          f"unparsable {fails} ({100 * fails / n:.1f}%)")
    print(f"SUPPORTED  P={ps:.3f} R={rs:.3f} F1={fs:.3f}   REFUTED  P={pr:.3f} R={rr:.3f} F1={fr:.3f}")
    print("confusion (gold -> pred): " + ", ".join(
        f"{g[:3]}->{(p or 'FAIL')[:3]}: {c}" for (g, p), c in sorted(conf.items(), key=str)))
    by_type = defaultdict(list)
    for r in rows:
        for t in r.get("types", []):
            if t in REASONING:
                by_type[t].append(r["pred"] == r["gold"])
    print("accuracy by reasoning type: " + "  ".join(
        f"{t}={sum(v) / len(v):.3f}(n={len(v)})" for t, v in sorted(by_type.items())))
    secs = [r["sec"] for r in rows if "sec" in r]
    if secs:
        print(f"avg {sum(secs) / len(secs):.1f}s per claim")


def load_results(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description="FactKG Phase 3a - LLM-CoT verification")
    ap.add_argument("--phase2", help="Phase 2 .jsonl with ranked triples")
    ap.add_argument("--scorer", default="cross", help="which ranking to use from the Phase 2 file")
    ap.add_argument("--n", type=int, default=20, help="triples given to the LLM (0 = claim only)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-few-shot", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--out", help="results .jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summarize", default=None, help="only summarize an existing results .jsonl")
    ap.add_argument("--show-errors", type=int, default=0, help="print N wrongly answered claims with their facts")
    args = ap.parse_args()

    if args.summarize:
        rows_ = load_results(args.summarize)
        summarize(rows_)
        if args.show_errors:
            show_errors(rows_, args.show_errors)
        return
    if not (args.phase2 and args.out):
        raise SystemExit("--phase2 and --out are required (or use --summarize)")

    done = {}
    if os.path.exists(args.out):
        if args.resume:
            done = {r["claim"]: r for r in load_results(args.out)}
            print(f"resuming: {len(done)} claims already done")
        elif not args.overwrite:
            raise SystemExit(f"{args.out} exists: use --resume to continue or --overwrite")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.phase2, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if args.n > 0 and rows and args.scorer not in rows[0]["scorers"]:
        raise SystemExit(f"scorer '{args.scorer}' not in the Phase 2 file; available: "
                         f"{list(rows[0]['scorers'])}")

    results = list(done.values())
    mode = "a" if (args.resume and done) else "w"
    from tqdm import tqdm
    with open(args.out, mode, encoding="utf-8") as out:
        for row in tqdm(rows, desc="LLM verification"):
            if row["claim"] in done:
                continue
            triples = []
            if args.n > 0:
                triples = [tuple(x[:3]) for x in row["scorers"][args.scorer]["ranked"][: args.n]]
            t0 = time.time()
            text = ask(build_messages(row["claim"], triples, not args.no_few_shot),
                       max_tokens=args.max_tokens)
            res = {"claim": row["claim"], "gold": gold_of(row), "types": row.get("types", []),
                   "pred": parse_verdict(text), "n_facts": len(triples),
                   "facts": [triple_text(*t) for t in triples],
                   "response": text, "sec": round(time.time() - t0, 2)}
            out.write(json.dumps(res, ensure_ascii=False) + "\n")
            out.flush()
            results.append(res)
    summarize(results)
    if args.show_errors:
        show_errors(results, args.show_errors)


if __name__ == "__main__":
    main()
