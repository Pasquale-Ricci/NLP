"""
Batch adversarial evaluation (ASR and delta-F1) of the full pipeline.

  python run_adversarial.py --phase2 phase2_outputs/ft_seed.jsonl --phase3 phase3_outputs/llm_seed_cross20.jsonl \
      --limit 100 --audit 30 --out adversarial_outputs/run1.jsonl

  # after filling the 'valid' (1/0) and optional 'label' columns of run1.jsonl.audit.csv:
  python run_adversarial.py --report adversarial_outputs/run1.jsonl --validated adversarial_outputs/run1.jsonl.audit.csv

Sources: labelled claims of a Phase 2 file. 'linguistic' keeps the label (applied to all
sources); 'semantic' and 'multihop' are applied to SUPPORTED sources and are ASSUMED to give
REFUTED claims (use the audit to check). ASR = share of attacks whose verdict differs from the
expected label. delta-F1 = macro-F1 on the clean sources minus macro-F1 on the attacks.
Generator and verifier are the same Llama unless FACTKG_ATTACK_MODEL is set.
"""
import argparse, csv, json, os, random
from collections import defaultdict
from adversarial import ATTACKS, generate_attack
from run_phase3_llm import f1_for, gold_of, wilson

def macro_f1(rows):
    return (f1_for(rows, "SUPPORTED")[2] + f1_for(rows, "REFUTED")[2]) / 2 if rows else 0.0

def report(attacks, clean):
    print("=" * 72)
    print(f"Adversarial evaluation: {len(attacks)} attacks on {len({a['source'] for a in attacks})} sources")
    print("=" * 72)
    cf = macro_f1(clean)
    print(f"Clean sources: {len(clean)} claims | accuracy {sum(r['pred'] == r['gold'] for r in clean) / max(len(clean), 1):.3f} | macro-F1 {cf:.3f}")
    by = defaultdict(list)
    for a in attacks:
        by[a["mode"]].append(a)
    for mode, rows in by.items():
        ok = [r for r in rows if r["pred"] is not None]
        fooled = sum(r["pred"] != r["gold"] for r in ok)
        lo, hi = wilson(fooled, len(ok))
        print(f"  {mode:10s} n={len(rows):3d} parsed={len(ok):3d}  ASR={fooled / max(len(ok), 1):.3f} (95% CI {lo:.2f}-{hi:.2f})  "
              f"accuracy={1 - fooled / max(len(ok), 1):.3f}")
    allr = [r for r in attacks if r["pred"] is not None]
    print(f"All attacks: macro-F1 {macro_f1(allr):.3f} | delta-F1 (clean - attacked) = {cf - macro_f1(allr):+.3f}")
    ref = [r for r in allr if r["gold"] == "REFUTED"]
    print(f"Attacks expected REFUTED: {sum(r['pred'] == 'SUPPORTED' for r in ref)}/{len(ref)} classified SUPPORTED (project ASR)")
    print("Caveat: expected labels of semantic/multihop attacks are assumed unless validated.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2"); ap.add_argument("--phase3", default=None, help="saved clean verdicts (skips re-running the sources)")
    ap.add_argument("--modes", nargs="+", default=list(ATTACKS), choices=list(ATTACKS))
    ap.add_argument("--limit", type=int, default=100); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out"); ap.add_argument("--audit", type=int, default=0)
    ap.add_argument("--report"); ap.add_argument("--validated")
    a = ap.parse_args()

    if a.report:
        attacks = [json.loads(l) for l in open(a.report, encoding="utf-8")]
        clean = [json.loads(l) for l in open(a.report + ".clean", encoding="utf-8")] if os.path.exists(a.report + ".clean") else []
        if a.validated:
            val = {}
            for r in csv.DictReader(open(a.validated, encoding="utf-8")):
                if str(r.get("valid", "")).strip().lower() in ("1", "y", "yes"):
                    val[(r["source"], r["mode"], r["attack"])] = (r.get("label") or "").strip().upper()
            attacks = [dict(x, gold=val[k] or x["gold"]) for x in attacks if (k := (x["source"], x["mode"], x["attack"])) in val]
            print(f"validated attacks kept: {len(attacks)}")
        report(attacks, clean)
        return

    from live_pipeline import LivePipeline
    rows = [json.loads(l) for l in open(a.phase2, encoding="utf-8")]
    random.Random(a.seed).shuffle(rows); rows = rows[: a.limit]
    saved = {}
    if a.phase3:
        saved = {r["claim"]: r["pred"] for r in map(json.loads, open(a.phase3, encoding="utf-8"))}
    pipe = LivePipeline()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    attacks, clean = [], []
    with open(a.out, "w", encoding="utf-8") as out:
        for row in rows:
            src, gold = row["claim"], gold_of(row)
            pred = saved[src] if src in saved else pipe.verify(src)["pred"]
            clean.append({"claim": src, "gold": gold, "pred": pred})
            for mode in a.modes:
                if mode != "linguistic" and gold != "SUPPORTED":
                    continue
                atk = generate_attack(src, mode)
                if not atk or atk.lower() == src.lower():
                    continue
                exp = gold if mode == "linguistic" else "REFUTED"
                rec = {"source": src, "mode": mode, "attack": atk, "gold": exp, "pred": pipe.verify(atk)["pred"]}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush(); attacks.append(rec)
    with open(a.out + ".clean", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in clean)
    if a.audit and attacks:
        sample = random.Random(a.seed).sample(attacks, min(a.audit, len(attacks)))
        with open(a.out + ".audit.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["source", "attack", "mode", "expected", "valid", "label"])
            w.writerows([[s["source"], s["attack"], s["mode"], s["gold"], "", ""] for s in sample])
        print(f"audit sheet: {a.out}.audit.csv (fill valid=1/0; label=SUPPORTED/REFUTED only to correct the expected label)")
    report(attacks, clean)

if __name__ == "__main__":
    main()
