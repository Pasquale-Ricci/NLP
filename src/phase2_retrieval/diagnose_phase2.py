"""
Phase 2 - Why does the candidate set miss gold evidence paths?

Usage:
    python diagnose_phase2.py phase2_outputs/dev_300_gold.jsonl [--show 6] [--kg path]

Every gold path is first checked against the FULL KG (the KG stores inverse
edges as '~rel' keys, so a path is a sequence of dictionary lookups):
  realizable      the path can be walked in the KG
  not realizable  it cannot (the evidence "does not exist" in the KG; for false
                  existence/negation claims this absence is the expected signal,
                  not a retrieval failure)
Recall is reported on ALL gold paths and on REALIZABLE paths only (the honest
measure of the retriever). Paths missed by the candidate set are classified:

  entity_not_in_kg   start node is not a KG node
  not_in_kg          path not realizable in the KG
  too_long_for_k     more edges than the expansion depth k
  lost_hub           first relation has more tails than the hub cap
  lost_many_rels     start node has more relations than the hop-1 budget
  lost_later_step    walkable, short, no hub/budget issue at step 1: lost later
The last three are HEURISTIC labels; confirm with ablate_candidates.py.
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import config
import config_phase2 as cfg
from evidence_eval import path_end_nodes
from kg_access import load_kg, walk_full_kg

TYPES = ("num1", "num2", "multi claim", "existence", "multi hop", "negation")


def label_of(row):
    lab = row.get("label")
    return lab[0] if isinstance(lab, list) and lab else lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--kg", default=config.KG_PICKLE)
    ap.add_argument("--show", type=int, default=6, help="examples per category")
    args = ap.parse_args()

    params = {"rels1": cfg.RELS_PER_NODE_HOP1, "tails_cap": cfg.TAILS_PER_RELATION_CAP, "k": 2}
    sp = args.jsonl + ".summary.json"
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            s = json.load(f)
        params.update(s.get("params", {}))
        params["k"] = s.get("k", 2)
    print("Using params:", params)

    rows = []
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("candidate_metrics"):
                rows.append(r)
    print("Loading KG...")
    kg = load_kg(args.kg)

    cats, examples = Counter(), defaultdict(list)
    cat_by_type, cat_by_label = defaultdict(Counter), defaultdict(Counter)
    total = missed = n_real = cand_real = 0
    n_strict_real = 0
    strict_unreal_label = Counter()
    scorer_real = defaultdict(lambda: [0, 0])          # scorer -> [realized@20 among realizable, n]
    for r in rows:
        flags = r["candidate_metrics"]["flags"]
        lab = label_of(r)
        ends = path_end_nodes([(e, tuple(p)) for e, p in r["gold_paths"]])
        for i, ((ent, path), ok) in enumerate(zip(r["gold_paths"], flags)):
            total += 1
            walk = False if ent not in kg else walk_full_kg(kg, ent, path)
            realizable = walk is not False
            end = ends.get((ent, tuple(path)))
            strict_ok = ent in kg and walk_full_kg(kg, ent, path, end=end) is not False
            n_strict_real += int(strict_ok)
            if realizable and not strict_ok:
                strict_unreal_label[lab] += 1
            if realizable:
                n_real += 1
                cand_real += int(ok)
                for name, sc in r["scorers"].items():
                    m = sc["metrics"].get("20")
                    if m:
                        scorer_real[name][0] += int(m["flags"][i])
                        scorer_real[name][1] += 1
            if ok:
                continue
            missed += 1
            if ent not in kg:
                cat = "entity_not_in_kg"
            elif walk is False:
                cat = "not_in_kg"
            elif len(path) > params["k"]:
                cat = "too_long_for_k"
            else:
                rels = kg[ent]
                if len(rels.get(path[0], ())) > params["tails_cap"]:
                    cat = "lost_hub"
                elif len(rels) > params["rels1"]:
                    cat = "lost_many_rels"
                else:
                    cat = "lost_later_step"
            cats[cat] += 1
            cat_by_label[cat][lab] += 1
            for t in r["types"]:
                if t in TYPES:
                    cat_by_type[t][cat] += 1
            if len(examples[cat]) < args.show:
                examples[cat].append((r["claim"], ent, path))

    print(f"\nGold paths: {total} | realizable in the KG: {n_real} "
          f"({100 * n_real / max(total, 1):.1f}%)")
    print(f"Realizable with the END-ENTITY constraint (strict): {n_strict_real} "
          f"({100 * n_strict_real / max(total, 1):.1f}%)")
    print(f"  paths walkable but NOT ending at the other entity, by claim label: "
          f"{dict(strict_unreal_label)}  (expected: refuted claims built by substitution)")
    print(f"Candidate recall, ALL paths:        {(total - missed) / max(total, 1):.3f}")
    print(f"Candidate recall, REALIZABLE paths: {cand_real / max(n_real, 1):.3f}   <- retriever quality")
    for name, (a, b) in scorer_real.items():
        print(f"  {name:9s} recall@20 on realizable paths: {a / max(b, 1):.3f}")
    print(f"\nMissed by candidates: {missed} ({100 * missed / max(total, 1):.1f}% of paths)\n")
    for cat, c in cats.most_common():
        labs = ", ".join(f"label={k}: {v}" for k, v in cat_by_label[cat].most_common())
        print(f"  {cat:18s} {c:5d}  ({100 * c / max(missed, 1):4.1f}% of misses)   [{labs}]")
    print("\nMisses by reasoning type:")
    for t, cc in sorted(cat_by_type.items()):
        print(f"  {t:12s} " + "  ".join(f"{k}={v}" for k, v in cc.most_common()))
    for cat, ex in examples.items():
        print(f"\n--- {cat} ---")
        for claim, ent, path in ex:
            print(f"  {claim[:100]}\n      {ent}  {path}")


if __name__ == "__main__":
    main()
