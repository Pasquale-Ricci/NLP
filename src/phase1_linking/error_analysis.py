"""
Phase 1 - Error analysis on a saved results JSON.

Usage:
    python error_analysis.py phase1_outputs/dev_300_v2.json [--show 20]

Reports false positives (with a trace raw mention -> resolved entity),
false negatives and unmatched mentions. It also estimates how much recall
a claim-n-gram lookup could recover: among missed gold ENTITIES, how many
have a name that appears verbatim in the claim text.
"""

import argparse
import json
import re
from collections import Counter

from evaluate import _entity_matches
from local_kg_index import _normalize


def _trace(details):
    return " | ".join(
        f'{d["mention"]} -> {d["canonical"]} [{d["method"]}'
        + (f', {d["score"]:.0f}' if isinstance(d.get("score"), (int, float)) else "")
        + "]"
        for d in details
    )


def _name_in_claim(gold_entity: str, norm_claim: str) -> bool:
    """True if the entity's name (with or without a '_(disambiguation)' suffix)
    appears verbatim, on word boundaries, in the normalized claim."""
    variants = {gold_entity, re.sub(r"_\([^)]*\)$", "", gold_entity)}
    return any(
        (n := _normalize(v)) and f" {n} " in norm_claim for v in variants
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("--show", type=int, default=15, help="Max examples per category")
    args = ap.parse_args()

    with open(args.results_json, encoding="utf-8") as f:
        results = json.load(f)["results"]

    fps, fns, unmatched = [], [], []
    method_counter, fp_method_counter = Counter(), Counter()
    missed_total = missed_literal = missed_entity = missed_entity_in_text = 0
    gold_total = 0

    for claim, r in results.items():
        pred, gold = r["predicted_entities"], r["gold_entities"]
        gold_total += len(gold)
        extra = [p for p in pred if not any(_entity_matches(p, g) for g in gold)]
        missed = [g for g in gold if not any(_entity_matches(p, g) for p in pred)]
        bad = [d["mention"] for d in r["validation_details"] if d["method"] == "unmatched"]
        for d in r["validation_details"]:
            method_counter[d["method"]] += 1
            if d["canonical"] in extra:
                fp_method_counter[d["method"]] += 1

        norm_claim = f" {_normalize(claim)} "
        for g in missed:
            missed_total += 1
            if g.startswith('"'):
                missed_literal += 1
            else:
                missed_entity += 1
                if _name_in_claim(g, norm_claim):
                    missed_entity_in_text += 1

        if extra:
            fps.append((claim, extra, gold, _trace(r["validation_details"])))
        if missed:
            fns.append((claim, missed, pred))
        if bad:
            unmatched.append((claim, bad))

    print(f"Claims: {len(results)} | with FP: {len(fps)} | with FN: {len(fns)} | with unmatched: {len(unmatched)}")
    print("\nResolution methods (all mentions):", dict(method_counter))
    print("Resolution methods (false positives only):", dict(fp_method_counter))

    print("\n--- Recall diagnosis ---")
    print(f"Gold items: {gold_total} | missed: {missed_total} "
          f"({100 * missed_total / max(gold_total, 1):.1f}%)")
    print(f"  missed literals: {missed_literal}")
    print(f"  missed entities: {missed_entity}")
    print(f"    of which the name appears verbatim in the claim: {missed_entity_in_text} "
          f"(upper bound of what n-gram lookup could recover)")
    print(f"    of which NOT verbatim in the claim (aliases, implied entities): "
          f"{missed_entity - missed_entity_in_text}")

    print("\nMost frequent false-positive strings:")
    for s, c in Counter(e for _, ex, _, _ in fps for e in ex).most_common(10):
        print(f"  {c:3d}  {s}")

    print("\n=== FALSE POSITIVES (claim / extra / gold / trace) ===")
    for row in fps[: args.show]:
        print(f"- {row[0]}\n    extra: {row[1]}\n    gold:  {row[2]}\n    trace: {row[3]}")

    print("\n=== FALSE NEGATIVES (claim / missed / predicted) ===")
    for row in fns[: args.show]:
        print(f"- {row[0]}\n    missed:    {row[1]}\n    predicted: {row[2]}")

    print("\n=== UNMATCHED MENTIONS ===")
    for row in unmatched[: args.show]:
        print(f"- {row[0]}\n    {row[1]}")


if __name__ == "__main__":
    main()
