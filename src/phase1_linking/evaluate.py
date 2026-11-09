"""
Phase 1 - Evaluation: Entity Extraction Precision / Recall / F1 against
FactKG's gold Entity_set, with digit-tolerant matching for literals.
"""

from collections import defaultdict
from decimal import Decimal, InvalidOperation

KNOWN_REASONING_TYPES = {
    "num1": "One-hop",
    "num2": "One-hop (multi-entity)",
    "multi claim": "Conjunction",
    "existence": "Existence",
    "multi hop": "Multi-hop",
    "negation": "Negation",
}


def _is_literal(s: str) -> bool:
    s = s.strip()
    return s.startswith('"') and s.endswith('"') and len(s) > 1


def _as_decimal(s: str):
    t = s.strip().strip('"').replace(",", "")
    try:
        return Decimal(t)
    except InvalidOperation:
        return None


def _literal_matches(pred: str, gold: str) -> bool:
    dp, dg = _as_decimal(pred), _as_decimal(gold)
    if dp is not None and dg is not None:
        if dp == dg:  # 1604 == 1604.0, 1,147 == 1147.0
            return True
        # FactKG sometimes shifts the decimal point (94.8024 vs 94802.4).
        # Tolerate it only for numbers with >= 5 significant digits, so that
        # round numbers (1, 10, 1000) do not collapse into each other.
        sp = dp.normalize().as_tuple().digits
        sg = dg.normalize().as_tuple().digits
        return len(sp) >= 5 and sp == sg
    norm = lambda x: x.strip().strip('"').replace("_", " ").lower()
    return norm(pred) == norm(gold)


def _entity_matches(pred: str, gold: str) -> bool:
    if pred == gold:
        return True
    if _is_literal(pred) and _is_literal(gold):
        return _literal_matches(pred, gold)
    return False


def score_claim(predicted: list, gold: list) -> dict:
    """Set-based matching (order-independent), each gold item matched
    at most once, allowing digit-tolerant literal equivalence."""
    matched_gold_idx = set()
    tp = 0
    for p in predicted:
        for i, g in enumerate(gold):
            if i in matched_gold_idx:
                continue
            if _entity_matches(p, g):
                matched_gold_idx.add(i)
                tp += 1
                break

    fp = len(predicted) - tp
    fn = len(gold) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate(results: dict) -> dict:
    """
    results: output of pipeline.run_entity_linking
    Returns overall micro/macro metrics plus a breakdown by FactKG
    reasoning type (useful later to cross-reference with Phase 3
    reasoning-type performance, per the project's Breakdown Analysis
    metric).
    """
    per_claim_scores = {}
    by_reasoning_type = defaultdict(list)
    total_tp = total_fp = total_fn = 0
    seed_tp = gold_total = seed_size = pred_size = 0

    for claim_text, r in results.items():
        s = score_claim(r["predicted_entities"], r["gold_entities"])
        per_claim_scores[claim_text] = s
        total_tp += s["tp"]
        total_fp += s["fp"]
        total_fn += s["fn"]
        seeds = r.get("seed_entities", r["predicted_entities"])
        seed_tp += score_claim(seeds, r["gold_entities"])["tp"]
        gold_total += len(r["gold_entities"])
        seed_size += len(seeds)
        pred_size += len(r["predicted_entities"])

        reasoning_types = [t for t in r.get("types", []) if t in KNOWN_REASONING_TYPES]
        for t in reasoning_types or ["unknown"]:
            by_reasoning_type[t].append(s["f1"])

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    macro_f1 = (
        sum(s["f1"] for s in per_claim_scores.values()) / len(per_claim_scores)
        if per_claim_scores
        else 0.0
    )

    breakdown = {t: sum(v) / len(v) for t, v in by_reasoning_type.items()}
    counts = {t: len(v) for t, v in by_reasoning_type.items()}

    return {
        "num_claims": len(results),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "seed_recall": seed_tp / gold_total if gold_total else 0.0,
        "avg_seed_size": seed_size / len(results) if results else 0.0,
        "avg_pred_size": pred_size / len(results) if results else 0.0,
        "f1_by_reasoning_type": breakdown,
        "n_by_reasoning_type": counts,
        "per_claim_scores": per_claim_scores,
    }


def print_report(metrics: dict):
    print("=" * 60)
    print(f"Phase 1 - Entity Linking Evaluation ({metrics['num_claims']} claims)")
    print("=" * 60)
    print(f"Micro Precision: {metrics['micro_precision']:.4f}")
    print(f"Micro Recall:    {metrics['micro_recall']:.4f}")
    print(f"Micro F1:        {metrics['micro_f1']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    if "seed_recall" in metrics:
        print(f"Seed recall (Phase 2 seeds incl. alternates): {metrics['seed_recall']:.4f}"
              f"  [avg seeds/claim {metrics['avg_seed_size']:.2f} vs predicted {metrics['avg_pred_size']:.2f}]")
    print("-" * 60)
    print("F1 by reasoning type:")
    counts = metrics.get("n_by_reasoning_type", {})
    for t, f1 in sorted(metrics["f1_by_reasoning_type"].items()):
        label = KNOWN_REASONING_TYPES.get(t, t)
        print(f"  {t:15s} ({label:25s}) {f1:.4f}  n={counts.get(t, '?')}")
    print("=" * 60)
