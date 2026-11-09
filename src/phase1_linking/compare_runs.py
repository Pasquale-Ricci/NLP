"""
Phase 1 - Paired comparison of two saved runs on the SAME claims.

Usage:
    python compare_runs.py run_A.json run_B.json [--n-boot 5000]

Computes the per-claim F1 difference (B - A), its mean, win/loss/tie counts
and a bootstrap 95% confidence interval over claims. If the interval
contains 0, the difference between the two runs is not distinguishable from
noise on this sample. Note: this uses the mean of per-claim F1 (macro), not
the micro-averaged F1 printed by run_phase1.py.
"""

import argparse
import json
import random

from evaluate import score_claim


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["results"]


def per_claim_f1(results):
    return {
        c: score_claim(r["predicted_entities"], r["gold_entities"])["f1"]
        for c, r in results.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fa, fb = per_claim_f1(load(args.run_a)), per_claim_f1(load(args.run_b))
    common = sorted(set(fa) & set(fb))
    if not common:
        raise SystemExit("No claims in common: were both runs made with the same split/seed/limit?")

    diffs = [fb[c] - fa[c] for c in common]
    n = len(diffs)
    mean_diff = sum(diffs) / n
    rng = random.Random(args.seed)
    boots = sorted(
        sum(rng.choices(diffs, k=n)) / n for _ in range(args.n_boot)
    )
    lo, hi = boots[int(0.025 * args.n_boot)], boots[int(0.975 * args.n_boot) - 1]

    print(f"Claims in common: {n}")
    print(f"Mean per-claim F1  A: {sum(fa[c] for c in common)/n:.4f} | B: {sum(fb[c] for c in common)/n:.4f}")
    print(f"Mean diff (B - A): {mean_diff:+.4f}  (95% bootstrap CI: {lo:+.4f} to {hi:+.4f})")
    print(f"B better on {sum(d > 0 for d in diffs)} claims, A better on {sum(d < 0 for d in diffs)}, tie on {sum(d == 0 for d in diffs)}")
    print("=> " + ("difference likely real" if lo > 0 or hi < 0 else "difference NOT distinguishable from noise on this sample"))


if __name__ == "__main__":
    main()
