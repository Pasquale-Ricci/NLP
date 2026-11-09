"""
CLI entry point for Phase 1: Entity Linking & Literal Extraction.

Examples:
    python run_phase1.py --split dev --limit 200 --save phase1_outputs/dev_results.json
    python run_phase1.py --split train --save phase1_outputs/train_results.json
"""

import argparse
import json
import os

import config
from local_kg_index import LocalKGIndex
from pipeline import load_claims, run_entity_linking
from evaluate import evaluate, print_report


def _split_path(split: str) -> str:
    return {
        "train": config.TRAIN_PICKLE,
        "dev": config.DEV_PICKLE,
        "test": config.TEST_PICKLE,
    }[split]


def main():
    parser = argparse.ArgumentParser(description="FactKG Phase 1 - Entity Linking")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap number of claims (for quick tests)"
    )
    parser.add_argument("--shuffle", action="store_true", help="Randomly sample claims (recommended with --limit)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None, help="Path to save JSON results")
    args = parser.parse_args()

    if args.split == "test":
        print(
            "Note: the 'test' split has no gold 'Evidence' and typically an "
            "unreliable 'Entity_set' for scoring — results will still be "
            "produced, but evaluate() metrics may not be meaningful."
        )

    print(f"Loading claims from split: {args.split}")
    claims = load_claims(_split_path(args.split))
    print(f"Loaded {len(claims)} claims.")

    print("Building local KG index (this can take a moment)...")
    kg_index = LocalKGIndex(config.KG_PICKLE)
    print(f"Indexed {kg_index.num_entities} entities ({kg_index.num_literals} literal nodes skipped).")

    results = run_entity_linking(
        claims, kg_index, limit=args.limit, shuffle=args.shuffle, seed=args.seed
    )

    metrics = evaluate(results)
    print_report(metrics)

    if args.save:
        out_dir = os.path.dirname(args.save)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        serializable = {
            "metrics": {k: v for k, v in metrics.items() if k != "per_claim_scores"},
            "results": results,
        }
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {args.save}")


if __name__ == "__main__":
    main()
