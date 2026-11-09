"""
Phase 2b - Build (claim, triple, label) pairs to fine-tune the Cross-Encoder.

For each claim (gold Entity_set as seeds) the SAME candidate generator used at
inference runs; a triple is POSITIVE if it lies on a complete walk of a gold
evidence path. Negatives: `--neg-hard` candidates with the best lexical score
among non-positives (the distractors that look relevant) plus `--neg-rand`
uniformly sampled ones. Claims whose gold paths are not realized by the
candidates (no positives, e.g. evidence absent from the KG) are skipped.

  python build_rerank_data.py --split train --limit 20000 --shuffle --out data/rerank_train.jsonl
  python build_rerank_data.py --split train --limit 1000 --offset 20000 --shuffle --out data/rerank_val.jsonl

(--offset makes the validation claims disjoint from the training ones.)
"""

import argparse
import json
import os
import pickle
import random

from tqdm import tqdm

import config
import config_phase2 as cfg
from candidate_retrieval import CandidateRetriever
from evidence_eval import parse_gold_paths, positive_edges
from kg_access import load_kg
from rerankers import LexicalScorer
from triple_verbalizer import triple_text

SPLITS = {"train": config.TRAIN_PICKLE, "dev": config.DEV_PICKLE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev"], default="train")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kg", default=config.KG_PICKLE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg-hard", type=int, default=8)
    ap.add_argument("--neg-rand", type=int, default=8)
    ap.add_argument("--max-pos", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} already exists: pick another name or pass --overwrite")

    with open(SPLITS[args.split], "rb") as f:
        claims = pickle.load(f)
    items = list(claims.items())
    if args.shuffle:
        random.Random(args.seed).shuffle(items)
    items = items[args.offset: args.offset + args.limit]
    print(f"{len(items)} claims; loading KG...")
    kg = load_kg(args.kg)

    lexical = LexicalScorer()
    retriever = CandidateRetriever(
        kg, lexical, cfg.RELS_PER_NODE_HOP1, cfg.RELS_PER_NODE_HOP2, cfg.TAILS_PER_RELATION_CAP,
        cfg.HOP2_BEAM, cfg.MAX_SEEDS, cfg.HUB_SAMPLE, cfg.MEET_PAIRS, cfg.MEET_CAP, cfg.MEET_MAX_EDGES)

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_used = n_no_paths = n_no_pos = n_pos = n_neg = n_fallback = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for claim, info in tqdm(items, desc="Building pairs"):
            gp = parse_gold_paths(info.get("Evidence"))
            if not gp:
                n_no_paths += 1
                continue
            triples = list(retriever.retrieve(claim, info.get("Entity_set", []), k=2))
            pos_set = positive_edges(triples, gp)
            if pos_set and not positive_edges(triples, gp, lenient_fallback=False):
                n_fallback += 1
            pos = list(pos_set)
            if not pos:
                n_no_pos += 1
                continue
            negs = [t for t in triples if t not in pos_set]
            texts = [triple_text(*t) for t in negs]
            scores = lexical.score_triples(claim, texts)
            order = sorted(range(len(negs)), key=lambda i: -scores[i])
            chosen = order[: args.neg_hard]
            rest = order[args.neg_hard:]
            chosen += rng.sample(rest, min(args.neg_rand, len(rest)))
            if len(pos) > args.max_pos:
                pos = rng.sample(pos, args.max_pos)
            for t in pos:
                out.write(json.dumps({"claim": claim, "triple": triple_text(*t), "label": 1},
                                     ensure_ascii=False) + "\n")
            for i in chosen:
                out.write(json.dumps({"claim": claim, "triple": texts[i], "label": 0},
                                     ensure_ascii=False) + "\n")
            n_used += 1
            n_pos += len(pos)
            n_neg += len(chosen)

    print(f"claims used: {n_used} (of which {n_fallback} only via the lenient fallback, i.e. refuted claims) "
          f"| no gold paths: {n_no_paths} | gold not realized: {n_no_pos}")
    print(f"pairs: {n_pos + n_neg} (positives {n_pos}, negatives {n_neg}) -> {args.out}")


if __name__ == "__main__":
    main()
