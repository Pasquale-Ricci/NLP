"""
Phase 2 entry point: candidate retrieval + reranking + evidence evaluation.

Examples
  # realistic: seeds from Phase 1 (predicted + alternates), compare scorers
  python run_phase2.py --split dev --phase1 phase1_outputs/dev_300_v3.json \
      --seeds seed --scorers lexical cross --save phase2_outputs/dev_300_seed.jsonl

  # oracle seeds (upper bound; isolates retrieval errors from entity linking)
  python run_phase2.py --split dev --phase1 phase1_outputs/dev_300_v3.json \
      --seeds gold --scorers lexical --save phase2_outputs/dev_300_gold.jsonl

  # FAST ablations of candidate generation (no reranker, ~1 min per 300 claims)
  python run_phase2.py --split dev --phase1 phase1_outputs/dev_300_v3.json --seeds gold \
      --scorers --prefilter lexical --k 3 --save phase2_outputs/abl_k3.jsonl

Candidates are generated once per claim and re-ranked by every requested
scorer. `--scorers` with no values = candidate generation only. An existing
--save file is never overwritten unless --overwrite is given.
"""

import argparse
import json
import os
import pickle
import random
import time
from collections import defaultdict

from tqdm import tqdm

import config
import config_phase2 as cfg
from candidate_retrieval import CandidateRetriever
from evidence_eval import aggregate, evaluate_triples, parse_gold_paths, path_end_nodes
from kg_access import load_kg, walk_full_kg
from rerankers import ScorerFactory, get_device
from triple_verbalizer import triple_text

SPLITS = {"train": config.TRAIN_PICKLE, "dev": config.DEV_PICKLE, "test": config.TEST_PICKLE}
REASONING = {"num1", "num2", "multi claim", "existence", "multi hop", "negation"}


def select_claims(claims, args):
    """-> list of (claim, seeds). Sampling mirrors run_phase1.py (same seed)."""
    if args.phase1:
        with open(args.phase1, encoding="utf-8") as f:
            res = json.load(f)["results"]
        sel = []
        for c, r in res.items():
            if c not in claims:
                continue
            if args.seeds == "gold":
                seeds = claims[c].get("Entity_set", [])
            elif args.seeds == "predicted":
                seeds = r.get("predicted_entities", [])
            else:
                seeds = r.get("seed_entities", r.get("predicted_entities", []))
            sel.append((c, seeds))
        return sel[: args.limit] if args.limit else sel
    if args.seeds != "gold":
        raise SystemExit("--seeds predicted/seed need --phase1 <phase1 results json>")
    items = list(claims.items())
    if args.shuffle:
        random.Random(args.seed).shuffle(items)
    if args.limit:
        items = items[: args.limit]
    return [(c, v.get("Entity_set", [])) for c, v in items]


def length_recall(rows):
    """Recall of gold paths by path length (1, 2, 3+): {bucket: [recall, n_paths]}."""
    c = defaultdict(lambda: [0, 0])
    for m in rows:
        if not m:
            continue
        for length, ok in zip(m["lengths"], m["flags"]):
            b = min(length, 3)
            c[b][0] += int(ok)
            c[b][1] += 1
    return {str(b): [v[0] / v[1], v[1]] for b, v in sorted(c.items())}


def fmt_len(d):
    return "  ".join(f"len{k}{'+' if k == '3' else ''}={v[0]:.3f}(n={v[1]})" for k, v in d.items())


def print_summary(summary, cutoffs, scorers):
    cand = summary["candidates"]
    print("=" * 78)
    print(f"Phase 2 - Evidence retrieval ({summary['num_claims']} claims, "
          f"{cand.get('n', 0)} with gold evidence)")
    print("=" * 78)
    print(f"Gold paths realizable in the KG ({'strict end' if summary['args']['strict_eval'] else 'any end'}): "
          f"{100 * summary['gold_paths_realizable']:.1f}%")
    print(f"Candidate set (before reranking): rec_real={cand.get('rec_real', 0):.3f} (on realizable paths) | "
          f"path_recall={cand.get('path_recall', 0):.3f} (all, per-claim mean)  "
          f"full_cover={cand.get('full_cover', 0):.3f}  "
          f"avg candidates/claim={summary['avg_candidates']:.0f}")
    print("  by path length:", fmt_len(summary["candidates_by_length"]))
    print("  by type:       ", "  ".join(
        f"{t}={v:.3f}" for t, v in sorted(summary["candidates_by_type"].items())))
    print(f"Avg time/claim: retrieval {summary['sec_retrieval']:.2f}s"
          + "".join(f" | {s} {summary['sec_scoring'][s]:.2f}s" for s in scorers))
    if not scorers:
        return
    print("-" * 88)
    print(f"{'scorer':9s}{'N':>4s}{'rec_real':>10s}{'path_rec':>10s}{'edge_prec':>11s}{'F1':>8s}"
          f"{'full_cov':>10s}{'rel_rec':>9s}")
    for s in scorers:
        for n in cutoffs:
            a = summary["scorers"][s][str(n)]
            if a.get("n"):
                print(f"{s:9s}{n:>4d}{a['rec_real']:>10.3f}{a['path_recall']:>10.3f}{a['edge_precision']:>11.3f}"
                      f"{a['f1']:>8.3f}{a['full_cover']:>10.3f}{a['relation_recall']:>9.3f}")
        print("-" * 78)
    print("path_recall@20 by reasoning type / by path length:")
    for s in scorers:
        print(f"  {s:9s}" + "  ".join(
            f"{t}={v:.3f}" for t, v in sorted(summary["path_recall_at_20_by_type"][s].items())))
        print(f"  {'':9s}" + fmt_len(summary["path_recall_at_20_by_length"][s]))


def main():
    ap = argparse.ArgumentParser(description="FactKG Phase 2 - evidence retrieval")
    ap.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    ap.add_argument("--phase1", default=None, help="Phase 1 results JSON (claims + seeds)")
    ap.add_argument("--seeds", choices=["seed", "predicted", "gold"], default="seed",
                    help="seed = Phase 1 predicted+alternates, predicted = strict, gold = Entity_set")
    ap.add_argument("--scorers", nargs="*", default=["lexical", "bi", "cross"],
                    choices=["lexical", "bi", "cross", "cascade"],
                    help="no values = candidate generation only")
    ap.add_argument("--prefilter", choices=["bi", "lexical"], default="bi",
                    help="cheap scorer used to pick relations / frontier during expansion")
    ap.add_argument("--k", type=int, choices=[1, 2, 3], default=2)
    ap.add_argument("--rels1", type=int, default=None, help="relations kept per seed (hop 1)")
    ap.add_argument("--rels2", type=int, default=None, help="relations kept per node (hop >= 2)")
    ap.add_argument("--tails-cap", type=int, default=None)
    ap.add_argument("--beam", type=int, default=None)
    ap.add_argument("--hub-sample", type=int, default=None)
    ap.add_argument("--max-seeds", type=int, default=None)
    ap.add_argument("--meet-pairs", type=int, default=None,
                    help="meet-in-the-middle relation pairs per seed pair (0 = off)")
    ap.add_argument("--meet-cap", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kg", default=config.KG_PICKLE)
    ap.add_argument("--bi-model", default=None, help="HF name or local dir")
    ap.add_argument("--cross-model", default=None,
                    help="HF name or local dir (e.g. models/cross_ft after fine-tuning)")
    ap.add_argument("--save", default=None, help="Output .jsonl (one line per claim)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--strict-eval", action="store_true",
                    help="paths connecting two entities must end at the other entity")
    args = ap.parse_args()

    if args.save and os.path.exists(args.save) and not args.overwrite:
        raise SystemExit(f"{args.save} already exists: pick another name or pass --overwrite")

    pick = lambda v, d: d if v is None else v
    params = {
        "rels1": pick(args.rels1, cfg.RELS_PER_NODE_HOP1),
        "rels2": pick(args.rels2, cfg.RELS_PER_NODE_HOP2),
        "tails_cap": pick(args.tails_cap, cfg.TAILS_PER_RELATION_CAP),
        "beam": pick(args.beam, cfg.HOP2_BEAM),
        "hub_sample": pick(args.hub_sample, cfg.HUB_SAMPLE),
        "max_seeds": pick(args.max_seeds, cfg.MAX_SEEDS),
        "meet_pairs": pick(args.meet_pairs, cfg.MEET_PAIRS),
        "meet_cap": pick(args.meet_cap, cfg.MEET_CAP),
    }
    cutoffs = [n for n in cfg.EVAL_CUTOFFS if n <= cfg.MAX_SAVED]

    print(f"Loading claims ({args.split})...")
    with open(SPLITS[args.split], "rb") as f:
        claims = pickle.load(f)
    selected = select_claims(claims, args)
    print(f"{len(selected)} claims selected. k={args.k}, params={params}")

    print("Loading KG (adjacency only, lighter than the Phase 1 index)...")
    kg = load_kg(args.kg)
    print(f"KG nodes: {len(kg)}")

    factory = ScorerFactory(args.bi_model or cfg.BI_ENCODER_MODEL,
                            args.cross_model or cfg.CROSS_ENCODER_MODEL, cfg.CASCADE_K)
    scorers = {s: factory.get(s) for s in args.scorers}
    prefilter = factory.get(args.prefilter)
    if args.prefilter != "lexical" or set(args.scorers) - {"lexical"}:
        dev = get_device()
        print(f"Neural models on device: {dev}")
        if dev == "cpu":
            print("WARNING: running on CPU (torch.cuda.is_available() is False). Install a CUDA "
                  "build of PyTorch (pytorch.org) for a large speed-up.")
    retriever = CandidateRetriever(
        kg, prefilter, params["rels1"], params["rels2"], params["tails_cap"],
        params["beam"], params["max_seeds"], params["hub_sample"],
        params["meet_pairs"], params["meet_cap"], cfg.MEET_MAX_EDGES)

    acc = {s: {n: [] for n in cutoffs} for s in scorers}
    by_type = {s: defaultdict(list) for s in scorers}       # metrics at N=20 per type
    cand_rows, cand_by_type, n_cands = [], defaultdict(list), []
    t_ret, t_score = 0.0, {s: 0.0 for s in scorers}
    real_tot = [0, 0]                                   # realizable / total gold paths
    cand_real = [0, 0]
    acc_real = {s: {n: [0, 0] for n in cutoffs} for s in scorers}

    out_f = None
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        out_f = open(args.save, "w", encoding="utf-8")

    for claim, seeds in tqdm(selected, desc="Retrieval"):
        info = claims[claim]
        gold_paths = parse_gold_paths(info.get("Evidence"))
        types = info.get("types", [])
        ends = path_end_nodes(gold_paths) if args.strict_eval else {}
        real = [e in kg and walk_full_kg(kg, e, p, end=ends.get((e, p))) is not False
                for e, p in gold_paths]
        real_tot[0] += sum(real)
        real_tot[1] += len(real)

        t0 = time.time()
        cands = retriever.retrieve(claim, seeds, k=args.k)
        t_ret += time.time() - t0
        triples = list(cands)
        texts = [triple_text(*t) for t in triples]
        n_cands.append(len(triples))
        cand_m = evaluate_triples(triples, gold_paths, args.strict_eval)
        cand_rows.append(cand_m)
        if cand_m:
            for f, r_ in zip(cand_m["flags"], real):
                if r_:
                    cand_real[0] += int(f)
                    cand_real[1] += 1
        if cand_m:
            for t in types:
                if t in REASONING:
                    cand_by_type[t].append(cand_m)

        rec = {"claim": claim, "label": info.get("Label"), "types": types,
               "seeds": list(seeds), "n_candidates": len(triples),
               "hops": {str(h): sum(1 for v in cands.values() if v == h) for h in (0, 1, 2, 3)},
               "gold_paths": [[e, list(p)] for e, p in gold_paths],
               "realizable": real,
               "candidate_metrics": cand_m, "scorers": {}}

        for name, sc in scorers.items():
            t0 = time.time()
            scores = sc.score_triples(claim, texts)
            t_score[name] += time.time() - t0
            order = sorted(range(len(triples)), key=lambda i: -scores[i])[: cfg.MAX_SAVED]
            ranked = [(triples[i], scores[i]) for i in order]
            metrics = {}
            for n in cutoffs:
                m = evaluate_triples([t for t, _ in ranked[:n]], gold_paths, args.strict_eval)
                if m:
                    metrics[str(n)] = m
                    acc[name][n].append(m)
                    for f, r_ in zip(m["flags"], real):
                        if r_:
                            acc_real[name][n][0] += int(f)
                            acc_real[name][n][1] += 1
            if "20" in metrics:
                for t in types:
                    if t in REASONING:
                        by_type[name][t].append(metrics["20"])
            rec["scorers"][name] = {
                "ranked": [[h, r, t, round(s, 4)] for (h, r, t), s in ranked],
                "metrics": metrics}
        if out_f:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if out_f:
        out_f.close()

    n = max(len(selected), 1)
    summary = {
        "num_claims": len(selected), "k": args.k, "params": params, "args": vars(args),
        "avg_candidates": sum(n_cands) / n,
        "sec_retrieval": t_ret / n,
        "sec_scoring": {s: t_score[s] / n for s in scorers},
        "gold_paths_realizable": real_tot[0] / max(real_tot[1], 1),
        "candidates": {**aggregate(cand_rows), "rec_real": cand_real[0] / max(cand_real[1], 1)},
        "candidates_by_length": length_recall(cand_rows),
        "candidates_by_type": {t: aggregate(v)["path_recall"] for t, v in cand_by_type.items()},
        "scorers": {s: {str(c): {**aggregate(acc[s][c]),
                                 "rec_real": acc_real[s][c][0] / max(acc_real[s][c][1], 1)}
                        for c in cutoffs} for s in scorers},
        "path_recall_at_20_by_type": {
            s: {t: aggregate(v)["path_recall"] for t, v in by_type[s].items()} for s in scorers},
        "path_recall_at_20_by_length": {s: length_recall(acc[s][20]) for s in scorers if 20 in cutoffs},
    }
    print_summary(summary, cutoffs, list(scorers))
    if args.save:
        with open(args.save + ".summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved {args.save} and {args.save}.summary.json")


if __name__ == "__main__":
    main()
