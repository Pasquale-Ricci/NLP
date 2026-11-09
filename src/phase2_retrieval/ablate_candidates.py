"""
Phase 2 - Fast sweep of candidate-generation settings. The KG is loaded ONCE
and every configuration is run on the same claims, so a whole sweep costs
little more than a single run (no reranker needed).

Usage:
  python ablate_candidates.py --split dev --phase1 phase1_outputs/dev_300_v3.json --seeds gold
  python ablate_candidates.py ... --config "mine:rels1=40,rels2=10,beam=60,hub=3,meet=10"

Keys for --config: k, rels1, rels2, tails, beam, hub, meet, mcap  (unset = default).
Default sweep: old_default, hub3, rels30, rels60, beam100, meet10, new_default.
Recall is reported on ALL gold paths and on REALIZABLE paths only.
"""

import argparse
import json
import pickle
import time
from collections import defaultdict

import config
import config_phase2 as cfg
from candidate_retrieval import CandidateRetriever
from evidence_eval import evaluate_triples, parse_gold_paths
from kg_access import load_kg, walk_full_kg
from rerankers import ScorerFactory
from run_phase2 import REASONING, SPLITS, select_claims

DEFAULT_SWEEP = [
    "old_default:rels1=15,rels2=6,beam=30,hub=0,meet=0",
    "hub3:rels1=15,rels2=6,beam=30,hub=3,meet=0",
    "rels30:rels1=30,rels2=10,beam=30,hub=0,meet=0",
    "rels60:rels1=60,rels2=15,beam=30,hub=0,meet=0",
    "beam100:rels1=15,rels2=6,beam=100,hub=0,meet=0",
    "meet10:rels1=15,rels2=6,beam=30,hub=0,meet=10",
    "new_default:",
]
KEYS = {"k": "k", "rels1": "rels1", "rels2": "rels2", "tails": "tails_cap", "beam": "beam",
        "hub": "hub_sample", "meet": "meet_pairs", "mcap": "meet_cap"}


def parse_config(text):
    name, _, rest = text.partition(":")
    cfg_d = {}
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=")
        cfg_d[KEYS[k.strip()]] = int(v)
    return name.strip(), cfg_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    ap.add_argument("--phase1", default=None)
    ap.add_argument("--seeds", choices=["seed", "predicted", "gold"], default="gold")
    ap.add_argument("--prefilter", choices=["bi", "lexical"], default="lexical")
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kg", default=config.KG_PICKLE)
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    with open(SPLITS[args.split], "rb") as f:
        claims = pickle.load(f)
    selected = select_claims(claims, args)
    kg = load_kg(args.kg)
    print(f"{len(selected)} claims, KG nodes: {len(kg)}")
    prefilter = ScorerFactory(cfg.BI_ENCODER_MODEL, cfg.CROSS_ENCODER_MODEL, cfg.CASCADE_K).get(args.prefilter)

    data = []
    for claim, seeds in selected:
        info = claims[claim]
        gp = parse_gold_paths(info.get("Evidence"))
        real = [False if e not in kg else walk_full_kg(kg, e, p) is not False for e, p in gp]
        data.append((claim, seeds, gp, real, info.get("types", [])))
    n_paths = sum(len(d[2]) for d in data)
    n_real = sum(sum(d[3]) for d in data)
    print(f"gold paths: {n_paths} | realizable in the KG: {n_real} ({100 * n_real / max(n_paths, 1):.1f}%)\n")

    results = {}
    for text in (args.config or DEFAULT_SWEEP):
        name, c = parse_config(text)
        k = c.pop("k", 2)
        p = {"rels1": cfg.RELS_PER_NODE_HOP1, "rels2": cfg.RELS_PER_NODE_HOP2,
             "tails_cap": cfg.TAILS_PER_RELATION_CAP, "beam": cfg.HOP2_BEAM,
             "hub_sample": cfg.HUB_SAMPLE, "meet_pairs": cfg.MEET_PAIRS, "meet_cap": cfg.MEET_CAP}
        p.update(c)
        ret = CandidateRetriever(kg, prefilter, p["rels1"], p["rels2"], p["tails_cap"], p["beam"],
                                 cfg.MAX_SEEDS, p["hub_sample"], p["meet_pairs"], p["meet_cap"],
                                 cfg.MEET_MAX_EDGES)
        got_all = got_real = n_cand = 0
        full_cov = full_n = 0
        by_len, by_type = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        t0 = time.time()
        for claim, seeds, gp, real, types in data:
            cands = ret.retrieve(claim, seeds, k=k)
            n_cand += len(cands)
            m = evaluate_triples(list(cands), gp)
            if not m:
                continue
            ok_real = []
            for (e, path), ok, rl in zip(gp, m["flags"], real):
                got_all += int(ok)
                if rl:
                    got_real += int(ok)
                    ok_real.append(ok)
                    b = min(len(path), 3)
                    by_len[b][0] += int(ok)
                    by_len[b][1] += 1
                    for t in types:
                        if t in REASONING:
                            by_type[t][0] += int(ok)
                            by_type[t][1] += 1
            if ok_real:
                full_n += 1
                full_cov += int(all(ok_real))
        sec = (time.time() - t0) / max(len(data), 1)
        res = {
            "recall_all": got_all / max(n_paths, 1),
            "recall_realizable": got_real / max(n_real, 1),
            "full_cover_realizable": full_cov / max(full_n, 1),
            "avg_candidates": n_cand / max(len(data), 1), "sec_per_claim": sec,
            "by_length": {str(b): v[0] / v[1] for b, v in sorted(by_len.items())},
            "by_type": {t: v[0] / v[1] for t, v in sorted(by_type.items())},
        }
        results[name] = res
        print(f"[{name}] k={k} {p}")
        print(f"    recall: all={res['recall_all']:.3f}  REALIZABLE={res['recall_realizable']:.3f}  "
              f"full_cover(realizable)={res['full_cover_realizable']:.3f}  "
              f"cands/claim={res['avg_candidates']:.0f}  {sec:.3f}s/claim")
        print("    by length:", "  ".join(f"len{b}={v:.3f}" for b, v in res["by_length"].items()))
        print("    by type:  ", "  ".join(f"{t}={v:.3f}" for t, v in res["by_type"].items()))

    print("\nSummary (recall on realizable paths | avg candidates):")
    for name, r in results.items():
        print(f"  {name:10s} {r['recall_realizable']:.3f} | {r['avg_candidates']:.0f}")
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
