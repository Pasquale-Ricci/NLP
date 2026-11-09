"""
FactKG demo site (local). Run from the folder with the other scripts:
  pip install flask
  python app.py        ->  http://127.0.0.1:5000
Env: FACTKG_BENCH (Phase 2 .jsonl), FACTKG_P3 (Phase 3 results .jsonl), FACTKG_SCORER
(lexical|cross), FACTKG_CROSS (model dir). The live pipeline (Sandbox/Adversarial tabs)
loads the KG on first use (minutes, several GB) and needs the Llama server running.
"""
import json, os, random
from flask import Flask, jsonify, request, send_from_directory
import config, config_phase2 as cfg
from llm_verifier import ask, build_messages, parse_verdict
from triple_verbalizer import triple_text

app = Flask(__name__, static_folder="static")
BENCH = os.environ.get("FACTKG_BENCH", "phase2_outputs/ft_seed.jsonl")
P3 = os.environ.get("FACTKG_P3", "phase3_outputs/llm_seed_cross20.jsonl")
SCORER = os.environ.get("FACTKG_SCORER", "cross")
N_FACTS = 20
_cache = {}

def _load(path):
    if path not in _cache:
        _cache[path] = ([json.loads(l) for l in open(path, encoding="utf-8")]
                        if os.path.exists(path) else [])
    return _cache[path]

def _label(row):
    lab = row.get("label"); lab = lab[0] if isinstance(lab, list) and lab else lab
    return "SUPPORTED" if lab else "REFUTED"

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/sample")
def sample():
    rows = _load(BENCH)
    t, lab = request.args.get("type"), request.args.get("label")
    rows = [r for r in rows if (not t or t in r["types"]) and (not lab or _label(r) == lab.upper())]
    if not rows:
        return jsonify(error=f"No claims in {BENCH} for this filter (set FACTKG_BENCH)."), 404
    r = random.choice(rows)
    sc = SCORER if SCORER in r["scorers"] else next(iter(r["scorers"]))
    triples = r["scorers"][sc]["ranked"][:N_FACTS]
    p3 = {x["claim"]: x for x in _load(P3)}.get(r["claim"], {})
    return jsonify(claim=r["claim"], gold=_label(r), types=r["types"], seeds=r["seeds"],
                   triples=[x[:3] for x in triples], pred=p3.get("pred"), response=p3.get("response"))

_live = {}
def _pipeline():
    if not _live:
        from local_kg_index import LocalKGIndex
        from entity_validator import EntityValidator
        from candidate_retrieval import CandidateRetriever
        from rerankers import LexicalScorer, CrossEncoderScorer
        idx = LocalKGIndex(config.KG_PICKLE)
        lex = LexicalScorer()
        _live.update(idx=idx, val=EntityValidator(idx), lex=lex,
                     scorer=CrossEncoderScorer(os.environ.get("FACTKG_CROSS", cfg.CROSS_ENCODER_MODEL))
                     if SCORER == "cross" else lex,
                     ret=CandidateRetriever(idx.kg, lex, cfg.RELS_PER_NODE_HOP1, cfg.RELS_PER_NODE_HOP2,
                                            cfg.TAILS_PER_RELATION_CAP, cfg.HOP2_BEAM, cfg.MAX_SEEDS,
                                            cfg.HUB_SAMPLE, cfg.MEET_PAIRS, cfg.MEET_CAP, cfg.MEET_MAX_EDGES))
    return _live

@app.route("/api/verify", methods=["POST"])
def verify():
    claim = (request.json or {}).get("claim", "").strip()
    if not claim:
        return jsonify(error="Write a claim first."), 400
    try:
        from llm_extractor import extract_mentions
        from ngram_fallback import ngram_candidates
        p = _pipeline()
        mentions = extract_mentions(claim)
        details = p["val"].validate_all(mentions, claim)
        if not mentions:
            details = [p["val"].finalize_entity(k, k, "ngram_fallback") for k in ngram_candidates(claim, p["idx"])]
        seeds = list(dict.fromkeys([d["canonical"] for d in details if d["canonical"]]
                                   + [a for d in details for a in d.get("alternates", [])]))
        cands = list(p["ret"].retrieve(claim, seeds, k=2))
        scores = p["scorer"].score_triples(claim, [triple_text(*t) for t in cands])
        top = [cands[i] for i in sorted(range(len(cands)), key=lambda i: -scores[i])[:N_FACTS]]
        text = ask(build_messages(claim, top))
        return jsonify(claim=claim, seeds=seeds, n_candidates=len(cands), triples=top,
                       entities=[{k: d[k] for k in ("mention", "canonical", "method")} for d in details],
                       pred=parse_verdict(text), response=text)
    except Exception as exc:  # LLM down, KG path wrong...
        return jsonify(error=f"{type(exc).__name__}: {exc}. Is the Llama server running and FACTKG_DATA_DIR set?"), 500

ATTACKS = {
 "linguistic": "Rewrite the claim with intricate syntax, passive voice and redundant entities, keeping its meaning exactly the same.",
 "semantic": "Change exactly one fact of the claim (an entity, relation or number) into a plausible but different alternative (e.g. a neighbouring place, a similar number) so that the claim becomes false. Keep the style.",
 "multihop": "Rewrite the claim so it looks true when one fact is checked alone but becomes false when a second, related fact is followed (e.g. right person, wrong attribute of a related entity).",
}

@app.route("/api/attack", methods=["POST"])
def attack():
    d = request.json or {}
    if d.get("mode") not in ATTACKS or not d.get("claim"):
        return jsonify(error="Pick an attack mode and a source claim."), 400
    msgs = [{"role": "system", "content": "You rewrite claims for robustness testing. " + ATTACKS[d["mode"]] + " Answer with the new claim only."},
            {"role": "user", "content": d["claim"]}]
    text = ask(msgs, max_tokens=120)
    if not text:
        return jsonify(error="The Llama server did not answer."), 500
    return jsonify(claim=text.strip().strip('"'), expected="SUPPORTED" if d["mode"] == "linguistic" else "REFUTED")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
