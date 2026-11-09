"""Full pipeline on a free-text claim: entity linking -> retrieval -> ranking -> Llama verdict.
Loads the KG on first use (minutes, several GB). Same logic as the Sandbox tab of app.py."""
import os
import config, config_phase2 as cfg
from llm_verifier import ask, build_messages, parse_verdict
from triple_verbalizer import triple_text

class LivePipeline:
    def __init__(self, scorer=None, n_facts=20):
        from local_kg_index import LocalKGIndex
        from entity_validator import EntityValidator
        from candidate_retrieval import CandidateRetriever
        from rerankers import LexicalScorer, CrossEncoderScorer
        self.idx = LocalKGIndex(config.KG_PICKLE)
        self.val = EntityValidator(self.idx)
        lex = LexicalScorer()
        self.scorer = CrossEncoderScorer(os.environ.get("FACTKG_CROSS", cfg.CROSS_ENCODER_MODEL)) \
            if (scorer or os.environ.get("FACTKG_SCORER", "cross")) == "cross" else lex
        self.ret = CandidateRetriever(self.idx.kg, lex, cfg.RELS_PER_NODE_HOP1, cfg.RELS_PER_NODE_HOP2,
                                      cfg.TAILS_PER_RELATION_CAP, cfg.HOP2_BEAM, cfg.MAX_SEEDS, cfg.HUB_SAMPLE,
                                      cfg.MEET_PAIRS, cfg.MEET_CAP, cfg.MEET_MAX_EDGES)
        self.n = n_facts

    def verify(self, claim):
        from llm_extractor import extract_mentions
        from ngram_fallback import ngram_candidates
        mentions = extract_mentions(claim)
        details = self.val.validate_all(mentions, claim)
        if not mentions:
            details = [self.val.finalize_entity(k, k, "ngram_fallback") for k in ngram_candidates(claim, self.idx)]
        seeds = list(dict.fromkeys([d["canonical"] for d in details if d["canonical"]]
                                   + [a for d in details for a in d.get("alternates", [])]))
        cands = list(self.ret.retrieve(claim, seeds, k=2))
        sc = self.scorer.score_triples(claim, [triple_text(*t) for t in cands])
        top = [cands[i] for i in sorted(range(len(cands)), key=lambda i: -sc[i])[: self.n]]
        text = ask(build_messages(claim, top))
        return {"pred": parse_verdict(text), "response": text, "seeds": seeds, "triples": top}
