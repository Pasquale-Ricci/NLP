"""
Phase 2 - Scorers used to rank candidate triples against the claim.

  lexical : token-overlap baseline (no model, no GPU)
  bi      : Bi-Encoder cosine similarity (claim and triple encoded separately)
  cross   : Cross-Encoder (claim and triple read jointly, cross-attention)
  cascade : bi-encoder keeps the top-K, cross-encoder re-scores only those

All expose:
  score_triples(claim, texts)    -> list[float]  (higher = more relevant)
  score_relations(claim, rels)   -> list[float]  (only lexical / bi; used to
                                                  pre-filter relations during
                                                  expansion)
"""

import math
import re

import numpy as np

from triple_verbalizer import relation_text

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in _WORD.findall(text.lower())}


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


get_device = _device


def _require_sentence_transformers():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required for the 'bi', 'cross' and 'cascade' "
            "scorers: pip install sentence-transformers"
        ) from exc


class LexicalScorer:
    name = "lexical"

    def __init__(self):
        self._claim = (None, set())

    def _ct(self, claim):
        if self._claim[0] != claim:
            self._claim = (claim, _tokens(claim))
        return self._claim[1]

    def score_relations(self, claim, rels):
        ct = self._ct(claim)
        out = []
        for r in rels:
            rt = _tokens(relation_text(r))
            out.append(len(ct & rt) / max(len(rt), 1))
        return out

    def score_triples(self, claim, texts):
        ct = self._ct(claim)
        return [len(ct & (tt := _tokens(t))) / math.sqrt(max(len(tt), 1)) for t in texts]


class BiEncoderScorer:
    name = "bi"

    def __init__(self, model_name, device=None):
        _require_sentence_transformers()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device or _device())
        self._rel_cache = {}
        self._claim = (None, None)

    def _encode(self, texts, batch_size=256):
        return np.asarray(self.model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False))

    def _claim_emb(self, claim):
        if self._claim[0] != claim:
            self._claim = (claim, self._encode([claim])[0])
        return self._claim[1]

    def score_relations(self, claim, rels):
        if not rels:
            return []
        missing = [r for r in rels if r not in self._rel_cache]
        if missing:
            embs = self._encode([relation_text(r) for r in missing])
            for r, e in zip(missing, embs):
                self._rel_cache[r] = e
        mat = np.stack([self._rel_cache[r] for r in rels])
        return (mat @ self._claim_emb(claim)).tolist()

    def score_triples(self, claim, texts):
        if not texts:
            return []
        return (self._encode(texts) @ self._claim_emb(claim)).tolist()


class CrossEncoderScorer:
    name = "cross"

    def __init__(self, model_name, device=None, batch_size=64):
        _require_sentence_transformers()
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device or _device())
        self.batch_size = batch_size

    def score_relations(self, claim, rels):
        raise NotImplementedError("use 'bi' or 'lexical' as the relation pre-filter")

    def score_triples(self, claim, texts):
        if not texts:
            return []
        preds = self.model.predict([(claim, t) for t in texts],
                                   batch_size=self.batch_size, show_progress_bar=False)
        return [float(x) for x in np.asarray(preds).reshape(-1)]


class CascadeScorer:
    name = "cascade"

    def __init__(self, bi, cross, k):
        self.bi, self.cross, self.k = bi, cross, k

    def score_relations(self, claim, rels):
        return self.bi.score_relations(claim, rels)

    def score_triples(self, claim, texts):
        n = len(texts)
        if n == 0:
            return []
        bi_scores = self.bi.score_triples(claim, texts)
        top = sorted(range(n), key=lambda i: -bi_scores[i])[: self.k]
        cross_scores = self.cross.score_triples(claim, [texts[i] for i in top])
        out = [-1e6 + b for b in bi_scores]      # never reranked: below all
        for i, s in zip(top, cross_scores):
            out[i] = s
        return out


class ScorerFactory:
    def __init__(self, bi_model, cross_model, cascade_k):
        self.bi_model, self.cross_model, self.cascade_k = bi_model, cross_model, cascade_k
        self._cache = {}

    def get(self, name):
        if name not in self._cache:
            if name == "lexical":
                s = LexicalScorer()
            elif name == "bi":
                s = BiEncoderScorer(self.bi_model)
            elif name == "cross":
                s = CrossEncoderScorer(self.cross_model)
            elif name == "cascade":
                s = CascadeScorer(self.get("bi"), self.get("cross"), self.cascade_k)
            else:
                raise ValueError(f"unknown scorer: {name}")
            self._cache[name] = s
        return self._cache[name]
