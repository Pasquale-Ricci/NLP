"""
Phase 1 - Orchestration: run LLM extraction + local validation over a
FactKG split and produce a predicted Entity_set per claim.
"""

import pickle
import random

from tqdm import tqdm

from llm_extractor import extract_mentions
from local_kg_index import LocalKGIndex
from entity_validator import EntityValidator
from ngram_fallback import ngram_candidates


def load_claims(pickle_path: str) -> dict:
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def run_entity_linking(
    claims: dict,
    kg_index: LocalKGIndex,
    limit: int = None,
    shuffle: bool = False,
    seed: int = 42,
) -> dict:
    """
    claims: dict as loaded from a FactKG pickle, {claim_text: {...}}
            (see FactKG_README.md for the value schema).

    Returns:
        {
          claim_text: {
            "predicted_entities": [...],   # validated, KG-grounded set (strict)
            "seed_entities": [...],        # predicted + alternate KG spellings
            "raw_mentions": [...],         # raw LLM output, pre-validation
            "validation_details": [...],   # per-mention resolution trace
            "gold_entities": [...],        # from FactKG, for evaluation
            "types": [...],                # FactKG claim metadata
          },
          ...
        }
    """
    validator = EntityValidator(kg_index)
    results = {}

    items = list(claims.items())
    if shuffle:
        # FactKG splits are ordered by reasoning type, so a plain
        # head-of-file --limit gives a biased sample.
        random.Random(seed).shuffle(items)
    if limit:
        items = items[:limit]

    for claim_text, gold in tqdm(items, desc="Entity Linking"):
        mentions = extract_mentions(claim_text)
        details = validator.validate_all(mentions, claim_text)
        if not mentions:
            # LLM returned nothing (questions, colloquial claims): fall back to
            # exact n-gram lookup of the claim in the KG.
            details = [
                validator.finalize_entity(k, k, "ngram_fallback")
                for k in ngram_candidates(claim_text, kg_index)
            ]

        predicted_entities = list(
            dict.fromkeys(d["canonical"] for d in details if d["canonical"] is not None)
        )
        # Seeds for Phase 2 = strict prediction + alternate KG spellings
        # (literal twins, year <-> full date, 1604 <-> 1604.0).
        seed_entities = list(
            dict.fromkeys(
                predicted_entities
                + [a for d in details for a in d.get("alternates", [])]
            )
        )
        # unmatched mentions are dropped from the final predicted set,
        # but kept in validation_details for error analysis
        # (see lookup_fallback.diagnose_unmatched).

        results[claim_text] = {
            "predicted_entities": predicted_entities,
            "seed_entities": seed_entities,
            "raw_mentions": mentions,
            "validation_details": details,
            "gold_entities": gold.get("Entity_set", []),
            "types": gold.get("types", []),
        }

    return results
