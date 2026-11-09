"""
Phase 1 - Optional diagnostic: DBpedia Lookup API fallback.

NOT used in the main validation path. Rationale: the live DBpedia has
drifted from the 2015 dump the local KG is built on, so a live-API match
does not guarantee the entity exists in
dbpedia_2015_undirected_light.pickle. Using it as the primary validator
would silently accept entities the rest of the pipeline (retrieval,
reasoning) can never actually query.

This module is instead meant for interactive error analysis, to
distinguish two failure modes behind an "unmatched" mention:
  (a) the LLM extracted a genuinely non-existent/garbled entity, or
  (b) the LLM extracted a real, current DBpedia entity that simply isn't
      present in the local (light, 2015) KG subset used by this project.
"""

import requests

LOOKUP_URL = "https://lookup.dbpedia.org/api/search"


def lookup_entity(mention: str, max_results: int = 3, timeout: int = 10) -> list:
    """
    Query the live DBpedia Lookup API for a mention.
    Returns a list of candidate DBpedia resource URIs, or [] on failure.

    This makes a live network call — use it interactively for error
    analysis on a handful of "unmatched" mentions, not inside large batch
    evaluation runs (config.ENABLE_LOOKUP_API_FALLBACK is False by
    default for this reason).
    """
    mention = mention.strip().strip('"')
    try:
        resp = requests.get(
            LOOKUP_URL,
            params={"query": mention, "format": "json", "maxResults": max_results},
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return [doc.get("resource", [None])[0] for doc in data.get("docs", [])]
    except (requests.RequestException, ValueError) as exc:
        print(f"[lookup_fallback] Lookup API failed for {mention!r}: {exc}")
        return []


def diagnose_unmatched(validation_results: dict, max_checks: int = 50) -> dict:
    """
    Given the output of pipeline.run_entity_linking, sample up to
    max_checks 'unmatched' entity mentions and check them against the
    live Lookup API, to split failures into
    'exists_in_live_dbpedia_not_local' vs 'likely_bad_extraction'.
    """
    checked = 0
    diagnosis = {"exists_in_live_dbpedia_not_local": [], "likely_bad_extraction": []}

    for claim_text, r in validation_results.items():
        for d in r["validation_details"]:
            if d["method"] != "unmatched" or d["type"] != "entity":
                continue
            if checked >= max_checks:
                return diagnosis
            candidates = lookup_entity(d["mention"])
            checked += 1
            if candidates:
                diagnosis["exists_in_live_dbpedia_not_local"].append(
                    {"claim": claim_text, "mention": d["mention"], "candidates": candidates}
                )
            else:
                diagnosis["likely_bad_extraction"].append(
                    {"claim": claim_text, "mention": d["mention"]}
                )

    return diagnosis
