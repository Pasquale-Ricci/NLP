"""
Phase 1 - Step A: Few-shot JSON entity/literal extraction using a local LLM.

The LLM is prompted to read a natural language FactKG claim and return a
JSON list of "mention" strings formatted the way FactKG/DBpedia expects:
  - named entities as DBpedia-style titles, spaces -> underscores
    (e.g. "Adam Mcquaid" -> "Adam_McQuaid")
  - literal values (numbers, years, quantities) wrapped in double quotes
    exactly as they appear in the claim (e.g. "94.8024" -> '"94.8024"')

This mirrors the 'Entity_set' field documented in FactKG_README.md.
The LLM's job here is *extraction*, not validation: validation against the
real DBpedia entity space happens later, in entity_validator.py.
"""

import json
import re
import time

import requests

import config

SYSTEM_PROMPT = """You are an entity and literal extraction system for a knowledge-graph fact-verification pipeline over DBpedia.
Given a claim, return every entity and literal value the claim explicitly mentions.
Rules:
1. Named entities (people, places, organisations, works, events, ...) are written as DBpedia page titles: spaces replaced by underscores, original capitalisation (e.g. 'barack obama' -> Barack_Obama). Never put quotes around an entity.
2. Keep multi-word names whole and prefer the longest full name (e.g. School_of_Applied_Arts_in_Stuttgart, not two separate items).
3. Literals are numbers, dates and explicit string values. Wrap each literal in double quotes and give each literal its own array item.
   - Numbers: no thousands separators ('1,063' -> "1063").
   - Dates: ISO format ('22 November 1988' -> "1988-11-22").
4. Do NOT extract relation words, verbs or common nouns that merely describe a relationship (e.g. successor, spouse, born, child, member).
5. Do NOT add entities that are not named in the claim.
6. Claims may be questions or colloquial ("What is the name of X's spouse?", "Well, X ..."): extract the entities exactly the same way, and never include filler words (Well, Yes, No, The) in a name.
Respond with ONLY a JSON array of strings. No explanation, no markdown fences, no extra keys."""

FEW_SHOT_EXAMPLES = [
    {
        "claim": "Adam McQuaid weighed 94.8024 kilograms and is from Pleasant Springs, Wisconsin.",
        "entities": ["Adam_McQuaid", '"94.8024"', "Pleasant_Springs,_Wisconsin"],
    },
    {
        "claim": "The band that Michael Stipe formed was active from 1980 until 2011.",
        "entities": ["Michael_Stipe", '"1980"', '"2011"'],
    },
    {
        "claim": "Her name is Ada Lovelace, and she had a child.",
        "entities": ["Ada_Lovelace"],
    },
    {
        "claim": "Grace Hopper, born on 9 December 1906, studied at Vassar College.",
        "entities": ["Grace_Hopper", '"1906-12-09"', "Vassar_College"],
    },
    {
        "claim": "The Eiffel Tower in Paris is 1,063 feet tall.",
        "entities": ["Eiffel_Tower", "Paris", '"1063"'],
    },
    {
        "claim": "What is the name of Rosalind Franklin's doctoral advisor?",
        "entities": ["Rosalind_Franklin"],
    },
    {
        "claim": "Well, Marie Curie was not born in Paris!",
        "entities": ["Marie_Curie", "Paris"],
    },
    {
        "claim": "Neither France nor Germany borders Japan.",
        "entities": ["France", "Germany", "Japan"],
    },
]


def _build_messages(claim: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["claim"]})
        messages.append({"role": "assistant", "content": json.dumps(ex["entities"])})
    messages.append({"role": "user", "content": claim})
    return messages


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _parse_json_array(raw_text: str):
    """Best-effort parse of a JSON array from a raw LLM response."""
    candidate = _strip_code_fences(raw_text)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except json.JSONDecodeError:
        pass

    # Fallback: grab the first [...] block anywhere in the text.
    match = re.search(r"\[.*\]", candidate, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass

    return None


def extract_mentions(claim: str) -> list:
    """
    Call the local LLM with few-shot prompting to extract candidate
    entity/literal mentions from a single claim.

    Returns a list of strings (empty list on repeated failure, so the
    pipeline can keep running over a batch without crashing).
    """
    payload = {
        "model": config.LLM_MODEL_NAME,
        "messages": _build_messages(claim),
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
    }
    url = f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}

    last_error = None
    for attempt in range(1, config.LLM_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=config.LLM_TIMEOUT_S
            )
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            mentions = _parse_json_array(raw_text)
            if mentions is not None:
                return mentions
            last_error = f"Could not parse JSON array from: {raw_text[:200]!r}"
        except (requests.RequestException, KeyError, IndexError) as exc:
            last_error = str(exc)

        time.sleep(min(2 ** attempt, 8))

    print(f"[llm_extractor] Extraction failed for claim after retries: {last_error}")
    return []
