"""
Phase 3a - LLM-based verification: serialize retrieved KG triples, ask the local
Llama (Chain-of-Thought) and parse a SUPPORTED / REFUTED verdict.

Talks to the same OpenAI-compatible endpoint as Phase 1 (config.LLM_BASE_URL,
config.LLM_MODEL_NAME; override with FACTKG_LLM_BASE_URL / FACTKG_LLM_MODEL).
"""

import re
import time

import requests

import config
from triple_verbalizer import triple_text

SYSTEM_PROMPT = """You are a fact-checking system. You verify a claim against facts retrieved from the DBpedia knowledge graph (2015 version).
Rules:
- Facts are triples written 'subject | relation | object'. Use ONLY these facts, never outside knowledge.
- Closed-world assumption: the claim is SUPPORTED only if the facts support everything it states. If a stated fact is missing, or the facts give a different value or a different entity, the claim is REFUTED.
- Negated claims ("did not", "is not", "neither") are SUPPORTED when the negated fact is absent or contradicted by the facts, and REFUTED when the facts confirm it.
- Numbers and dates must match the facts; formatting differences (1,147 vs 1147.0) do not matter.
- Several facts may be irrelevant: ignore them.
Reason briefly (at most 4 short numbered steps), then end with exactly one last line: VERDICT: SUPPORTED or VERDICT: REFUTED."""

# (claim, facts, answer) - invented examples, none taken from FactKG.
FEW_SHOT = [
    ("Rosa Ortiz was born in a city in Chile.",
     [("Rosa_Ortiz", "birthPlace", "Valparaíso"), ("Valparaíso", "country", "Chile"),
      ("Rosa_Ortiz", "occupation", "Painter")],
     "1. The claim says Rosa Ortiz was born in a city located in Chile.\n"
     "2. Fact 1 gives her birth place as Valparaíso and fact 2 says Valparaíso is in Chile.\n"
     "3. Both parts are supported.\nVERDICT: SUPPORTED"),
    ("The Hawk Tower is 52 metres tall and is located in Norway.",
     [("Hawk_Tower", "height", "48.0"), ("Hawk_Tower", "location", "Norway")],
     "1. The claim states a height of 52 and a location in Norway.\n"
     "2. The facts give a height of 48.0, which differs from 52; the location matches.\n"
     "3. One stated fact is contradicted.\nVERDICT: REFUTED"),
    ("Kai Lindqvist did not play for HC Nord.",
     [("Kai_Lindqvist", "team", "HC_Sud")],
     "1. The claim says Kai Lindqvist did not play for HC Nord.\n"
     "2. The only team fact is HC Sud; no fact says he played for HC Nord.\n"
     "3. The negated fact is absent, so the claim holds.\nVERDICT: SUPPORTED"),
    ("Mara Velasco is a sculptor and her spouse is Tomas Brandt.",
     [("Mara_Velasco", "occupation", "Sculptor")],
     "1. The claim states an occupation (sculptor) and a spouse (Tomas Brandt).\n"
     "2. The occupation is confirmed, but no fact mentions a spouse.\n"
     "3. A stated fact is missing, so it cannot be supported.\nVERDICT: REFUTED"),
]


def format_user(claim, triples):
    if triples:
        facts = "\n".join(f"{i}. {triple_text(*t)}" for i, t in enumerate(triples, 1))
    else:
        facts = "(no facts retrieved)"
    return f"Claim: {claim}\n\nFacts:\n{facts}"


def build_messages(claim, triples, few_shot=True):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for c, facts, answer in FEW_SHOT:
            msgs.append({"role": "user", "content": format_user(c, facts)})
            msgs.append({"role": "assistant", "content": answer})
    msgs.append({"role": "user", "content": format_user(claim, triples)})
    return msgs


_VERDICT = re.compile(r"VERDICT\s*:?\s*[*_\s]*(SUPPORTED|REFUTED)", re.I)
_WORD = re.compile(r"\b(SUPPORTED|REFUTED)\b", re.I)


def parse_verdict(text):
    """'SUPPORTED' / 'REFUTED' / None. Uses the LAST 'VERDICT:' line; falls back to
    the last occurrence of either word."""
    if not text:
        return None
    m = _VERDICT.findall(text)
    if m:
        return m[-1].upper()
    w = _WORD.findall(text)
    return w[-1].upper() if w else None


def ask(messages, max_tokens=300, temperature=0.0, retries=3):
    """One chat-completion call. Returns the text, or None after repeated failures."""
    url = f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions"
    payload = {"model": config.LLM_MODEL_NAME, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=config.LLM_TIMEOUT_S * 3)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            last = exc
            time.sleep(min(2 ** attempt, 8))
    print(f"[llm_verifier] request failed after {retries} attempts: {last}")
    return None
