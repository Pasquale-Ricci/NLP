"""
Phase 1 - Fallback candidate generation (only used when the LLM returns
nothing, e.g. for questions or colloquial claims).

Scans claim n-grams (longest first, up to 6 tokens), keeps those that match
a KG entity exactly (after normalization), never reusing a token. Single
tokens must be capitalized and not a common function word; multi-token spans
must start with a capital letter or a digit.
"""

import re

_TOKEN = re.compile(r"[\w][\w'\u2019.\-]*")

STOP_SINGLE = {
    "what", "who", "whom", "which", "where", "when", "how", "why", "the", "a",
    "an", "is", "are", "was", "were", "be", "he", "she", "it", "they", "his",
    "her", "its", "their", "yes", "no", "well", "oh", "and", "but", "or", "do",
    "does", "did", "you", "i", "we", "there", "this", "that", "these", "those",
    "if", "one", "also", "however", "not", "has", "have", "had", "in", "on",
    "of", "at", "to", "for", "by", "with", "from", "as", "so", "yea", "yeah",
    "hes", "shes", "name", "did",
}


def _clean(tok: str) -> str:
    tok = re.sub(r"['\u2019]s$", "", tok)
    return tok.rstrip(".,;:!?'\u2019-") if len(tok) > 3 else tok


def ngram_candidates(claim: str, kg_index, max_n: int = 6) -> list:
    tokens = [_clean(t) for t in _TOKEN.findall(claim)]
    tokens = [t for t in tokens if t]
    used = [False] * len(tokens)
    found = []
    for n in range(min(max_n, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            if any(used[i:i + n]):
                continue
            span = tokens[i:i + n]
            first = span[0]
            if n == 1:
                if not first[0].isupper() or first.lower() in STOP_SINGLE:
                    continue
            elif not (first[0].isupper() or first[0].isdigit()):
                continue
            elif all(t.lower() in STOP_SINGLE for t in span):
                continue
            key = kg_index.exact_lookup("_".join(span))
            if key:
                found.append(key)
                for j in range(i, i + n):
                    used[j] = True
    return list(dict.fromkeys(found))
