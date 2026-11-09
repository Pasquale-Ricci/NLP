"""
Phase 1 - Step C: Resolve LLM-extracted mentions against the local KG.

Resolution order (quotes from the LLM are NOT trusted, the KG arbitrates
entity-vs-literal):
  1. dates (month name, ISO, or slash)   -> ISO literal; "10/03/1983" is read
                                            US-style (MM/DD) unless only the
                                            DD/MM form exists in the KG
  2. number                              -> literal grounded to an existing KG key
  3. entity (exact / normalized / disambiguated by suffix + claim context)
  4. string literal present in the KG
  5. quoted by the LLM but unknown       -> literal passthrough
  6. strict fuzzy entity match
  7. unmatched -> retries: strip leading filler word ('Well_X', 'The_X'),
                  split 'City,_Country' into two entities

Every result also carries `alternates`: other KG nodes that spell the same
thing (entity <-> literal twin, '1977' <-> '1977-01-01', '1604' <-> '1604.0').
They are NOT part of the strict Entity_set prediction but are used as extra
seeds for Phase 2 retrieval.
"""

import re

from local_kg_index import LocalKGIndex, _normalize

_MONTHS = {}
for _i, _name in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], 1):
    _MONTHS[_name] = _i
    _MONTHS[_name[:3]] = _i
_MONTHS["sept"] = 9

_DMY = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\.?,?\s+(\d{4})$")
_MDY = re.compile(r"^([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})$")
_SLASH = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NUMBER = re.compile(r"^[+-]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?$")

FILLERS = {"the", "a", "an", "well", "yes", "no", "yea", "yeah", "oh", "so",
           "and", "but", "also"}


def to_iso_date(text: str):
    """Return 'YYYY-MM-DD' if text is a full date with a month name, else None."""
    t = text.replace("_", " ").strip()
    m = _DMY.match(t)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
    else:
        m = _MDY.match(t)
        if not m:
            return None
        month, day, year = m.group(1), m.group(2), m.group(3)
    month_num = _MONTHS.get(month.lower())
    if month_num is None or not (1 <= int(day) <= 31):
        return None
    return f"{year}-{month_num:02d}-{int(day):02d}"


def slash_date_candidates(text: str):
    """'10/03/1983' -> ['1983-10-03' (US MM/DD), '1983-03-10' (EU DD/MM)] (valid ones only)."""
    m = _SLASH.match(text.strip())
    if not m:
        return None
    a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
    cands = []
    if 1 <= a <= 12 and 1 <= b <= 31:
        cands.append(f"{y}-{a:02d}-{b:02d}")
    if 1 <= b <= 12 and 1 <= a <= 31:
        cands.append(f"{y}-{b:02d}-{a:02d}")
    return list(dict.fromkeys(cands)) or None


def numeric_candidates(text: str):
    """Candidate literal texts for a number, or None if not numeric."""
    t = text.strip()
    if not _NUMBER.match(t):
        return None
    t = t.replace(",", "")
    cands = [t]
    if "." in t:
        head, tail = t.split(".", 1)
        if set(tail) <= {"0"}:
            cands.append(head)
    else:
        cands.append(t + ".0")
    return list(dict.fromkeys(cands))


def _result(mention, typ, canonical, method, score=None):
    return {"mention": mention, "type": typ, "canonical": canonical,
            "method": method, "score": score, "alternates": []}


class EntityValidator:
    def __init__(self, kg_index: LocalKGIndex):
        self.kg_index = kg_index

    # ---- alternates (extra seeds) -------------------------------------
    def _alternates(self, d: dict) -> list:
        c = d["canonical"]
        if c is None:
            return []
        kg = self.kg_index.kg
        alts = []
        if d["type"] == "entity":
            alts += self.kg_index.literal_twins(c)
        else:
            t = c.strip('"')
            if re.fullmatch(r"\d{4}", t):
                alts.append(f'"{t}-01-01"')
            elif _ISO.match(t):
                alts.append(f'"{t[:4]}"')
            nums = numeric_candidates(t)
            if nums:
                alts += [f'"{n}"' for n in nums]
        return [a for a in dict.fromkeys(alts) if a != c and a in kg]

    def finalize_entity(self, key, mention, method, score=100):
        """Build a result dict for an already-resolved entity key."""
        d = _result(mention, "entity", key, method, score)
        d["alternates"] = self._alternates(d)
        return d

    # ---- core resolution ----------------------------------------------
    def _resolve(self, mention: str, claim_norm: str) -> dict:
        raw = mention.strip()
        quoted = LocalKGIndex.is_literal_mention(raw)
        text = raw.strip('"').strip()
        if not text:
            return _result(raw, "entity", None, "unmatched")
        kg = self.kg_index

        iso = to_iso_date(text)
        if iso:
            return _result(raw, "literal", f'"{iso}"', "literal_date")
        if _ISO.match(text):
            return _result(raw, "literal", f'"{text}"', "literal_date")
        slash = slash_date_candidates(text)
        if slash:
            grounded = kg.lookup_literal(slash)
            if grounded:
                return _result(raw, "literal", grounded, "literal_date_kg")
            return _result(raw, "literal", f'"{slash[0]}"', "literal_date")

        nums = numeric_candidates(text)
        if nums:
            grounded = kg.lookup_literal(nums)
            if grounded:
                return _result(raw, "literal", grounded, "literal_numeric_kg")
            return _result(raw, "literal", f'"{nums[0]}"', "literal_numeric")

        key, method = kg.resolve_entity(text, claim_norm)
        if key is not None:
            if method != "disambiguated":
                method = "exact" if key == text else method
            return _result(raw, "entity", key, method, 100)

        spaced = text.replace("_", " ")
        lit = kg.lookup_literal([text, spaced])
        if lit:
            return _result(raw, "literal", lit, "literal_kg")

        if quoted:
            return _result(raw, "literal", f'"{spaced}"', "literal_passthrough")

        fuzzy_match, score = kg.fuzzy_lookup(text)
        if fuzzy_match is not None:
            return _result(raw, "entity", fuzzy_match, "fuzzy", score)

        return _result(raw, "entity", None, "unmatched", score)

    def validate_mention(self, mention: str, claim_norm: str = "") -> dict:
        d = self._resolve(mention, claim_norm)
        d["alternates"] = self._alternates(d)
        return d

    # ---- retries for unmatched mentions --------------------------------
    def _retry_unmatched(self, d: dict, claim_norm: str) -> list:
        text = d["mention"].strip().strip('"').strip()
        tokens = [t for t in re.split(r"[_\s]+", text) if t]

        i = 0
        while i < len(tokens) - 1 and tokens[i].lower() in FILLERS:
            i += 1
        if i > 0:
            key, _ = self.kg_index.resolve_entity("_".join(tokens[i:]), claim_norm)
            if key:
                return [self.finalize_entity(key, d["mention"], "stripped_filler")]

        if "," in text:
            parts = [p.strip(" _") for p in text.split(",") if p.strip(" _")]
            if len(parts) > 1:
                keys = [self.kg_index.resolve_entity(p, claim_norm)[0] for p in parts]
                if all(keys):
                    return [self.finalize_entity(k, p, "comma_split")
                            for k, p in zip(keys, parts)]
        return [d]

    def validate_all(self, mentions: list, claim: str = None) -> list:
        claim_norm = f" {_normalize(claim)} " if claim else ""
        out = []
        for m in mentions:
            # defensive: the LLM sometimes packs several literals in one item,
            # e.g. '"January_1", "1942"'
            parts = re.split(r'"\s*,\s*"', m.strip()) if '", "' in m else [m]
            for p in parts:
                d = self.validate_mention(p, claim_norm)
                if d["method"] == "unmatched" and d["type"] == "entity":
                    out.extend(self._retry_unmatched(d, claim_norm))
                else:
                    out.append(d)
        return out
