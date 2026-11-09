"""
Phase 1 - Step B: Local DBpedia index for entity validation.

Design choice: rather than relying on the live DBpedia Lookup API (which
reflects *current* DBpedia, while FactKG's graph is a static 2015 dump),
we build a local lookup index directly from the KG pickle. This guarantees
every validated entity actually exists in the graph the rest of the
pipeline (retrieval, reasoning) will query.

Scale notes (the light KG has ~9.9M keys):
  - Literal nodes (keys starting with a double quote, e.g. '"1142"') are
    NOT in the entity index; they are looked up separately and exactly.
  - Entities with a disambiguation suffix ('Aleksey_Chirikov_(icebreaker)')
    are also indexed under their bare name, so a mention without suffix can
    be resolved to them (with claim context / degree as tie-breakers).
  - Fuzzy matching is restricted to a block of candidates sharing the same
    3-character normalized prefix, uses the strict `fuzz.ratio` scorer and
    is built lazily on first use.
"""

import pickle
import re
import unicodedata

import config

try:
    from rapidfuzz import process, fuzz

    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

BLOCK_PREFIX_LEN = 3

_DROP_CHARS = re.compile(r"[.'\u2019]")
_NON_WORD = re.compile(r"[^\w\s]")
_MULTI_SPACE = re.compile(r"\s+")
_SUFFIX = re.compile(r"_\(([^)]*)\)$")


def _merge_initials(tokens):
    """['a','s','gubbio'] -> ['as','gubbio'] (consecutive single letters)."""
    out, buf = [], ""
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            buf += t
        else:
            if buf:
                out.append(buf)
                buf = ""
            out.append(t)
    if buf:
        out.append(buf)
    return out


def _normalize(name: str) -> str:
    """
    Normalize an entity string for exact/fuzzy matching.
    'Wheeler,_Texas' == 'Wheeler Texas'; 'U.S._Route_83' == 'US Route 83';
    'A._S._Gubbio' == 'A.S._Gubbio'; accents and case are ignored.
    """
    name = name.strip().strip('"')
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.replace("_", " ")
    name = _DROP_CHARS.sub("", name)
    name = _NON_WORD.sub(" ", name)
    name = _MULTI_SPACE.sub(" ", name).lower().strip()
    tokens = name.split()
    if any(len(t) == 1 and t.isalpha() for t in tokens):
        tokens = _merge_initials(tokens)
    return " ".join(tokens)


def strip_suffix(key: str) -> str:
    return _SUFFIX.sub("", key)


def _is_literal_key(key: str) -> bool:
    return key.startswith('"')


class LocalKGIndex:
    def __init__(self, kg_path: str = config.KG_PICKLE):
        with open(kg_path, "rb") as f:
            self.kg = pickle.load(f)

        self._norm_index = {}    # normalized name -> [entity keys]
        self._suffix_index = {}  # normalized bare name -> [keys with '_(...)' suffix]
        self.num_entities = 0
        self.num_literals = 0
        for ent in self.kg:
            if _is_literal_key(ent):
                self.num_literals += 1
                continue
            self.num_entities += 1
            norm = _normalize(ent)
            bucket = self._norm_index.get(norm)
            if bucket is None:
                self._norm_index[norm] = [ent]
            else:
                bucket.append(ent)
            if ent.endswith(")"):
                m = _SUFFIX.search(ent)
                if m:
                    self._suffix_index.setdefault(_normalize(ent[: m.start()]), []).append(ent)

        self._blocks = None  # built lazily, only if fuzzy matching is used

        if not _HAS_RAPIDFUZZ:
            print(
                "[local_kg_index] rapidfuzz not installed - fuzzy matching "
                "disabled, falling back to exact/normalized matching only."
            )

    # ---- basic helpers -------------------------------------------------
    def exists(self, entity_key: str) -> bool:
        return entity_key in self.kg

    def _degree(self, key: str) -> int:
        return len(self.kg.get(key, ()))

    def _pick(self, cands, mention):
        """Choose among entities sharing a normalized name."""
        if len(cands) == 1:
            return cands[0]
        target = mention.replace(" ", "_").lower()
        for c in cands:
            if c.lower() == target:
                return c
        return max(cands, key=self._degree)

    # ---- entity resolution ---------------------------------------------
    def exact_lookup(self, mention: str):
        """Exact or normalized match against ENTITY keys (no suffix logic)."""
        mention = mention.strip().strip('"')
        if mention in self.kg and not _is_literal_key(mention):
            return mention
        cands = self._norm_index.get(_normalize(mention))
        return self._pick(cands, mention) if cands else None

    def resolve_entity(self, mention: str, claim_norm: str = ""):
        """
        Resolve a mention to an entity key, using disambiguation suffixes.
        claim_norm is the normalized claim wrapped in spaces (' ... '), used
        to pick between candidates whose suffix words appear in the claim.
        Returns (key, method) or (None, None);
        method in {'exact', 'normalized', 'disambiguated'}.
        """
        mention = mention.strip().strip('"')
        base, method = None, None
        if mention in self.kg and not _is_literal_key(mention):
            base, method = mention, "exact"
        else:
            cands = self._norm_index.get(_normalize(mention))
            if cands:
                base, method = self._pick(cands, mention), "normalized"

        suffixed = self._suffix_index.get(_normalize(mention), [])
        if suffixed:
            scored = []
            for s in suffixed:
                words = set(_normalize(_SUFFIX.search(s).group(1)).split())
                overlap = sum(1 for w in words if f" {w} " in claim_norm)
                scored.append((overlap, self._degree(s), s))
            scored.sort(reverse=True)
            best_overlap, _, best = scored[0]
            if base is None or best_overlap > 0:
                return best, "disambiguated"
        if base is not None:
            return base, method
        return None, None

    # ---- literals ----------------------------------------------------------
    def lookup_literal(self, texts):
        """First candidate text that exists as a literal node, as its quoted key."""
        for t in texts:
            key = f'"{t}"'
            if key in self.kg:
                return key
        return None

    def literal_twins(self, entity_key: str):
        """Literal nodes that spell the same thing as an entity, e.g.
        United_States -> '"United States"'."""
        bare = strip_suffix(entity_key)
        texts = [entity_key.replace("_", " "), bare.replace("_", " "), entity_key]
        out = []
        for t in dict.fromkeys(texts):
            key = f'"{t}"'
            if key in self.kg:
                out.append(key)
        return out

    # ---- fuzzy -------------------------------------------------------------
    def _ensure_blocks(self):
        if self._blocks is not None:
            return
        print("[local_kg_index] Building fuzzy-match blocks (one-time)...")
        blocks = {}
        for norm in self._norm_index:
            blocks.setdefault(norm[:BLOCK_PREFIX_LEN], []).append(norm)
        self._blocks = blocks

    def fuzzy_lookup(self, mention: str, threshold: int = None):
        """Strict fuzzy match (fuzz.ratio) within the same prefix block.
        Returns (canonical_key, score) or (None, 0)."""
        if not _HAS_RAPIDFUZZ:
            return None, 0
        threshold = threshold if threshold is not None else config.FUZZY_MATCH_THRESHOLD
        self._ensure_blocks()
        norm = _normalize(mention)
        block = self._blocks.get(norm[:BLOCK_PREFIX_LEN])
        if not block:
            return None, 0
        result = process.extractOne(norm, block, scorer=fuzz.ratio, score_cutoff=threshold)
        if result is None:
            return None, 0
        matched_norm, score, _ = result
        return self._pick(self._norm_index[matched_norm], mention), score

    @staticmethod
    def is_literal_mention(mention: str) -> bool:
        """Literals in FactKG are quoted strings, e.g. '"94802.4"'."""
        m = mention.strip()
        return m.startswith('"') and m.endswith('"') and len(m) > 1
