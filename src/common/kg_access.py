"""Phase 2 - Thin helpers around the FactKG DBpedia adjacency dict
(kg[entity][relation] -> list of tails; '~rel' = inverse relation)."""

import pickle


def load_kg(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def is_literal(node: str) -> bool:
    return node.startswith('"')


def canonical_triple(u: str, rel: str, v: str):
    """The KG stores every edge in both directions. Return the forward
    (head, relation, tail) so each fact appears exactly once."""
    if rel.startswith("~"):
        return (v, rel[1:], u)
    return (u, rel, v)


def total_edges(kg: dict, node: str) -> int:
    return sum(len(v) for v in kg.get(node, {}).values())


def walk_full_kg(kg, start, path, max_frontier=20000, end=None):
    """Can `path` (a sequence of relation keys, '~rel' = inverse) be walked in the FULL KG
    from `start`? If `end` is given the walk must finish there.
    True / False, or None when the frontier exceeded max_frontier (treat as realizable)."""
    cur = {start}
    for rel in path:
        nxt = set()
        for u in cur:
            nxt.update(kg.get(u, {}).get(rel, ()))
            if len(nxt) > max_frontier:
                return None
        if not nxt:
            return False
        cur = nxt
    return end in cur if end is not None else True
