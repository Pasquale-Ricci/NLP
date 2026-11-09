"""
Phase 2 - Evaluation of retrieved triples against FactKG's gold `Evidence`.

Gold evidence is a set of relation paths starting at each claim entity, e.g.
  {'A': [['child','birthYear'],['spouse']], 'B': [['~spouse']],
   '"1998"': [['~birthYear','~child']]}
(a leading '~' means "traverse the relation backwards"). Gold does not list
triples, so we check whether the retrieved triples REALIZE each path:

  path_recall     fraction of gold paths for which the retrieved graph contains
                  at least one complete walk (start entity -> relation 1 -> ...)
  edge_precision  fraction of retrieved triples that lie on some complete walk
                  of some gold path
  f1              harmonic mean of the two
  full_cover      1 if EVERY gold path is realized (the claim is fully supported
                  by the retrieved evidence), else 0
  relation_recall fraction of gold relation names present among retrieved triples

Precision is low by construction when N is large (gold needs 1-3 edges), so
metrics are reported at several cutoffs N.
"""

from collections import defaultdict


def parse_gold_paths(evidence):
    """-> list of (entity, (rel, rel, ...)). Handles nested lists of paths and
    the flat form (a flat list of strings is ONE path)."""
    paths = []
    for ent, val in (evidence or {}).items():
        if not val:
            continue
        if all(isinstance(x, str) for x in val):
            paths.append((ent, tuple(val)))
            continue
        for p in val:
            if isinstance(p, str):
                paths.append((ent, (p,)))
            elif p:
                paths.append((ent, tuple(p)))
    return paths


def _realize(start, path, out, inc, end=None):
    """Edges used by complete walks of `path` from `start`, or None.
    If `end` is given the walk must finish exactly there."""
    layers, steps = [{start}], []
    for rel in path:
        nxt, edges = set(), set()
        if rel.startswith("~"):
            r = rel[1:]
            for u in layers[-1]:
                for v in inc.get((u, r), ()):
                    nxt.add(v)
                    edges.add((v, r, u))
        else:
            for u in layers[-1]:
                for v in out.get((u, rel), ()):
                    nxt.add(v)
                    edges.add((u, rel, v))
        if not nxt:
            return None
        layers.append(nxt)
        steps.append(edges)

    if end is not None:
        layers[-1] = layers[-1] & {end}
        if not layers[-1]:
            return None
    good, used = layers[-1], set()           # backward prune: keep complete walks only
    for i in range(len(path) - 1, -1, -1):
        keep, new_good = set(), set()
        for e in steps[i]:
            h, _, t = e
            u, v = (t, h) if path[i].startswith("~") else (h, t)
            if v in good:
                keep.add(e)
                new_good.add(u)
        used |= keep
        good = new_good
    return used



def _inv(rel):
    return rel[1:] if rel.startswith("~") else "~" + rel


def path_end_nodes(gold_paths):
    """If the REVERSE of a path from entity e is listed under another entity e2,
    the path connects e and e2 and must END at e2. -> {(e, path): e2}"""
    ends = {}
    for e, p in gold_paths:
        rev = tuple(_inv(r) for r in reversed(p))
        for e2, p2 in gold_paths:
            if e2 != e and p2 == rev:
                ends[(e, p)] = e2
                break
    return ends


def evaluate_triples(triples, gold_paths, strict=False):
    """triples: iterable of (head, rel, tail) in forward direction.
    strict=True requires each path that connects two claim entities to end at the
    other entity (default: any end node counts, as in the earlier runs).
    Returns a metrics dict, or None when there is no gold evidence."""
    if not gold_paths:
        return None
    triples = list(triples)
    out, inc = defaultdict(set), defaultdict(set)
    for h, r, t in triples:
        out[(h, r)].add(t)
        inc[(t, r)].add(h)

    realized, used_edges, flags = 0, set(), []
    ends = path_end_nodes(gold_paths) if strict else {}
    for ent, path in gold_paths:
        used = _realize(ent, path, out, inc, ends.get((ent, path)))
        flags.append(used is not None)
        if used is not None:
            realized += 1
            used_edges |= used

    gold_rels = {r.lstrip("~") for _, p in gold_paths for r in p}
    got_rels = {r for _, r, _ in triples}
    recall = realized / len(gold_paths)
    precision = len(used_edges) / len(triples) if triples else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_paths": len(gold_paths),
        "path_recall": recall,
        "edge_precision": precision,
        "f1": f1,
        "full_cover": 1.0 if realized == len(gold_paths) else 0.0,
        "relation_recall": len(gold_rels & got_rels) / len(gold_rels) if gold_rels else 0.0,
        "flags": flags,                              # per gold path: realized?
        "lengths": [len(p) for _, p in gold_paths],
    }


METRIC_KEYS = ("path_recall", "edge_precision", "f1", "full_cover", "relation_recall")


def aggregate(rows):
    """Mean of each metric over a list of metric dicts (None entries skipped)."""
    rows = [r for r in rows if r]
    if not rows:
        return {"n": 0}
    agg = {k: sum(r[k] for r in rows) / len(rows) for k in METRIC_KEYS}
    agg["n"] = len(rows)
    return agg


def positive_edges(triples, gold_paths, lenient_fallback=True):
    """Union of the edges lying on complete walks of the gold paths (ending at the
    other claim entity when the path connects two): the triples a reranker should
    rank first. If a path cannot reach its end entity (refuted claims) and
    lenient_fallback is on, the walk ending anywhere is used. Empty set otherwise."""
    out, inc = defaultdict(set), defaultdict(set)
    for h, r, t in triples:
        out[(h, r)].add(t)
        inc[(t, r)].add(h)
    used = set()
    ends = path_end_nodes(gold_paths)
    for ent, path in gold_paths:
        end = ends.get((ent, path))
        u = _realize(ent, path, out, inc, end)
        if not u and lenient_fallback and end is not None:
            # False claims built by substitution list the substituted entity, which the
            # KG does not connect to the start node: the useful evidence is what the KG
            # DOES say along that relation chain, so accept a walk ending anywhere.
            u = _realize(ent, path, out, inc, None)
        if u:
            used |= u
    return used
