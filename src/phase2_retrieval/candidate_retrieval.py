"""
Phase 2 - Candidate triple retrieval: capped k-hop expansion around the seeds.

Why capped: DBpedia hubs (countries, years, 'United_States') have hundreds of
thousands of neighbours, so a plain 2-hop expansion explodes. Controls:
  - relation pre-filter: only the top-R relations of a node (scored against the
    claim with a cheap scorer) are expanded;
  - hub rule: if a relation has more than TAILS_PER_RELATION_CAP tails, only
    the tails that are themselves seeds (plus `hub_sample` witness tails) are kept (the edge between two seeds is
    what a claim usually states; the other tails are noise);
  - beam: only the endpoints of the best hop-1 triples are expanded at hop 2,
    and literal nodes are never expanded beyond hop 1;
  - bridging: direct edges between two seeds are always added;
  - meet-in-the-middle (optional): for two seeds a, b, nodes X with a -ra-> X -rb-> b
    are found by INTERSECTING their neighbour lists, which works even when both
    seeds are hubs (the unnamed middle node cannot be enumerated from one end).
Every triple is stored once, in forward direction.
"""

from collections import defaultdict

from kg_access import canonical_triple, is_literal, total_edges
from triple_verbalizer import triple_text


class CandidateRetriever:
    def __init__(self, kg, prefilter, rels_hop1, rels_hop2, tails_cap, beam, max_seeds,
                 hub_sample=0, meet_pairs=0, meet_cap=3, meet_max_edges=300000):
        self.kg = kg
        self.prefilter = prefilter
        self.rels_hop1, self.rels_hop2 = rels_hop1, rels_hop2
        self.tails_cap, self.beam, self.max_seeds = tails_cap, beam, max_seeds
        self.hub_sample = hub_sample
        self.meet_pairs, self.meet_cap, self.meet_max_edges = meet_pairs, meet_cap, meet_max_edges

    def _bridge(self, seeds):
        """Direct edges between pairs of seeds (scan from the lighter node)."""
        kg, out = self.kg, set()
        for i, a in enumerate(seeds):
            for b in seeds[i + 1:]:
                src, dst = (a, b) if total_edges(kg, a) <= total_edges(kg, b) else (b, a)
                for rel, tails in kg[src].items():
                    if dst in tails:
                        out.add(canonical_triple(src, rel, dst))
        return out

    def _meet(self, claim, seeds):
        """Unnamed middle nodes X with a -ra-> X -rb-> b for pairs of seeds.
        Cost O(edges(a)+edges(b)) per pair; relation pairs are ranked by the
        pre-filter and at most meet_pairs x meet_cap middle nodes are kept."""
        kg, out = self.kg, set()
        if self.meet_pairs <= 0:
            return out
        pool = [s for s in seeds if total_edges(kg, s) <= self.meet_max_edges]
        for i, a in enumerate(pool):
            for b in pool[i + 1:]:
                a_, b_ = (a, b) if total_edges(kg, a) <= total_edges(kg, b) else (b, a)
                inv = defaultdict(list)                  # X -> relations of a_ reaching X
                for ra, tails in kg[a_].items():
                    for x in tails:
                        inv[x].append(ra)
                pairs = {}                               # (ra, rb) -> [X, ...]
                for rb, tails in kg[b_].items():
                    for x in tails:
                        ras = inv.get(x)
                        if ras and x != a_ and x != b_:
                            for ra in ras:
                                lst = pairs.setdefault((ra, rb), [])
                                if len(lst) < self.meet_cap:
                                    lst.append(x)
                if not pairs:
                    continue
                rels = list({r for pair in pairs for r in pair})
                sc = dict(zip(rels, self.prefilter.score_relations(claim, rels)))
                best = sorted(pairs, key=lambda p: -(sc[p[0]] + sc[p[1]]))[: self.meet_pairs]
                for ra, rb in best:
                    for x in pairs[(ra, rb)]:
                        out.add(canonical_triple(a_, ra, x))
                        out.add(canonical_triple(b_, rb, x))
        return out

    def _expand(self, claim, nodes, seedset, n_rels):
        kg, out = self.kg, set()
        for u in nodes:
            rels = kg.get(u)
            if not rels:
                continue
            names = list(rels.keys())
            if len(names) > n_rels:
                scores = self.prefilter.score_relations(claim, names)
                names = [names[i] for i in
                         sorted(range(len(names)), key=lambda i: -scores[i])[:n_rels]]
            for rel in names:
                tails = rels[rel]
                if len(tails) <= self.tails_cap:
                    chosen = tails
                else:   # hub relation: seed tails + a few witness tails
                    chosen = [x for x in tails if x in seedset]
                    chosen += [x for x in tails[: self.hub_sample] if x not in seedset]
                for x in chosen:
                    out.add(canonical_triple(u, rel, x))
        return out

    def retrieve(self, claim, seeds, k=2):
        """Returns {triple: hop} where hop 0 = bridging edge, 1/2 = expansion hop."""
        kg = self.kg
        seeds = [s for s in dict.fromkeys(seeds) if s in kg][: self.max_seeds]
        seedset = set(seeds)
        cands = {t: 0 for t in self._bridge(seeds)}
        for t in self._meet(claim, seeds):
            cands.setdefault(t, 0)
        frontier, visited = list(seeds), set(seeds)

        for hop in range(1, k + 1):
            new = self._expand(claim, frontier, seedset,
                               self.rels_hop1 if hop == 1 else self.rels_hop2)
            for t in new:
                cands.setdefault(t, hop)
            if hop == k:
                break
            items = list(new)
            if not items:
                break
            scores = self.prefilter.score_triples(claim, [triple_text(*t) for t in items])
            frontier = []
            for i in sorted(range(len(items)), key=lambda i: -scores[i]):
                h, _, t = items[i]
                for node in (h, t):
                    if node not in visited and node in kg and not is_literal(node):
                        visited.add(node)
                        frontier.append(node)
                if len(frontier) >= self.beam:
                    break
        return cands
