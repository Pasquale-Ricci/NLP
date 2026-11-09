"""
Phase 2 settings (candidate retrieval + reranking). Paths are reused from
config.py (FACTKG_DATA_DIR, KG_PICKLE).
"""

import os

PHASE2_OUTPUT_DIR = os.environ.get("FACTKG_PHASE2_OUTPUT_DIR", "./phase2_outputs")

# Pretrained rerankers (zero-shot). Both are small (~22M params) and fit
# easily in 8 GB of VRAM next to everything else.
BI_ENCODER_MODEL = os.environ.get(
    "FACTKG_BI_ENCODER", "sentence-transformers/all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.environ.get(
    "FACTKG_CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- k-hop expansion with caps (avoids the combinatorial explosion) ------
# Defaults = the 'combo' configuration of ablate_candidates.py (dev sample of 300
# claims, gold seeds: recall on KG-realizable paths 0.850 -> 0.973). Previous
# defaults: rels1=15, rels2=6, beam=30, hub_sample=0, meet_pairs=0.
MAX_SEEDS = 10                  # seeds beyond this are ignored
RELS_PER_NODE_HOP1 = 30         # relations kept per seed (scored vs the claim)
RELS_PER_NODE_HOP2 = 10          # relations kept per frontier node at hop 2
TAILS_PER_RELATION_CAP = 50     # if a relation has more tails (a "hub"), only
                                # tails that are themselves seeds are kept
HOP2_BEAM = 60                  # frontier nodes expanded at hop 2
MEET_PAIRS = 10                  # meet-in-the-middle: relation pairs kept per pair of seeds (0 = off)
MEET_CAP = 3                    # intermediate nodes kept per relation pair
MEET_MAX_EDGES = 300000         # seeds with more edges than this are skipped
HUB_SAMPLE = 3                  # extra non-seed tails kept for a hub relation
                                # (witness edges for 'existence' claims); 0 = off

# --- reranking / evaluation ----------------------------------------------
CASCADE_K = 200                 # bi-encoder keeps top-K, cross-encoder rescored
MAX_SAVED = 50                  # ranked triples saved per claim and scorer
EVAL_CUTOFFS = (5, 10, 20, 30, 50)
