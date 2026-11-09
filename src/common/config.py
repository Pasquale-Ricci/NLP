"""
Configuration for Phase 1: Entity & Literal Extraction.

Assumption: the local Llama model is served through an OpenAI-compatible
chat-completions endpoint. This covers the most common local serving
setups:
  - Ollama (OpenAI-compat route, usually http://localhost:11434/v1)
  - llama.cpp server (built-in OpenAI-compatible server)
  - vLLM (OpenAI server mode)
  - text-generation-webui (openai extension)
  - LM Studio local server

If your setup differs, only LLM_BASE_URL / LLM_MODEL_NAME below need to
change (or, for a non-OpenAI-style API, the request payload built in
llm_extractor.py).
"""

import os

# --- Local LLM (Llama) serving endpoint ---------------------------------
LLM_BASE_URL = os.environ.get("FACTKG_LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL_NAME = os.environ.get("FACTKG_LLM_MODEL", "llama3")
LLM_API_KEY = os.environ.get("FACTKG_LLM_API_KEY", "not-needed")  # most local servers ignore this
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 512
LLM_TIMEOUT_S = 60
LLM_MAX_RETRIES = 3

# --- Data paths -----------------------------------------------------------
DATA_DIR = os.environ.get("FACTKG_DATA_DIR", "./data")
TRAIN_PICKLE = os.path.join(DATA_DIR, "factkg_train.pickle")
DEV_PICKLE = os.path.join(DATA_DIR, "factkg_dev.pickle")
TEST_PICKLE = os.path.join(DATA_DIR, "factkg_test.pickle")

# Prefer the light KG (only relations actually used in FactKG) for the
# local entity index: smaller, faster to load, and sufficient for
# validating entity mentions extracted from FactKG claims.
KG_PICKLE = os.path.join(DATA_DIR, "dbpedia_2015_undirected_light.pickle")

# --- Entity validation ------------------------------------------------
FUZZY_MATCH_THRESHOLD = 92  # rapidfuzz ratio score, 0-100 (strict on purpose)
ENABLE_LOOKUP_API_FALLBACK = False  # diagnostic only, see lookup_fallback.py

# --- Output -------------------------------------------------------------
OUTPUT_DIR = os.environ.get("FACTKG_OUTPUT_DIR", "./phase1_outputs")
