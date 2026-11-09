"""Attack generation with the local Llama: the three tactics of the project design."""
import os, re
import requests
import config

ATTACKS = {
 "linguistic": "Rewrite the claim with intricate syntax, passive voice and redundant entities, keeping its meaning, entities and numbers exactly the same.",
 "semantic": "Change exactly ONE fact of the claim (an entity, a relation or a number) into a plausible but different alternative (e.g. a neighbouring place, a similar number) so that the claim becomes false. Keep everything else and the style.",
 "multihop": "Rewrite the claim so it looks true when one fact is checked alone but becomes false when a second, related fact is followed (e.g. right person, wrong attribute of a related entity).",
}

def _clean(text):
    t = (text or "").strip().split("\n")[0].strip()
    t = re.sub(r"^(new claim|rewritten claim|claim)\s*:\s*", "", t, flags=re.I)
    return t.strip().strip('"').strip()

def generate_attack(claim, mode, model=None, temperature=0.3):
    """Returns the attacked claim, or None. `model` (or FACTKG_ATTACK_MODEL) lets the
    generator differ from the verifier, removing the generator/target circularity."""
    model = model or os.environ.get("FACTKG_ATTACK_MODEL") or config.LLM_MODEL_NAME
    msgs = [{"role": "system", "content": "You rewrite claims for robustness testing. " + ATTACKS[mode] + " Answer with the new claim only, on one line."},
            {"role": "user", "content": claim}]
    try:
        r = requests.post(f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
                          json={"model": model, "messages": msgs, "temperature": temperature, "max_tokens": 120},
                          headers={"Authorization": f"Bearer {config.LLM_API_KEY}"}, timeout=config.LLM_TIMEOUT_S * 2)
        r.raise_for_status()
        return _clean(r.json()["choices"][0]["message"]["content"]) or None
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return None
