"""Phase 2 - Turn KG nodes/relations/triples into text for the rerankers."""

import re

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def relation_text(rel: str) -> str:
    """'~placeOfBirth' -> 'place of birth' (direction is dropped on purpose)."""
    rel = rel.lstrip("~")
    return _CAMEL.sub(" ", rel.replace("_", " ")).lower().strip()


def entity_text(node: str) -> str:
    return node.strip('"').replace("_", " ").strip()


def triple_text(head: str, rel: str, tail: str) -> str:
    return f"{entity_text(head)} | {relation_text(rel)} | {entity_text(tail)}"
