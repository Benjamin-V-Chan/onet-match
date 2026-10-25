"""Small text helpers shared by dataset building, retrieval and evaluation."""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN = re.compile(r"[a-z0-9]+")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace; used to dedupe texts across sources."""
    return _SPACE.sub(" ", str(text).strip().lower())


def _stem(token: str) -> str:
    """Very light plural stripping so "nurses" and "nurse" count as one token."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> set[str]:
    """Lowercased, stop-word-free, lightly stemmed alphanumeric tokens."""
    return {_stem(t) for t in _TOKEN.findall(str(text).lower()) if t not in ENGLISH_STOP_WORDS}


def shares_no_tokens(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` have no content token in common."""
    return not (content_tokens(a) & content_tokens(b))
