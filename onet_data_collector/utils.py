"""Utility helpers used across the package.

Includes the original ``get_user_input`` / ``check_for_error`` helpers (kept for
backwards compatibility) plus small helpers for normalising O*NET's
inconsistent JSON, where a field may be absent, a single object, or a list.
"""

from __future__ import annotations

from typing import Any, Mapping

from .exceptions import OnetValidationError


def get_user_input(prompt: str) -> str:
    """Prompt until the user enters a non-empty value (legacy helper)."""
    result = ""
    while not result:
        result = input(f"{prompt}: ").strip()
    return result


def check_for_error(service_result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Raise :class:`OnetValidationError` if an O*NET result carries an error.

    Historically this called ``sys.exit`` on error. It now raises a typed
    exception (still terminating unhandled callers) so it can be caught and
    handled programmatically. Returns the result unchanged when it is clean.
    """
    if isinstance(service_result, Mapping) and "error" in service_result:
        raise OnetValidationError(str(service_result["error"]), payload=dict(service_result))
    return service_result


def as_list(value: Any) -> list[Any]:
    """Coerce an O*NET field into a list.

    O*NET may return ``None`` (absent), a single object, or a list for the same
    logical field depending on how many items exist. This normalises all three.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def dig(mapping: Any, *keys: str, default: Any = None) -> Any:
    """Safely walk a chain of dict keys, returning ``default`` if any is missing."""
    current = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current
