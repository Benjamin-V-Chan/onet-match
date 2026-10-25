"""HTTP client for the O*NET Web Services REST API.

The public API is :class:`OnetClient`. ``OnetWebService`` is kept as a
backwards-compatible alias of the same class.

Design notes
------------
* Built on the standard library (``urllib``) so the client has no hard runtime
  dependency beyond Python itself. ``pandas`` is only needed by the helpers that
  build tabular output.
* Every request is retried with exponential backoff on transient failures
  (network errors, HTTP 429, and HTTP 5xx), honouring a ``Retry-After`` header
  when the server sends one.
* Errors surface as typed exceptions (:mod:`onet_data_collector.exceptions`)
  rather than magic ``{"error": ...}`` dictionaries, though the legacy dict
  behaviour is still available via :meth:`OnetClient.call` for callers that used
  ``check_for_error``.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Iterable, Mapping

from ._logging import get_logger
from .exceptions import OnetAuthError, OnetHTTPError, OnetValidationError

log = get_logger("client")

DEFAULT_BASE_URL = "https://services.onetcenter.org/ws/"
DEFAULT_USER_AGENT = "python-onet-data-collector/1.0"
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

QueryItem = tuple[str, Any]


class OnetClient:
    """A small, resilient wrapper around O*NET Web Services.

    Args:
        username: O*NET Web Services username (register at
            https://services.onetcenter.org/developer/signup).
        password: O*NET Web Services password.
        version: Optional API version string (e.g. ``"1.9"``). ``None`` uses the
            latest version exposed at the default ``/ws/`` root.
        timeout: Per-request timeout in seconds.
        max_retries: Number of retries for transient failures (network, 429,
            5xx). ``0`` disables retrying.
        backoff_factor: Base for exponential backoff; the n-th retry sleeps
            ``backoff_factor * 2**(n-1)`` seconds (unless overridden by
            ``Retry-After``).
        user_agent: Value for the ``User-Agent`` header.
        opener: Optional custom ``urllib`` opener (mainly for testing).
        sleep: Sleep function used between retries (injectable for testing).
    """

    def __init__(
        self,
        username: str,
        password: str,
        *,
        version: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        user_agent: str = DEFAULT_USER_AGENT,
        opener: urllib.request.OpenerDirector | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not username or not password:
            from .exceptions import OnetConfigError

            raise OnetConfigError("Both username and password are required.")

        token = base64.standard_b64encode(f"{username}:{password}".encode()).decode()
        self._headers = {
            "User-Agent": user_agent,
            "Authorization": f"Basic {token}",
            "Accept": "application/json",
        }
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._opener = opener or urllib.request.build_opener()
        self._sleep = sleep
        self.set_version(version)

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    def set_version(self, version: str | None = None) -> None:
        """Point the client at a specific API version (or the latest)."""
        if version is None:
            self._url_root = DEFAULT_BASE_URL
        else:
            self._url_root = f"https://services.onetcenter.org/v{version}/ws/"

    def _build_url(self, path: str, query: Iterable[QueryItem]) -> str:
        url = self._url_root + path.lstrip("/")
        query = list(query)
        if query:
            url += "?" + urllib.parse.urlencode(query, doseq=True)
        return url

    # ------------------------------------------------------------------ #
    # Core request
    # ------------------------------------------------------------------ #
    def request(self, path: str, *query: QueryItem) -> dict[str, Any]:
        """Perform a GET request and return the parsed JSON body.

        Raises:
            OnetAuthError: on HTTP 401/403.
            OnetValidationError: on HTTP 422 (the parsed body is attached).
            OnetHTTPError: on other non-2xx responses or network failures that
                persist after all retries.
        """
        url = self._build_url(path, query)
        req = urllib.request.Request(url, data=None, headers=self._headers, method="GET")

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._opener.open(req, timeout=self.timeout) as handle:
                    return json.load(handle)
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise OnetAuthError(
                        f"Authentication failed for {url} (HTTP {exc.code}). "
                        "Check your O*NET Web Services username/password.",
                        url=url,
                        status_code=exc.code,
                    ) from exc
                if exc.code == 422:
                    body = _safe_json(exc)
                    message = body.get("error") if isinstance(body, dict) else None
                    raise OnetValidationError(
                        message or f"O*NET returned HTTP 422 for {url}.",
                        url=url,
                        payload=body if isinstance(body, dict) else {},
                    ) from exc
                if exc.code in _RETRY_STATUSES and attempt < self.max_retries:
                    last_exc = exc
                    self._wait(attempt, exc)
                    continue
                raise OnetHTTPError(
                    f"Call to {url} failed with HTTP {exc.code}.",
                    url=url,
                    status_code=exc.code,
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    last_exc = exc
                    self._wait(attempt, None)
                    continue
                raise OnetHTTPError(
                    f"Call to {url} failed: {exc.reason}", url=url
                ) from exc

        # Should be unreachable, but keep the type checker and logic honest.
        raise OnetHTTPError(f"Call to {url} failed after retries.", url=url) from last_exc

    def call(self, path: str, *query: QueryItem) -> dict[str, Any]:
        """Legacy-compatible wrapper around :meth:`request`.

        Returns the parsed JSON on success. On error it returns
        ``{"error": "..."}`` instead of raising, matching the original
        ``OnetWebService.call`` contract used with ``check_for_error``.
        """
        try:
            return self.request(path, *query)
        except OnetValidationError as exc:
            # Preserve the validation body if present, else a generic error.
            return exc.payload or {"error": str(exc)}
        except OnetHTTPError as exc:
            return {"error": str(exc)}

    def _wait(self, attempt: int, http_error: urllib.error.HTTPError | None) -> None:
        delay = self.backoff_factor * (2 ** attempt)
        if http_error is not None:
            retry_after = http_error.headers.get("Retry-After") if http_error.headers else None
            if retry_after and retry_after.isdigit():
                delay = max(delay, float(retry_after))
        log.warning("Transient failure; retrying in %.1fs (attempt %d).", delay, attempt + 1)
        self._sleep(delay)

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    def about(self) -> dict[str, Any]:
        """Return the API ``about`` document (used as a connectivity check)."""
        return self.request("about")

    def paginate(
        self,
        path: str,
        *query: QueryItem,
        item_key: str,
        page_size: int = 50,
        max_items: int | None = None,
    ) -> list[dict[str, Any]]:
        """Follow O*NET ``start``/``end`` pagination and collect all items.

        O*NET list endpoints return ``total``, ``start`` and ``end`` fields plus
        a list under ``item_key`` (e.g. ``"occupation"``). This walks every page
        until ``total`` is reached or ``max_items`` is collected.
        """
        collected: list[dict[str, Any]] = []
        start = 1
        base_query = [(k, v) for k, v in query if k not in ("start", "end")]
        while True:
            end = start + page_size - 1
            page = self.request(path, *base_query, ("start", start), ("end", end))
            items = page.get(item_key) or []
            if isinstance(items, Mapping):  # single-item responses come back as a dict
                items = [items]
            collected.extend(items)

            total = _as_int(page.get("total"), default=len(collected))
            page_end = _as_int(page.get("end"), default=end)
            if max_items is not None and len(collected) >= max_items:
                return collected[:max_items]
            if not items or page_end >= total:
                break
            start = page_end + 1
        return collected


def _safe_json(handle: Any) -> Any:
    try:
        return json.load(handle)
    except (ValueError, OSError):
        return {}


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# Backwards-compatible alias for the original class name.
OnetWebService = OnetClient
