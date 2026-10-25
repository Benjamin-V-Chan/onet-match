"""Exception hierarchy for onet_data_collector.

All errors raised by the package derive from :class:`OnetError`, so callers can
catch everything with a single ``except OnetError``.
"""

from __future__ import annotations


class OnetError(Exception):
    """Base class for every error raised by this package."""


class OnetConfigError(OnetError):
    """Raised when credentials or configuration are missing or invalid."""


class OnetHTTPError(OnetError):
    """Raised when an O*NET request fails at the HTTP/transport layer.

    Attributes:
        url: The request URL that failed.
        status_code: HTTP status code, if the failure produced one.
    """

    def __init__(self, message: str, *, url: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.url = url
        self.status_code = status_code


class OnetAuthError(OnetHTTPError):
    """Raised for authentication/authorization failures (HTTP 401/403)."""


class OnetValidationError(OnetError):
    """Raised when the O*NET API returns a 422 validation error body."""

    def __init__(self, message: str, *, url: str | None = None, payload: dict | None = None):
        super().__init__(message)
        self.url = url
        self.payload = payload or {}
