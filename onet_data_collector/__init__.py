"""onet_data_collector — a small ETL toolkit for O*NET Web Services.

The package turns O*NET occupational data into tidy, analysis-ready tables in
three stages:

1. :func:`keyword_search` / :func:`keyword_search_many` — discover occupations
   from search terms.
2. :func:`fetch_job_details` — pull and preserve the full raw detail JSON.
3. :func:`condense_job_details` — flatten the raw JSON into a wide CSV.

:func:`download_bulk` / :func:`load_bulk` fetch and cache the *entire* O*NET
database release (no credentials) for offline modelling.

The lower-level :class:`OnetClient` handles auth, retries, timeouts, and
pagination and can be used directly for any O*NET endpoint.
"""

from __future__ import annotations

from .bulk import BulkRelease, download_bulk, load_bulk
from .client import OnetClient, OnetWebService
from .condensed_job_details import condense_job_details
from .config import resolve_credentials
from .exceptions import (
    OnetAuthError,
    OnetConfigError,
    OnetError,
    OnetHTTPError,
    OnetValidationError,
)
from .job_details import fetch_job_details
from .keyword_search import keyword_search, keyword_search_many

__version__ = "0.2.0"

__all__ = [
    "OnetClient",
    "OnetWebService",
    "keyword_search",
    "keyword_search_many",
    "fetch_job_details",
    "condense_job_details",
    "resolve_credentials",
    "BulkRelease",
    "download_bulk",
    "load_bulk",
    "OnetError",
    "OnetConfigError",
    "OnetHTTPError",
    "OnetAuthError",
    "OnetValidationError",
    "__version__",
]
