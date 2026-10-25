"""Credential resolution for O*NET Web Services.

Credentials are resolved from, in order of precedence:

1. Explicit ``username`` / ``password`` arguments.
2. Environment variables ``ONET_USERNAME`` / ``ONET_PASSWORD``.
3. A ``.env`` file in the current directory (or a path passed to
   :func:`load_dotenv`), parsed without any third-party dependency.
4. Interactive prompt (only when ``allow_prompt=True`` and running on a TTY).

This keeps secrets out of source code and makes both the library and the CLI
usable in scripts, notebooks, and CI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .exceptions import OnetConfigError

ENV_USERNAME = "ONET_USERNAME"
ENV_PASSWORD = "ONET_PASSWORD"


def load_dotenv(path: str | os.PathLike[str] = ".env", *, override: bool = False) -> dict[str, str]:
    """Parse a ``.env`` file into ``os.environ`` and return the parsed values.

    Supports ``KEY=VALUE`` lines, ``#`` comments, blank lines, ``export KEY=``
    prefixes, and single/double quoted values. Missing files are ignored.
    """
    parsed: dict[str, str] = {}
    file_path = Path(path)
    if not file_path.is_file():
        return parsed

    for raw in file_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


def resolve_credentials(
    username: str | None = None,
    password: str | None = None,
    *,
    dotenv_path: str | os.PathLike[str] | None = ".env",
    allow_prompt: bool = False,
) -> tuple[str, str]:
    """Resolve O*NET credentials, raising :class:`OnetConfigError` if not found.

    Args:
        username: Explicit username (highest precedence).
        password: Explicit password.
        dotenv_path: Optional ``.env`` file to load before reading env vars.
            Pass ``None`` to skip ``.env`` loading.
        allow_prompt: If ``True`` and stdin is a TTY, prompt interactively for
            any missing value.
    """
    if dotenv_path is not None:
        load_dotenv(dotenv_path)

    username = username or os.environ.get(ENV_USERNAME)
    password = password or os.environ.get(ENV_PASSWORD)

    if allow_prompt and sys.stdin.isatty():
        if not username:
            username = _prompt("Enter O*NET Web Services username")
        if not password:
            import getpass

            password = getpass.getpass("Enter O*NET Web Services password: ").strip()

    if not username or not password:
        raise OnetConfigError(
            "O*NET credentials not found. Set the "
            f"{ENV_USERNAME}/{ENV_PASSWORD} environment variables, add them to a "
            ".env file, or pass them explicitly. Register for free at "
            "https://services.onetcenter.org/developer/signup"
        )
    return username, password


def _prompt(text: str) -> str:
    result = ""
    while not result:
        result = input(f"{text}: ").strip()
    return result
