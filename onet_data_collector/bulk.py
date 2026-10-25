"""Loader for the full O*NET database bulk download.

O*NET publishes its complete database (every occupation, title, task, rating,
...) as a versioned zip at https://www.onetcenter.org/database.html. Unlike the
Web Services API this needs no credentials and returns everything at once, which
is what offline modelling (e.g. semantic occupation matching) needs.

    from onet_data_collector.bulk import download_bulk, load_bulk

    release = download_bulk()            # downloads + caches the current release
    release = load_bulk()                # later: reuse the cached copy
    occupations = release.table("occupation_data")

Design notes
------------
* **Cached outside the repo.** The default cache is
  ``~/.cache/onet_data_collector/bulk/db_<version>/`` (override with
  ``ONET_BULK_DIR`` or ``cache_dir=``), so ~100 MB of data never lands in git.
* **Provenance.** Each cached release carries a ``manifest.json`` recording the
  release version, source URL, SHA-256 of the zip, and download timestamp.
* **Stable logical names across releases.** O*NET renames files between
  releases (31.0 split ``Skills.txt`` into ``Essential Skills.txt`` +
  ``Transferable Skills.txt`` and replaced ``Alternate Titles.txt`` with
  ``Job Titles.txt``). :data:`TABLES` maps one logical name to every known file
  name, and columns are normalised to ``snake_case``.
* Stdlib ``urllib`` + ``zipfile`` only (plus pandas for the tables), with an
  injectable ``opener`` so tests run offline.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import re
import shutil
import ssl
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ._logging import get_logger
from .client import DEFAULT_USER_AGENT
from .exceptions import OnetConfigError, OnetHTTPError

log = get_logger("bulk")

DATABASE_PAGE_URL = "https://www.onetcenter.org/database.html"
BULK_URL_TEMPLATE = "https://www.onetcenter.org/dl_files/database/db_{tag}_text.zip"
# Used when the database page cannot be reached or parsed.
FALLBACK_RELEASE = "31.0"
MANIFEST_NAME = "manifest.json"

# Logical table name -> candidate file names, newest naming first.
TABLES: dict[str, tuple[str, ...]] = {
    "occupation_data": ("Occupation Data.txt",),
    "job_titles": ("Job Titles.txt", "Alternate Titles.txt"),
    "reported_titles": ("Sample of Reported Titles.txt",),
    "task_statements": ("Task Statements.txt",),
    "task_ratings": ("Task Ratings.txt",),
    "tasks_to_dwas": ("Tasks to DWAs.txt",),
    "dwa_reference": ("GWAs to IWAs to DWAs.txt", "DWA Reference.txt"),
    "essential_skills": ("Essential Skills.txt",),
    "transferable_skills": ("Transferable Skills.txt",),
    "skills_legacy": ("Skills.txt",),
    "knowledge": ("Knowledge.txt",),
    "abilities": ("Abilities.txt",),
    "interests": ("Career Interest Types.txt", "Interests.txt"),
    "job_zones": ("Job Zones.txt",),
    "job_zone_reference": ("Job Zone Reference.txt",),
}

# Raw column name -> normalised name, for columns whose snake_case form is
# awkward or that were renamed between releases.
_COLUMN_RENAMES = {
    "o*net-soc code": "soc_code",
    "alternate title": "job_title",
    "reported job title": "job_title",
    "shown in my next move": "shown_in_my_next_move",
    "source(s)": "sources",
    "dwa element id": "dwa_id",
    "dwa element name": "dwa_title",
    "dwa title": "dwa_title",
    "iwa element id": "iwa_id",
    "gwa element id": "gwa_id",
}


def _snake(name: str) -> str:
    key = name.strip().lower()
    if key in _COLUMN_RENAMES:
        return _COLUMN_RENAMES[key]
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_")


def release_tag(release: str) -> str:
    """``"31.0"`` -> ``"31_0"`` (the form used in O*NET download URLs)."""
    if not re.fullmatch(r"\d+\.\d+", release):
        raise OnetConfigError(f"Release must look like '31.0', got {release!r}.")
    return release.replace(".", "_")


def default_cache_dir() -> Path:
    """``$ONET_BULK_DIR`` or ``~/.cache/onet_data_collector/bulk``."""
    env = os.environ.get("ONET_BULK_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "onet_data_collector" / "bulk"


def _default_opener() -> urllib.request.OpenerDirector:
    """An opener that verifies TLS with certifi's CA bundle when available.

    python.org macOS builds ship without system CA certificates until
    "Install Certificates.command" is run, which makes plain ``urllib`` fail
    with CERTIFICATE_VERIFY_FAILED. certifi (if installed) avoids that.
    """
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover - environment dependent
        context = ssl.create_default_context()
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))


def _fetch(url: str, opener, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    try:
        with opener.open(req, timeout=timeout) as handle:
            return handle.read()
    except urllib.error.HTTPError as exc:
        raise OnetHTTPError(f"HTTP {exc.code} fetching {url}", url=url, status_code=exc.code) from exc
    except urllib.error.URLError as exc:
        raise OnetHTTPError(f"Could not reach {url}: {exc.reason}", url=url) from exc


def latest_release(*, opener=None, timeout: float = 30.0) -> str:
    """Scrape the O*NET database page for the newest release number.

    Falls back to :data:`FALLBACK_RELEASE` if the page cannot be read.
    """
    opener = opener or _default_opener()
    try:
        html = _fetch(DATABASE_PAGE_URL, opener, timeout).decode("utf-8", "replace")
    except OnetHTTPError as exc:
        log.warning("Could not read %s (%s); assuming release %s.", DATABASE_PAGE_URL, exc, FALLBACK_RELEASE)
        return FALLBACK_RELEASE
    found = {(int(a), int(b)) for a, b in re.findall(r"db_(\d+)_(\d+)_", html)}
    if not found:
        log.warning("No release links found on %s; assuming %s.", DATABASE_PAGE_URL, FALLBACK_RELEASE)
        return FALLBACK_RELEASE
    major, minor = max(found)
    return f"{major}.{minor}"


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> list[str]:
    """Extract every ``.txt`` member flat into ``dest``, refusing path traversal."""
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    for member in zf.infolist():
        if member.is_dir():
            continue
        name = Path(member.filename).name
        if not name.lower().endswith(".txt") or name.startswith(".") or name in ("", ".", ".."):
            continue
        target = (dest / name).resolve()
        if target.parent != dest.resolve():  # pragma: no cover - defensive
            raise OnetConfigError(f"Refusing to extract suspicious member {member.filename!r}.")
        with zf.open(member) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
        names.append(name)
    return sorted(names)


@dataclass
class BulkRelease:
    """A downloaded, extracted O*NET database release on disk."""

    path: Path
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def release(self) -> str:
        return str(self.manifest.get("release", ""))

    def file_for(self, name: str) -> Path | None:
        """Path of the first existing file for logical table ``name``."""
        if name not in TABLES:
            raise KeyError(f"Unknown table {name!r}. Known: {sorted(TABLES)}")
        for candidate in TABLES[name]:
            p = self.path / candidate
            if p.is_file():
                return p
        return None

    def available(self) -> list[str]:
        """Logical tables present in this release."""
        return [name for name in TABLES if self.file_for(name) is not None]

    def table(self, name: str) -> pd.DataFrame:
        """Load logical table ``name`` with ``snake_case`` columns (all strings).

        ``"skills"`` is a virtual table: Essential + Transferable Skills on
        releases >= 31.0, or the legacy ``Skills.txt`` on older releases.
        """
        if name == "skills":
            parts = [self.table(n) for n in ("essential_skills", "transferable_skills")
                     if self.file_for(n) is not None]
            if not parts and self.file_for("skills_legacy") is not None:
                parts = [self.table("skills_legacy")]
            if not parts:
                raise FileNotFoundError(f"No skills tables in {self.path}.")
            return pd.concat(parts, ignore_index=True)
        path = self.file_for(name)
        if path is None:
            raise FileNotFoundError(
                f"Table {name!r} ({' / '.join(TABLES[name])}) not found in {self.path}."
            )
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
        df.columns = [_snake(c) for c in df.columns]
        return df


def _release_dir(cache_dir: Path, release: str) -> Path:
    return cache_dir / f"db_{release_tag(release)}"


def download_bulk(
    release: str | None = None,
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
    opener=None,
    timeout: float = 120.0,
    now: Callable[[], _dt.datetime] | None = None,
) -> BulkRelease:
    """Download (or reuse) an O*NET bulk release and return it.

    Args:
        release: e.g. ``"31.0"``. ``None`` discovers the latest release.
        cache_dir: Cache root (default :func:`default_cache_dir`).
        force: Re-download even if a cached copy exists.
        opener: Optional ``urllib`` opener (for tests).
        timeout: Download timeout in seconds.
        now: Clock override (for tests).
    """
    opener = opener or _default_opener()
    cache_dir = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    release = release or latest_release(opener=opener)
    dest = _release_dir(cache_dir, release)

    if not force and (dest / MANIFEST_NAME).is_file():
        log.info("Using cached O*NET %s at %s", release, dest)
        return load_bulk(release, cache_dir=cache_dir)

    url = BULK_URL_TEMPLATE.format(tag=release_tag(release))
    log.info("Downloading O*NET %s database from %s", release, url)
    blob = _fetch(url, opener, timeout)
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as exc:
        raise OnetHTTPError(f"Download from {url} is not a valid zip.", url=url) from exc

    if dest.exists():
        shutil.rmtree(dest)
    with zf:
        files = _safe_extract(zf, dest)

    manifest = {
        "release": release,
        "source_url": url,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
        "downloaded_at": (now or (lambda: _dt.datetime.now(_dt.timezone.utc)))().isoformat(),
        "files": files,
        "license": "CC BY 4.0 — https://www.onetcenter.org/license_db.html",
    }
    (dest / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Cached %d files for O*NET %s in %s", len(files), release, dest)
    return BulkRelease(dest, manifest)


def cached_releases(cache_dir: str | Path | None = None) -> list[str]:
    """Release versions present in the cache, oldest first."""
    root = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    out = []
    for p in root.glob("db_*"):
        m = p / MANIFEST_NAME
        if m.is_file():
            out.append(json.loads(m.read_text(encoding="utf-8"))["release"])
    return sorted(out, key=lambda r: tuple(int(x) for x in r.split(".")))


def load_bulk(release: str | None = None, *, cache_dir: str | Path | None = None) -> BulkRelease:
    """Open a cached release (the newest cached one when ``release`` is None)."""
    root = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    if release is None:
        releases = cached_releases(root)
        if not releases:
            raise FileNotFoundError(
                f"No O*NET bulk release cached under {root}. "
                "Run `onet-collect bulk download` first."
            )
        release = releases[-1]
    dest = _release_dir(root, release)
    manifest_path = dest / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"O*NET {release} is not cached under {root}. Run `onet-collect bulk download --release {release}`."
        )
    return BulkRelease(dest, json.loads(manifest_path.read_text(encoding="utf-8")))
