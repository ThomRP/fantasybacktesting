"""Checksum-verified, atomic downloads with polite retries and resumable caches."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Callable
from urllib.parse import urlsplit
import requests

LOGGER = logging.getLogger(__name__)
RETRYABLE = {408, 425, 429, 500, 502, 503, 504}

class CacheIntegrityError(ValueError):
    """A cache no longer matches its recorded content or request identity."""


class CachedDownloader:
    """Sequential, TLS-verified HTTP downloads with bounded transient retries.

    An immutable payload and its SHA256/provenance sidecar are committed via
    atomic renames. A pending metadata file lets a run recover if interrupted
    between committing the payload and committing its final metadata. There is
    no force-refresh option: remove/archive a cache explicitly to refresh it.
    """

    def __init__(self, *, session=None, min_interval: float = 1.0,
                 retries: int = 3, timeout: float = 45.0,
                 sleep: Callable[[float], None] = time.sleep):
        if min_interval < 0 or retries < 0 or timeout <= 0:
            raise ValueError("Invalid downloader limits")
        self.session = session if session is not None else requests.Session()
        self.min_interval, self.retries, self.timeout = min_interval, retries, timeout
        self.sleep = sleep
        self._last_request = None

    @staticmethod
    def _metadata_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".metadata.json")

    @staticmethod
    def _atomic_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
                name = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, path)
        finally:
            if name is not None and os.path.exists(name):
                os.unlink(name)

    def get(self, url: str, path: Path, *, params: dict | None = None,
            headers: dict | None = None,
            validator: Callable[[bytes], None] | None = None) -> Path:
        """Fetch once; validate both downloaded and reused cache contents."""
        path = Path(path)
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise ValueError("Only credential-free HTTPS source URLs may be cached")
        request_identity = {"url": url, "params": params or {}}
        meta_path = self._metadata_path(path)
        pending_path = meta_path.with_suffix(meta_path.suffix + ".pending")
        if path.exists():
            # Recover only our fully staged metadata, never invent provenance.
            metadata_source = meta_path if meta_path.exists() else pending_path
            if not metadata_source.exists():
                raise CacheIntegrityError(f"Missing cache provenance for {path}")
            metadata = json.loads(metadata_source.read_text())
            payload = path.read_bytes()
            if metadata.get("sha256") != hashlib.sha256(payload).hexdigest():
                raise CacheIntegrityError(f"Cache checksum mismatch: {path}")
            if any(metadata.get(k) != v for k, v in request_identity.items()):
                raise CacheIntegrityError(f"Cache request identity mismatch: {path}")
            if validator:
                validator(payload)
            if metadata_source == pending_path:
                os.replace(pending_path, meta_path)
            LOGGER.info("Cache hit: %s", path)
            return path
        for attempt in range(self.retries + 1):
            if self._last_request is not None:
                self.sleep(max(0.0, self.min_interval - (time.monotonic() - self._last_request)))
            try:
                self._last_request = time.monotonic()
                response = self.session.get(url, params=params, headers=headers,
                                            timeout=self.timeout)
                if response.status_code in RETRYABLE and attempt < self.retries:
                    delay = min(60.0, 2.0 ** attempt)
                    retry_after = response.headers.get("Retry-After", "")
                    if retry_after.isdigit():
                        delay = min(60.0, max(delay, float(retry_after)))
                    LOGGER.warning("Transient HTTP %s; retrying in %.1fs", response.status_code, delay)
                    self.sleep(delay)
                    continue
                response.raise_for_status()
                payload = response.content
                if validator:
                    validator(payload)
                metadata = {**request_identity,
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "bytes": len(payload)}
                self._atomic_bytes(pending_path, json.dumps(metadata, indent=2, sort_keys=True).encode())
                self._atomic_bytes(path, payload)
                os.replace(pending_path, meta_path)
                LOGGER.info("Cached %s (%d bytes)", path, len(payload))
                return path
            except (requests.Timeout, requests.ConnectionError) as exc:
                # A denied CONNECT is a policy/access problem, not a transient timeout.
                if attempt >= self.retries or "403" in str(exc):
                    raise
                self.sleep(min(60.0, 2.0 ** attempt))
        raise AssertionError("Unreachable retry loop")
