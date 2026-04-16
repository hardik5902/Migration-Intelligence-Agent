"""Lightweight artifact storage for large pipeline payloads.

Stores JSON artifacts on the local filesystem for development and can switch to
Google Cloud Storage in Cloud Run by setting ``ARTIFACT_BUCKET``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from google.cloud import storage

_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_ROOT = (
    Path("/tmp/migration_intel_artifacts")
    if os.environ.get("K_SERVICE")
    else _ROOT / "cache" / "artifacts"
)


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _artifact_bucket() -> str:
    return (os.environ.get("ARTIFACT_BUCKET") or "").strip()


def _artifact_prefix() -> str:
    return (os.environ.get("ARTIFACT_PREFIX") or "migration-intel").strip("/")


def artifact_root() -> str:
    bucket = _artifact_bucket()
    if bucket:
        return f"gs://{bucket}/{_artifact_prefix()}"
    return str(_LOCAL_ROOT)


def _local_path(session_id: str, name: str) -> Path:
    return _LOCAL_ROOT / session_id / f"{name}.json"


def _blob_name(session_id: str, name: str) -> str:
    return f"{_artifact_prefix()}/{session_id}/{name}.json"


_storage_client: storage.Client | None = None


def _get_storage_client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


def write_json_artifact(session_id: str, name: str, payload: Any) -> str:
    """Persist JSON-serializable payload and return a reference string."""
    bucket = _artifact_bucket()
    text = json.dumps(payload, default=_json_default)

    if bucket:
        blob_name = _blob_name(session_id, name)
        blob = _get_storage_client().bucket(bucket).blob(blob_name)
        blob.upload_from_string(text, content_type="application/json")
        return f"gs://{bucket}/{blob_name}"

    path = _local_path(session_id, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def read_json_artifact(ref: str) -> Any:
    """Load a JSON artifact from a local path or ``gs://`` reference."""
    if not ref:
        return None

    if ref.startswith("gs://"):
        parsed = urlparse(ref)
        bucket = parsed.netloc
        blob_name = parsed.path.lstrip("/")
        blob = _get_storage_client().bucket(bucket).blob(blob_name)
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text())

    path = Path(ref)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
