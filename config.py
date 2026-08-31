"""Application configuration.

Everything that used to be a literal in the source is read from the
environment here, with defaults that work from a fresh clone. The original
hardcoded ``/home/sukhomay/Desktop/DBMS_Lab/P/graphs``, so the project only ran
on one machine.
"""

from __future__ import annotations

import os
import secrets

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def _as_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class Config:
    #: Where built graphs are stored. One subdirectory per graph.
    DATA_DIR = os.environ.get("GRAPHNEXUS_DATA_DIR", os.path.join(BASE_DIR, "data"))

    #: Generated per-process when unset, which is fine for local development
    #: and forces an explicit value in any real deployment.
    SECRET_KEY = os.environ.get("GRAPHNEXUS_SECRET_KEY") or secrets.token_hex(32)

    #: Buffer pool size in 512-byte blocks. 256 blocks is 128 KB.
    BUFFER_POOL_BLOCKS = _as_int("GRAPHNEXUS_BUFFER_BLOCKS", 256)

    #: Reject uploads larger than this. Default 64 MB.
    MAX_CONTENT_LENGTH = _as_int("GRAPHNEXUS_MAX_UPLOAD_MB", 64) * 1024 * 1024

    #: Extensions accepted by the upload form.
    ALLOWED_EXTENSIONS = {".txt", ".csv", ".tsv", ".edges"}

    WTF_CSRF_ENABLED = True


class TestConfig(Config):
    SECRET_KEY = "testing-only"
    WTF_CSRF_ENABLED = False
    TESTING = True
