"""Tests for the resumable-download behavior of ``radar_jupyter.download._download_file``.

The archive is streamed into a temporary ``<name>.part`` file which is renamed
to its final name only once the download completes. An interrupted download
leaves the ``.part`` file in place so a later call can resume it via an HTTP
``Range`` request (or restart from scratch if the server ignores the range).

All network access is mocked; these are fast unit tests.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from radar_jupyter.download import _download_file

RADAR_ID = "abc123"
FULL_BODY = b"0123456789ABCDEFGHIJ"  # 20 bytes


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #

class FakeResponse:
    """Minimal stand-in for a streamed ``requests`` response used as a context manager."""

    def __init__(self, status_code=200, headers=None, chunks=(), raise_after=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = list(chunks)
        # If set, ``iter_content`` yields this many chunks then raises, simulating
        # a dropped connection / aborted download.
        self._raise_after = raise_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def iter_content(self, chunk_size):
        for i, chunk in enumerate(self._chunks):
            if self._raise_after is not None and i == self._raise_after:
                raise requests.ConnectionError("connection dropped")
            yield chunk


def _make_get(responses, recorder):
    """Build a ``requests.get`` replacement that returns queued responses in order.

    ``responses`` may be a single response (reused for every call) or a list
    (one per call). The ``Range`` header of each call is appended to
    ``recorder["ranges"]``.
    """
    queue = responses if isinstance(responses, list) else None

    def _get(url, stream, timeout, headers):
        recorder["ranges"].append(headers.get("Range"))
        return queue.pop(0) if queue is not None else responses

    return _get


@pytest.fixture
def download_dir(tmp_path):
    """Point ``DOWNLOAD_DIR`` at a temp directory for the duration of a test."""
    with patch("radar_jupyter.download.DOWNLOAD_DIR", str(tmp_path)):
        yield tmp_path


def _run(responses):
    """Invoke ``_download_file`` with mocked network, returning (path, recorder)."""
    recorder = {"ranges": []}
    client = SimpleNamespace()
    with patch("radar_jupyter.download.requests.get", _make_get(responses, recorder)):
        path = _download_file(
            RADAR_ID, client, chunk_size=4, download_url="http://example.test/x"
        )
    return path, recorder


# --------------------------------------------------------------------------- #
# Fresh download
# --------------------------------------------------------------------------- #

def test_fresh_download_writes_final_file_and_no_range(download_dir):
    chunks = [FULL_BODY[i:i + 4] for i in range(0, len(FULL_BODY), 4)]
    response = FakeResponse(
        status_code=200,
        headers={"content-length": str(len(FULL_BODY))},
        chunks=chunks,
    )

    path, recorder = _run(response)

    assert path == download_dir / f"{RADAR_ID}.tar"
    assert path.read_bytes() == FULL_BODY
    # No Range header on a first download, and no leftover .part file.
    assert recorder["ranges"] == [None]
    assert not (download_dir / f"{RADAR_ID}.tar.part").exists()


# --------------------------------------------------------------------------- #
# Interruption leaves a resumable .part file
# --------------------------------------------------------------------------- #

def test_interrupted_download_leaves_partial_file(download_dir):
    chunks = [FULL_BODY[i:i + 4] for i in range(0, len(FULL_BODY), 4)]
    # Yield two chunks (8 bytes) then drop the connection.
    response = FakeResponse(
        status_code=200,
        headers={"content-length": str(len(FULL_BODY))},
        chunks=chunks,
        raise_after=2,
    )

    with pytest.raises(requests.ConnectionError):
        _run(response)

    partial = download_dir / f"{RADAR_ID}.tar.part"
    assert partial.exists()
    assert partial.read_bytes() == FULL_BODY[:8]
    # The final file must not exist until the download completes.
    assert not (download_dir / f"{RADAR_ID}.tar").exists()


# --------------------------------------------------------------------------- #
# Resume via HTTP 206 Partial Content
# --------------------------------------------------------------------------- #

def test_resume_appends_to_partial_file(download_dir):
    partial = download_dir / f"{RADAR_ID}.tar.part"
    partial.write_bytes(FULL_BODY[:8])  # 8 bytes already downloaded

    remaining = FULL_BODY[8:]
    chunks = [remaining[i:i + 4] for i in range(0, len(remaining), 4)]
    response = FakeResponse(
        status_code=206,
        headers={
            "content-length": str(len(remaining)),
            "content-range": f"bytes 8-{len(FULL_BODY) - 1}/{len(FULL_BODY)}",
        },
        chunks=chunks,
    )

    path, recorder = _run(response)

    assert recorder["ranges"] == ["bytes=8-"]
    assert path == download_dir / f"{RADAR_ID}.tar"
    assert path.read_bytes() == FULL_BODY  # appended, not overwritten
    assert not partial.exists()


# --------------------------------------------------------------------------- #
# Server ignores the Range header (responds 200) -> restart from scratch
# --------------------------------------------------------------------------- #

def test_server_ignores_range_restarts_download(download_dir):
    partial = download_dir / f"{RADAR_ID}.tar.part"
    partial.write_bytes(b"STALEDAT")  # 8 bytes of stale data

    chunks = [FULL_BODY[i:i + 4] for i in range(0, len(FULL_BODY), 4)]
    response = FakeResponse(
        status_code=200,  # server ignored the Range request
        headers={"content-length": str(len(FULL_BODY))},
        chunks=chunks,
    )

    path, recorder = _run(response)

    # A Range header was still sent, but the response was a full 200.
    assert recorder["ranges"] == ["bytes=8-"]
    # The stale partial must be overwritten, not appended to.
    assert path.read_bytes() == FULL_BODY
    assert not partial.exists()


# --------------------------------------------------------------------------- #
# Partial file already complete -> server responds 416, file is finalized
# --------------------------------------------------------------------------- #

def test_range_not_satisfiable_finalizes_complete_partial(download_dir):
    partial = download_dir / f"{RADAR_ID}.tar.part"
    partial.write_bytes(FULL_BODY)  # already complete

    response = FakeResponse(status_code=416, headers={}, chunks=[])

    path, recorder = _run(response)

    assert recorder["ranges"] == [f"bytes={len(FULL_BODY)}-"]
    assert path == download_dir / f"{RADAR_ID}.tar"
    assert path.read_bytes() == FULL_BODY
    assert not partial.exists()


# --------------------------------------------------------------------------- #
# End-to-end: interrupt, then resume on the next call
# --------------------------------------------------------------------------- #

def test_interrupt_then_resume_end_to_end(download_dir):
    chunks = [FULL_BODY[i:i + 4] for i in range(0, len(FULL_BODY), 4)]

    # First call: yields two chunks (8 bytes) then drops.
    first = FakeResponse(
        status_code=200,
        headers={"content-length": str(len(FULL_BODY))},
        chunks=chunks,
        raise_after=2,
    )
    # Second call: honours the Range request and serves the remaining bytes.
    remaining = FULL_BODY[8:]
    second = FakeResponse(
        status_code=206,
        headers={
            "content-length": str(len(remaining)),
            "content-range": f"bytes 8-{len(FULL_BODY) - 1}/{len(FULL_BODY)}",
        },
        chunks=[remaining[i:i + 4] for i in range(0, len(remaining), 4)],
    )

    recorder = {"ranges": []}
    client = SimpleNamespace()
    with patch(
        "radar_jupyter.download.requests.get", _make_get([first, second], recorder)
    ):
        with pytest.raises(requests.ConnectionError):
            _download_file(RADAR_ID, client, chunk_size=4, download_url="http://x.test/x")

        path = _download_file(
            RADAR_ID, client, chunk_size=4, download_url="http://x.test/x"
        )

    assert recorder["ranges"] == [None, "bytes=8-"]
    assert path == download_dir / f"{RADAR_ID}.tar"
    assert path.read_bytes() == FULL_BODY
    assert not (download_dir / f"{RADAR_ID}.tar.part").exists()


# --------------------------------------------------------------------------- #
# A completed download is reused; a lone .part file is not mistaken for one
# --------------------------------------------------------------------------- #

def test_completed_file_is_reused_without_downloading(download_dir):
    final = download_dir / f"{RADAR_ID}.tar"
    final.write_bytes(FULL_BODY)

    recorder = {"ranges": []}
    client = SimpleNamespace()
    with patch("radar_jupyter.download.requests.get", _make_get([], recorder)):
        path = _download_file(RADAR_ID, client, download_url="http://x.test/x")

    # No network call was made at all.
    assert recorder["ranges"] == []
    assert path == final
