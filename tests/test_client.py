"""Tests for the DOI / download-URL resolution logic in ``radar_jupyter.client``.

Fast unit tests mock the network. The live integration tests (marked
``network``) exercise the real DOI resolution against known datasets and are
skipped with ``pytest -m "not network"``.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from radar_jupyter.client import (
    is_doi,
    is_verifiable_download_host,
    resolve_doi_download_url,
)


# --------------------------------------------------------------------------- #
# is_doi
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "identifier",
    [
        "10.35097/prcfrppz0c1f1e28",
        "doi:10.35097/prcfrppz0c1f1e28",
        "https://doi.org/10.35097/prcfrppz0c1f1e28",
        "https://dx.doi.org/10.35097/prcfrppz0c1f1e28",
        "  10.35097/prcfrppz0c1f1e28  ",  # surrounding whitespace
    ],
)
def test_is_doi_accepts_dois(identifier):
    assert is_doi(identifier) is True


@pytest.mark.parametrize(
    "identifier",
    [
        "prcfrppz0c1f1e28",  # plain dataset id
        "RADAR/prcfrppz0c1f1e28",  # RADAR id
        "peskm3ukk7x3vc6x",
        "",
    ],
)
def test_is_doi_rejects_non_dois(identifier):
    assert is_doi(identifier) is False


# --------------------------------------------------------------------------- #
# is_verifiable_download_host
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "url",
    [
        "https://radar-service.eu/x.tar",
        "https://www.radar-service.eu/radar/x.tar",
        "https://radar4chem.radar-service.eu/radar-backend/archives/x/versions/1/content",
        "https://radar.kit.edu/x.tar",
        "https://RADAR.KIT.EDU/x.tar",  # case-insensitive host
    ],
)
def test_verifiable_hosts(url):
    assert is_verifiable_download_host(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://datathek.oeaw.ac.at/radar-backend/archives/x/versions/1/content",
        "https://example.com/x.tar",
        "https://radar-service.eu.evil.com/x.tar",  # spoofed suffix
        "https://notradar.kit.edu.evil.com/x.tar",
        "not-a-url",
    ],
)
def test_non_verifiable_hosts(url):
    assert is_verifiable_download_host(url) is False


# --------------------------------------------------------------------------- #
# resolve_doi_download_url (mocked network)
# --------------------------------------------------------------------------- #

def _fake_head(links):
    """Return a stand-in for ``requests.head`` yielding the given ``.links``."""
    def _head(url, allow_redirects, timeout):
        return SimpleNamespace(links=links, raise_for_status=lambda: None)

    return _head


def test_resolve_returns_item_link():
    item_url = "https://radar.kit.edu/radar-backend/archives/abc/versions/1/content"
    links = {"item": {"url": item_url, "rel": "item"}}
    with patch("radar_jupyter.client.requests.head", _fake_head(links)):
        assert resolve_doi_download_url("10.1234/abc") == item_url


def test_resolve_raises_when_no_item_link():
    links = {"describedby": {"url": "https://example.com/meta", "rel": "describedby"}}
    with patch("radar_jupyter.client.requests.head", _fake_head(links)):
        with pytest.raises(ValueError, match="cannot be used with this script"):
            resolve_doi_download_url("10.1234/abc")


def test_resolve_raises_when_no_links_at_all():
    with patch("radar_jupyter.client.requests.head", _fake_head({})):
        with pytest.raises(ValueError, match='rel="item"'):
            resolve_doi_download_url("10.1234/abc")


def test_resolve_raises_for_non_doi():
    with pytest.raises(ValueError, match="Not a DOI"):
        resolve_doi_download_url("prcfrppz0c1f1e28")


# --------------------------------------------------------------------------- #
# Live integration tests against real DOIs
# --------------------------------------------------------------------------- #

@pytest.mark.network
def test_live_radar_service_doi_is_downloadable_and_verifiable():
    """radar-service.eu DOI: resolves to a download URL that can be verified."""
    url = resolve_doi_download_url("10.22000/z4cf0vf3dzskxcx7")
    assert url.startswith("http")
    assert is_verifiable_download_host(url) is True


@pytest.mark.network
def test_live_radar_kit_doi_is_downloadable_and_verifiable():
    """radar.kit.edu DOI: resolves to a download URL that can be verified."""
    url = resolve_doi_download_url("10.35097/uya6rt5t4k3exf3g")
    assert url.startswith("http")
    assert is_verifiable_download_host(url) is True


@pytest.mark.network
def test_live_doi_downloadable_but_not_verifiable():
    """A DOI that exposes an item link on a host we cannot checksum-verify."""
    url = resolve_doi_download_url("10.60887/5d78hjkq42jpnbmz")
    assert url.startswith("http")
    assert is_verifiable_download_host(url) is False


@pytest.mark.network
def test_live_doi_neither_downloadable_nor_verifiable():
    """A DOI with no rel=\"item\" link cannot be used with this script."""
    with pytest.raises(ValueError, match="cannot be used with this script"):
        resolve_doi_download_url("10.1007/s10676-024-09775-5")
