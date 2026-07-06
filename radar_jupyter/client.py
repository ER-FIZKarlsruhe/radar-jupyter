import re
from enum import Enum
from os import getenv
from urllib.parse import unquote, urlparse

import requests


class RadarMetadataType(Enum):
    JSON = "Json"
    JSON_LD = "Jsonld"
    RADAR = "Radar"


DOWNLOAD_DIR = getenv("RADAR_DOWNLOAD_DIR", "downloads")
EXTRACTION_DIR = getenv("RADAR_EXTRACTION_DIR", "extracted")
RADAR_API_URL = getenv("RADAR_API_URL", "https://www.radar-service.eu/radar")
DOI_RESOLVER_URL = getenv("DOI_RESOLVER_URL", "https://doi.org")
RADAR_SPARQL_URL = getenv("RADAR_SPARQL_URL", "https://radar-service.eu/sparql")

# Hosts whose downloads can be checksum-verified via the RADAR metadata API.
_VERIFIABLE_DOWNLOAD_HOSTS = ("radar-service.eu", "radar.kit.edu")


_RADAR_ID_PATTERN = re.compile(r"^RADAR/(.+)$", re.IGNORECASE)
_DOI_PATTERN = re.compile(
    r"^(?:https?://(?:dx\.)?doi\.org/|doi:)?(10\.\d{4,}/(.+))$",
    re.IGNORECASE,
)


def resolve_dataset_id(identifier: str) -> str:
    """
    Extract the plain RADAR dataset ID from any supported identifier format.

    Accepted formats:

    - Plain dataset ID: ``d1a2b3c4-...``
    - RADAR ID: ``RADAR/d1a2b3c4-...``
    - Bare DOI: ``10.2222/d1a2b3c4-...``
    - DOI URL: ``https://doi.org/10.2222/d1a2b3c4-...``

    :param identifier: A dataset identifier in any of the formats above.
    :return: The plain dataset ID.
    :raises ValueError: If the identifier is empty.

    Example::

        resolve_dataset_id("https://doi.org/10.2222/abc123")  # → "abc123"
        resolve_dataset_id("RADAR/abc123")                    # → "abc123"
        resolve_dataset_id("abc123")                          # → "abc123"
    """
    identifier = identifier.strip()
    if not identifier:
        raise ValueError("Identifier must not be empty.")

    m = _RADAR_ID_PATTERN.match(identifier)
    if m:
        return m.group(1)

    m = _DOI_PATTERN.match(identifier)
    if m:
        return m.group(2)

    return identifier


def is_doi(identifier: str) -> bool:
    """
    Return ``True`` if the identifier is a DOI or DOI URL.

    :param identifier: An identifier in any supported format.
    :return: Whether the identifier is a DOI (bare, ``doi:`` prefixed, or a URL).
    """
    return bool(_DOI_PATTERN.match(identifier.strip()))


def resolve_doi_download_url(identifier: str, timeout: float = 30) -> str:
    """
    Resolve the dataset TAR download URL for a DOI via its HTTP ``Link`` headers.

    Sends an HTTP HEAD request to the DOI (following redirects) and looks for a
    ``Link`` header with ``rel="item"``. The target URL of that link is the
    location of the downloadable TAR archive.

    :param identifier: A DOI or DOI URL.
    :param timeout: Request timeout in seconds.
    :return: The URL of the ``rel="item"`` link.
    :raises ValueError: If the identifier is not a DOI, or if the DOI exposes no
        ``Link`` header with ``rel="item"`` (i.e. it cannot be used with this script).
    """
    m = _DOI_PATTERN.match(identifier.strip())
    if not m:
        raise ValueError(f"Not a DOI: {identifier!r}")

    doi_url = f"{DOI_RESOLVER_URL.rstrip('/')}/{m.group(1)}"
    response = requests.head(doi_url, allow_redirects=True, timeout=timeout)
    response.raise_for_status()

    item = response.links.get("item")
    if not item or not item.get("url"):
        raise ValueError(
            f"DOI {identifier!r} cannot be used with this script: "
            'no Link header with rel="item" was found.'
        )
    return item["url"]


def is_verifiable_download_host(url: str) -> bool:
    """
    Return ``True`` if a download URL points at a host whose files can be
    checksum-verified via the RADAR metadata API.

    :param url: The download URL to inspect.
    :return: Whether the URL's host is (a subdomain of) a known RADAR host.
    """
    host = (urlparse(url).hostname or "").lower()
    return any(
        host == known or host.endswith(f".{known}")
        for known in _VERIFIABLE_DOWNLOAD_HOSTS
    )


def _get_metadata_export_url(
    radar_id: str, metadata_type: RadarMetadataType = RadarMetadataType.JSON
) -> str:
    return f"{RADAR_API_URL}/en/export/{radar_id}/export{metadata_type.value}"


class RadarApiClient:
    """HTTP client for the RADAR REST API."""

    def __init__(self, base_url: str = RADAR_API_URL, timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def search_datasets(
        self,
        query: str | None = None,
        sort: str | None = None,
        offset: int | None = None,
        rows: int | None = None,
        **extra_params,
    ) -> dict:
        """
        Search for datasets in RADAR.

        :param query: Free-text search query.
        :param sort: Sort field.
        :param offset: Pagination offset.
        :param rows: Number of results to return.
        :return: Search result dict from the RADAR API.
        """
        params = {
            k: v
            for k, v in {
                "query": query,
                "sort": sort,
                "offset": offset,
                "rows": rows,
                **extra_params,
            }.items()
            if v is not None
        }
        resp = self._session.get(
            f"{self.base_url}/api/datasets", params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def get_dataset_metadata(
        self,
        dataset_id: str,
        metadata_type: RadarMetadataType = RadarMetadataType.JSON,
    ) -> dict:
        """
        Fetch dataset metadata via the RADAR export endpoint.

        Use this when you need a specific metadata schema (JSON-LD, RADAR XML).
        For plain JSON, ``get_dataset()`` is preferred as it uses the REST API directly.

        :param dataset_id: The RADAR dataset ID.
        :param metadata_type: The metadata schema to return.
        :return: Metadata dict.
        """
        resp = self._session.get(
            _get_metadata_export_url(dataset_id, metadata_type),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_download_url(self, dataset_id: str) -> str:
        """
        Resolve the download URL for a dataset TAR archive.

        Follows the API redirect and returns the target URL without downloading
        the file itself.

        :param dataset_id: The RADAR dataset ID.
        :return: Direct download URL for the dataset TAR archive.
        """
        resp = self._session.get(
            f"{self.base_url}/api/datasets/{dataset_id}/download",
            allow_redirects=False,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.headers.get("Location", resp.url)

    def get_statistics(self, dataset_id: str) -> dict:
        """
        Fetch download and access statistics for a dataset.

        :param dataset_id: The RADAR dataset ID.
        :return: Statistics dict from the RADAR API.
        """
        resp = self._session.get(
            f"{self.base_url}/api/datasets/{dataset_id}/statistics",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def fetch_facets(
        self,
        facet_name: str | None = None,
        search_query: str | None = None,
        rows: int | None = None,
        **extra_params,
    ) -> dict:
        """
        Fetch facet values for filtering dataset searches.

        :param facet_name: Name of the facet to retrieve.
        :param search_query: Optional search query to scope the facet values.
        :param rows: Maximum number of facet values to return.
        :return: Facet result dict from the RADAR API.
        """
        params = {
            k: v
            for k, v in {
                "facetName": facet_name,
                "searchQuery": search_query,
                "rows": rows,
                **extra_params,
            }.items()
            if v is not None
        }
        resp = self._session.get(
            f"{self.base_url}/api/datasets/facets",
            params=params,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()


def _get_filename_from_content_disposition_header(
    value: str | None, default: str
) -> str:
    """
    Tries to extract the filename given in a HTTP Content-Disposition header.

    :param value: The value of the Content-Disposition header.
    :param default: Fallback filename when no filename can be extracted.
    :return: The filename.
    """
    if value is None:
        return default

    pattern1 = r'attachment;\s*filename="(.+)"'
    pattern2 = r"attachment;\s*filename\*=.+''(.+)"

    attempt1 = re.match(pattern1, value)
    if attempt1:
        return attempt1.group(1)

    attempt2 = re.match(pattern2, value)
    if attempt2:
        return unquote(attempt2.group(1))

    return default
