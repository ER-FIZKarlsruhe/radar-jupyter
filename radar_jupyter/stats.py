import sys
from dataclasses import dataclass
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import requests

from .client import RadarApiClient
from .knowledge_graph import Dataset, list_datasets_by_year


def search_datasets(
    query: str | None = None,
    sort: str | None = None,
    offset: int | None = None,
    rows: int | None = None,
    client: RadarApiClient | None = None,
    **extra_params,
) -> dict:
    """
    Search for datasets in RADAR.

    :param query: Free-text search query.
    :param sort: Sort field.
    :param offset: Pagination offset.
    :param rows: Number of results to return.
    :param client: Optional ``RadarApiClient`` instance. A default client is created if omitted.
    :return: Search result dict from the RADAR API.

    Example::

        from radar_jupyter.stats import search_datasets
        results = search_datasets(query="climate", rows=10)
    """
    if client is None:
        client = RadarApiClient()
    return client.search_datasets(query, sort, offset, rows, **extra_params)


def fetch_facets(
    facet_name: str | None = None,
    search_query: str | None = None,
    rows: int | None = None,
    client: RadarApiClient | None = None,
    **extra_params,
) -> dict:
    """
    Fetch facet values for filtering dataset searches.

    :param facet_name: Name of the facet to retrieve.
    :param search_query: Optional search query to scope the facet values.
    :param rows: Maximum number of facet values to return.
    :param client: Optional ``RadarApiClient`` instance. A default client is created if omitted.
    :return: Facet result dict from the RADAR API.

    Example::

        from radar_jupyter.stats import fetch_facets
        subjects = fetch_facets(facet_name="subject")
    """
    if client is None:
        client = RadarApiClient()
    return client.fetch_facets(facet_name, search_query, rows, **extra_params)


@dataclass(frozen=True)
class YearAccessStats:
    """Aggregate access/download statistics for all datasets of one year.

    :ivar year: The publication year the figures were aggregated for.
    :ivar total_access: Sum of ``totalAccess`` over all counted datasets.
    :ivar total_downloads: Sum of ``totalDownloads`` over all counted datasets.
    :ivar dataset_count: Number of datasets published in ``year`` (from the KG).
    :ivar counted_datasets: Number of those datasets whose statistics could be
        retrieved. May be lower than ``dataset_count`` when a dataset's host
        does not expose statistics (e.g. remote datasets).
    """

    year: int
    total_access: int
    total_downloads: int
    dataset_count: int
    counted_datasets: int


def _statistics_base_url(dataset: Dataset) -> str:
    """
    Build the RADAR API base URL for the instance that hosts ``dataset``.

    Statistics are only served by the RADAR instance a dataset actually lives
    on (e.g. ``radar.kit.edu``, ``radar4chem.radar-service.eu``), which is
    encoded in the KG node URI — not by the central cloud search index.

    :param dataset: A dataset whose ``id`` is a full node URI.
    :return: The ``<scheme>://<host>/radar`` API base URL for that host.
    """
    parsed = urlparse(dataset.id)
    return f"{parsed.scheme}://{parsed.netloc}/radar"


def aggregate_access_download_stats(
    year: int, timeout: float = 30, progress: bool = True
) -> YearAccessStats:
    """
    Sum access and download counts over every dataset published in ``year``.

    Datasets are discovered via :func:`list_datasets_by_year`; each dataset's
    ``totalAccess``/``totalDownloads`` are fetched from the RADAR instance that
    hosts it. Datasets whose host does not return statistics are skipped.

    :param year: The four-digit publication year (e.g. ``2024``).
    :param timeout: Per-request timeout in seconds.
    :param progress: Whether to print a progress line to stdout while gathering.
    :return: The aggregated :class:`YearAccessStats`.
    :raises ValueError: If ``year`` is not a four-digit year (via
        :func:`list_datasets_by_year`).

    Example::

        from radar_jupyter.stats import aggregate_access_download_stats
        stats = aggregate_access_download_stats(2024)
        print(stats.total_access, stats.total_downloads)
    """
    datasets = list_datasets_by_year(year, timeout=timeout)

    total_access = 0
    total_downloads = 0
    counted = 0
    clients: dict[str, RadarApiClient] = {}

    for index, dataset in enumerate(datasets, start=1):
        base_url = _statistics_base_url(dataset)
        client = clients.get(base_url)
        if client is None:
            client = clients[base_url] = RadarApiClient(base_url, timeout=timeout)

        try:
            stats = client.get_statistics(dataset.radar_id)
        except requests.RequestException:
            # A dataset's host may not expose statistics (e.g. remote datasets).
            stats = None

        if stats is not None:
            total_access += int(stats.get("totalAccess", 0) or 0)
            total_downloads += int(stats.get("totalDownloads", 0) or 0)
            counted += 1

        if progress:
            sys.stdout.write(
                f"\rGathering statistics {index}/{len(datasets)} "
                f"(access={total_access:,}, downloads={total_downloads:,})"
            )
            sys.stdout.flush()

    if progress and datasets:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return YearAccessStats(
        year=year,
        total_access=total_access,
        total_downloads=total_downloads,
        dataset_count=len(datasets),
        counted_datasets=counted,
    )


def plot_access_download_ratio(
    year: int, timeout: float = 30, progress: bool = True, ax=None
):
    """
    Plot a pie chart of the access-vs-download ratio for a given year.

    Aggregates ``totalAccess`` and ``totalDownloads`` over all datasets
    published in ``year`` (see :func:`aggregate_access_download_stats`) and
    renders a two-wedge pie chart. Each wedge is annotated with both its
    absolute count and its share of the total; the number of datasets the year
    contains is shown in the title.

    :param year: The four-digit publication year (e.g. ``2024``).
    :param timeout: Per-request timeout in seconds.
    :param progress: Whether to print a progress line while gathering statistics.
    :param ax: An existing Matplotlib ``Axes`` to draw on. A new figure and axes
        are created when omitted.
    :return: The Matplotlib ``Axes`` the pie chart was drawn on.
    :raises ValueError: If no datasets were published in ``year``, or if their
        combined access and download counts are zero (nothing to plot).

    Example::

        from radar_jupyter.stats import plot_access_download_ratio
        plot_access_download_ratio(2024)
    """
    stats = aggregate_access_download_stats(year, timeout=timeout, progress=progress)

    if stats.dataset_count == 0:
        raise ValueError(f"No datasets were published in {year}.")

    total = stats.total_access + stats.total_downloads
    if total == 0:
        raise ValueError(
            f"Datasets published in {year} have no recorded access or downloads."
        )

    if ax is None:
        _, ax = plt.subplots()

    values = [stats.total_access, stats.total_downloads]
    labels = ["Access", "Downloads"]

    def _autopct(pct: float) -> str:
        value = int(round(pct / 100.0 * total))
        return f"{value:,}\n({pct:.1f}%)"

    ax.pie(values, labels=labels, autopct=_autopct, startangle=90)
    ax.axis("equal")  # keep the pie circular

    title = f"RADAR access vs. downloads in {year}\n{stats.dataset_count} datasets"
    if stats.counted_datasets != stats.dataset_count:
        title += f" ({stats.counted_datasets} with statistics)"
    ax.set_title(title)

    return ax
