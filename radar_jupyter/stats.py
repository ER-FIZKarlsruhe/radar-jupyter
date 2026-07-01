from .client import RadarApiClient


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
