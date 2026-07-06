"""Query the RADAR Knowledge Graph (KG) via its SPARQL endpoint.

The KG describes datasets using the schema.org vocabulary (``http://schema.org/``).
A dataset node is any resource of type ``schema:Dataset``. Authors are modelled as
``schema:creator`` (``schema:Person``) and institutions as ``schema:publisher``
(``schema:Organization``); both carry their label in ``schema:name``.
"""

from dataclasses import dataclass

from SPARQLWrapper import JSON, SPARQLWrapper

from .client import RADAR_SPARQL_URL

# The KG exclusively uses the (http, not https) schema.org namespace.
SCHEMA_ORG = "http://schema.org/"


@dataclass(frozen=True)
class Dataset:
    """A dataset node from the RADAR Knowledge Graph.

    :ivar id: The node's identifier — the full RDF URI
        (e.g. ``https://radar.kit.edu/id/RRseLXdaBQBUPykI``).
    :ivar name: The ``schema:name`` of the dataset.
    :ivar date_published: The ``schema:datePublished`` value (``YYYY-MM-DD``).
    :ivar radar_id: The plain RADAR dataset id — the local part of the node URI
        (everything after the final ``/``, e.g. ``RRseLXdaBQBUPykI``). This is
        the id accepted by ``download_and_extract()`` and the RADAR REST API.
        Derived automatically from :attr:`id` when omitted.
    """

    id: str
    name: str
    date_published: str
    radar_id: str = ""

    def __post_init__(self) -> None:
        if not self.radar_id:
            # The node URI is ``<host>/id/<radar_id>``; the id is its last segment.
            object.__setattr__(self, "radar_id", self.id.rsplit("/", 1)[-1])


def _sparql_string_literal(value: str) -> str:
    """
    Turn a Python string into a safely-quoted SPARQL string literal.

    Escapes the characters that would otherwise break out of / inject into a
    double-quoted SPARQL literal.

    :param value: The raw string value.
    :return: The value as a quoted, escaped SPARQL string literal.
    """
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _run_sparql_select(
    query: str, endpoint: str = RADAR_SPARQL_URL, timeout: float = 30
) -> list[dict]:
    """
    Run a SPARQL SELECT query and return its result bindings.

    :param query: A SPARQL SELECT query.
    :param endpoint: The SPARQL endpoint URL.
    :param timeout: Request timeout in seconds.
    :return: The list of result bindings (``results.bindings`` from the SPARQL
        JSON results), each a dict mapping variable name to a value object.
    """
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(int(timeout))
    results = sparql.queryAndConvert()
    return results["results"]["bindings"]


def _select_datasets(
    graph_pattern: str, endpoint: str = RADAR_SPARQL_URL, timeout: float = 30
) -> list[Dataset]:
    """
    Select dataset nodes matching an extra graph pattern.

    Every query selects the dataset's URI, name, and publication date; callers
    supply an additional ``graph_pattern`` (extra triples and/or a ``FILTER``)
    to constrain the result set.

    :param graph_pattern: SPARQL to inject into the ``WHERE`` block, in addition
        to the base ``?id a schema:Dataset ; schema:name ?name ; schema:datePublished ?datePublished``.
    :param endpoint: The SPARQL endpoint URL.
    :param timeout: Request timeout in seconds.
    :return: The matching datasets, ordered by publication date.
    """
    query = f"""PREFIX schema: <{SCHEMA_ORG}>
SELECT DISTINCT ?id ?name ?datePublished WHERE {{
  ?id a schema:Dataset ;
      schema:name ?name ;
      schema:datePublished ?datePublished .
{graph_pattern}
}}
ORDER BY ?datePublished"""

    bindings = _run_sparql_select(query, endpoint, timeout)
    return [
        Dataset(
            id=b["id"]["value"],
            name=b["name"]["value"],
            date_published=b["datePublished"]["value"],
        )
        for b in bindings
    ]


def list_datasets_by_year(
    year: int, endpoint: str = RADAR_SPARQL_URL, timeout: float = 30
) -> list[Dataset]:
    """
    List all datasets published in a given year.

    :param year: The four-digit publication year to filter on (e.g. ``2024``).
    :param endpoint: The SPARQL endpoint URL. Defaults to the RADAR KG endpoint.
    :param timeout: Request timeout in seconds.
    :return: A list of :class:`Dataset` items published in ``year``.
    :raises ValueError: If ``year`` is not a four-digit year.

    Example::

        from radar_jupyter.knowledge_graph import list_datasets_by_year
        for ds in list_datasets_by_year(2024):
            print(ds.date_published, ds.name, ds.id)
    """
    if not isinstance(year, int) or not (1000 <= year <= 9999):
        raise ValueError(f"year must be a four-digit year, got {year!r}.")

    # datePublished is stored as a plain string literal (e.g. "2024-01-02"),
    # so filter on the string prefix rather than the YEAR() date function.
    pattern = f'  FILTER(STRSTARTS(STR(?datePublished), "{year}"))'
    return _select_datasets(pattern, endpoint, timeout)


def list_datasets_by_author(
    author: str, endpoint: str = RADAR_SPARQL_URL, timeout: float = 30
) -> list[Dataset]:
    """
    List all datasets whose author (``schema:creator``) matches ``author``.

    Matching is case-insensitive and substring-based, so ``"kamp"`` matches
    ``"van de Kamp, Thomas"``. Author names in the KG are formatted
    ``"Last, First"``.

    :param author: Author name (or fragment) to match.
    :param endpoint: The SPARQL endpoint URL. Defaults to the RADAR KG endpoint.
    :param timeout: Request timeout in seconds.
    :return: A list of :class:`Dataset` items created by a matching author.
    :raises ValueError: If ``author`` is empty.

    Example::

        from radar_jupyter.knowledge_graph import list_datasets_by_author
        datasets = list_datasets_by_author("van de Kamp")
    """
    author = author.strip()
    if not author:
        raise ValueError("author must not be empty.")

    literal = _sparql_string_literal(author)
    pattern = (
        "  ?id schema:creator ?creator .\n"
        "  ?creator schema:name ?creatorName .\n"
        f"  FILTER(CONTAINS(LCASE(?creatorName), LCASE({literal})))"
    )
    return _select_datasets(pattern, endpoint, timeout)


def list_datasets_by_institution(
    institution: str, endpoint: str = RADAR_SPARQL_URL, timeout: float = 30
) -> list[Dataset]:
    """
    List all datasets published by an institution (``schema:publisher``).

    Matching is case-insensitive and substring-based, so ``"karlsruhe"``
    matches ``"Karlsruhe Institute of Technology(KIT)"``.

    :param institution: Institution name (or fragment) to match.
    :param endpoint: The SPARQL endpoint URL. Defaults to the RADAR KG endpoint.
    :param timeout: Request timeout in seconds.
    :return: A list of :class:`Dataset` items published by a matching institution.
    :raises ValueError: If ``institution`` is empty.

    Example::

        from radar_jupyter.knowledge_graph import list_datasets_by_institution
        datasets = list_datasets_by_institution("Karlsruhe Institute of Technology")
    """
    institution = institution.strip()
    if not institution:
        raise ValueError("institution must not be empty.")

    literal = _sparql_string_literal(institution)
    pattern = (
        "  ?id schema:publisher ?publisher .\n"
        "  ?publisher schema:name ?publisherName .\n"
        f"  FILTER(CONTAINS(LCASE(?publisherName), LCASE({literal})))"
    )
    return _select_datasets(pattern, endpoint, timeout)
