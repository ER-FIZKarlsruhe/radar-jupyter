"""Tests for the RADAR Knowledge Graph SPARQL queries.

Unit tests mock the SPARQL layer; the live integration tests (marked
``network``) query the real endpoint and are skipped with
``pytest -m "not network"``.
"""

from unittest.mock import patch

import pytest

from radar_jupyter.knowledge_graph import (
    Dataset,
    _sparql_string_literal,
    list_datasets_by_author,
    list_datasets_by_institution,
    list_datasets_by_year,
)


def _binding(uri, name, date):
    return {
        "id": {"type": "uri", "value": uri},
        "name": {"type": "literal", "value": name},
        "datePublished": {"type": "literal", "value": date},
    }


# --------------------------------------------------------------------------- #
# list_datasets_by_year (mocked SPARQL)
# --------------------------------------------------------------------------- #

def test_maps_bindings_to_datasets():
    fake = [
        _binding("https://radar.kit.edu/id/aaa", "First", "2024-01-02"),
        _binding("https://radar4chem.radar-service.eu/id/bbb", "Second", "2024-03-10"),
    ]
    with patch(
        "radar_jupyter.knowledge_graph._run_sparql_select", return_value=fake
    ) as m:
        result = list_datasets_by_year(2024)

    assert result == [
        Dataset("https://radar.kit.edu/id/aaa", "First", "2024-01-02"),
        Dataset("https://radar4chem.radar-service.eu/id/bbb", "Second", "2024-03-10"),
    ]
    # id is the full node URI
    assert result[0].id.startswith("https://")

    # The generated query filters on the requested year and the schema.org type.
    query = m.call_args.args[0]
    assert "schema:Dataset" in query
    assert "schema:datePublished" in query
    assert '"2024"' in query


def test_empty_result_returns_empty_list():
    with patch("radar_jupyter.knowledge_graph._run_sparql_select", return_value=[]):
        assert list_datasets_by_year(1999) == []


@pytest.mark.parametrize("bad_year", [999, 10000, "2024", 20.24, None])
def test_rejects_non_four_digit_year(bad_year):
    with pytest.raises(ValueError, match="four-digit year"):
        list_datasets_by_year(bad_year)


# --------------------------------------------------------------------------- #
# list_datasets_by_author / list_datasets_by_institution (mocked SPARQL)
# --------------------------------------------------------------------------- #

def test_by_author_builds_creator_filter():
    fake = [_binding("https://radar.kit.edu/id/aaa", "Some data", "2024-10-22")]
    with patch(
        "radar_jupyter.knowledge_graph._run_sparql_select", return_value=fake
    ) as m:
        result = list_datasets_by_author("van de Kamp")

    assert result == [Dataset("https://radar.kit.edu/id/aaa", "Some data", "2024-10-22")]
    query = m.call_args.args[0]
    assert "schema:creator" in query
    assert "?creatorName" in query
    # case-insensitive substring match on the requested author
    assert 'CONTAINS(LCASE(?creatorName), LCASE("van de Kamp"))' in query


def test_by_institution_builds_publisher_filter():
    fake = [_binding("https://radar.kit.edu/id/bbb", "Other data", "2024-01-04")]
    with patch(
        "radar_jupyter.knowledge_graph._run_sparql_select", return_value=fake
    ) as m:
        result = list_datasets_by_institution("Karlsruhe Institute of Technology")

    assert result == [Dataset("https://radar.kit.edu/id/bbb", "Other data", "2024-01-04")]
    query = m.call_args.args[0]
    assert "schema:publisher" in query
    assert "?publisherName" in query
    assert 'LCASE("Karlsruhe Institute of Technology")' in query


@pytest.mark.parametrize(
    "func", [list_datasets_by_author, list_datasets_by_institution]
)
@pytest.mark.parametrize("bad", ["", "   "])
def test_by_name_rejects_empty(func, bad):
    with pytest.raises(ValueError, match="must not be empty"):
        func(bad)


def test_sparql_string_literal_escapes_injection():
    # A value trying to break out of the literal must be neutralised.
    assert _sparql_string_literal('a"b') == '"a\\"b"'
    assert _sparql_string_literal("a\\b") == '"a\\\\b"'
    # The escaped literal must not contain an unescaped closing quote mid-string.
    dangerous = 'x") } INJECT {'
    literal = _sparql_string_literal(dangerous)
    assert literal.startswith('"') and literal.endswith('"')
    assert '\\"' in literal


# --------------------------------------------------------------------------- #
# Live integration test against the real KG endpoint
# --------------------------------------------------------------------------- #

@pytest.mark.network
def test_live_list_datasets_by_year():
    datasets = list_datasets_by_year(2024)
    assert len(datasets) > 0
    for ds in datasets:
        assert ds.id.startswith("http")
        assert ds.name
        assert ds.date_published.startswith("2024")


@pytest.mark.network
def test_live_list_datasets_by_author():
    datasets = list_datasets_by_author("van de Kamp")
    assert len(datasets) > 0
    for ds in datasets:
        assert ds.id.startswith("http")
        assert ds.name


@pytest.mark.network
def test_live_list_datasets_by_institution():
    datasets = list_datasets_by_institution("Karlsruhe Institute of Technology")
    assert len(datasets) > 0
    for ds in datasets:
        assert ds.id.startswith("http")
        assert ds.name
