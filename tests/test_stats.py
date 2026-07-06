"""Tests for the per-year access/download aggregation and pie chart.

Unit tests mock the KG query and the statistics HTTP calls; the live
integration test (marked ``network``) hits the real endpoints and is skipped
with ``pytest -m "not network"``.
"""

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless backend for tests
import matplotlib.pyplot as plt
import pytest
import requests

from radar_jupyter.knowledge_graph import Dataset
from radar_jupyter.stats import (
    YearAccessStats,
    _statistics_base_url,
    aggregate_access_download_stats,
    clear_stats_cache,
    plot_access_download_ratio,
)


def _dataset(uri, date="2024-01-01", name="Some data"):
    return Dataset(id=uri, name=name, date_published=date)


@pytest.fixture(autouse=True)
def _isolate_stats_cache():
    """Keep the module-level statistics cache from leaking between tests."""
    clear_stats_cache()
    yield
    clear_stats_cache()


# --------------------------------------------------------------------------- #
# radar_id derivation on Dataset
# --------------------------------------------------------------------------- #

def test_dataset_derives_radar_id_from_uri():
    ds = _dataset("https://radar.kit.edu/id/lDMMstsZhbBDbtXA")
    assert ds.radar_id == "lDMMstsZhbBDbtXA"


def test_dataset_keeps_explicit_radar_id():
    ds = Dataset("https://x/id/abc", "n", "2024-01-01", radar_id="override")
    assert ds.radar_id == "override"


# --------------------------------------------------------------------------- #
# _statistics_base_url — statistics come from the dataset's own host
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "uri,expected",
    [
        ("https://radar.kit.edu/id/aaa", "https://radar.kit.edu/radar"),
        (
            "https://radar4chem.radar-service.eu/id/bbb",
            "https://radar4chem.radar-service.eu/radar",
        ),
        (
            "https://www.radar-service.eu/id/ccc",
            "https://www.radar-service.eu/radar",
        ),
    ],
)
def test_statistics_base_url_uses_node_host(uri, expected):
    assert _statistics_base_url(_dataset(uri)) == expected


# --------------------------------------------------------------------------- #
# aggregate_access_download_stats (mocked KG + statistics)
# --------------------------------------------------------------------------- #

def test_aggregates_and_sums_statistics():
    datasets = [
        _dataset("https://radar.kit.edu/id/aaa"),
        _dataset("https://radar4chem.radar-service.eu/id/bbb"),
    ]
    stats_by_id = {
        "aaa": {"totalAccess": 100, "totalDownloads": 3},
        "bbb": {"totalAccess": 40, "totalDownloads": 7},
    }

    with patch(
        "radar_jupyter.stats.list_datasets_by_year", return_value=datasets
    ), patch(
        "radar_jupyter.stats.RadarApiClient.get_statistics",
        side_effect=lambda rid: stats_by_id[rid],
    ):
        result = aggregate_access_download_stats(2024, progress=False)

    assert result == YearAccessStats(
        year=2024,
        total_access=140,
        total_downloads=10,
        dataset_count=2,
        counted_datasets=2,
    )


def test_skips_datasets_whose_statistics_fail():
    datasets = [
        _dataset("https://radar.kit.edu/id/ok"),
        _dataset("https://radar.kit.edu/id/broken"),
    ]

    def _get(rid):
        if rid == "broken":
            raise requests.HTTPError("404")
        return {"totalAccess": 50, "totalDownloads": 2}

    with patch(
        "radar_jupyter.stats.list_datasets_by_year", return_value=datasets
    ), patch(
        "radar_jupyter.stats.RadarApiClient.get_statistics", side_effect=_get
    ):
        result = aggregate_access_download_stats(2024, progress=False)

    assert result.total_access == 50
    assert result.total_downloads == 2
    assert result.dataset_count == 2
    assert result.counted_datasets == 1


def test_reuses_one_client_per_host():
    datasets = [
        _dataset("https://radar.kit.edu/id/a"),
        _dataset("https://radar.kit.edu/id/b"),
        _dataset("https://radar4chem.radar-service.eu/id/c"),
    ]
    with patch(
        "radar_jupyter.stats.list_datasets_by_year", return_value=datasets
    ), patch(
        "radar_jupyter.stats.RadarApiClient.get_statistics",
        return_value={"totalAccess": 1, "totalDownloads": 1},
    ), patch("radar_jupyter.stats.RadarApiClient.__init__", return_value=None) as init:
        aggregate_access_download_stats(2024, progress=False)

    # Two distinct hosts → two client constructions, despite three datasets.
    base_urls = {call.args[0] for call in init.call_args_list}
    assert base_urls == {
        "https://radar.kit.edu/radar",
        "https://radar4chem.radar-service.eu/radar",
    }


# --------------------------------------------------------------------------- #
# Statistics caching
# --------------------------------------------------------------------------- #

def test_second_aggregation_uses_cache_and_skips_requests():
    datasets = [
        _dataset("https://radar.kit.edu/id/aaa"),
        _dataset("https://radar4chem.radar-service.eu/id/bbb"),
    ]
    stats_by_id = {
        "aaa": {"totalAccess": 100, "totalDownloads": 3},
        "bbb": {"totalAccess": 40, "totalDownloads": 7},
    }

    with patch(
        "radar_jupyter.stats.list_datasets_by_year", return_value=datasets
    ), patch(
        "radar_jupyter.stats.RadarApiClient.get_statistics",
        side_effect=lambda rid: stats_by_id[rid],
    ) as get_stats:
        first = aggregate_access_download_stats(2024, progress=False)
        second = aggregate_access_download_stats(2024, progress=False)

    assert first == second
    # Statistics fetched once per dataset on the first call; served from cache
    # on the second, so no additional HTTP calls.
    assert get_stats.call_count == 2


def test_use_cache_false_always_refetches():
    datasets = [_dataset("https://radar.kit.edu/id/aaa")]

    with patch(
        "radar_jupyter.stats.list_datasets_by_year", return_value=datasets
    ), patch(
        "radar_jupyter.stats.RadarApiClient.get_statistics",
        return_value={"totalAccess": 1, "totalDownloads": 1},
    ) as get_stats:
        aggregate_access_download_stats(2024, progress=False, use_cache=False)
        aggregate_access_download_stats(2024, progress=False, use_cache=False)

    assert get_stats.call_count == 2


def test_clear_stats_cache_forces_refetch():
    datasets = [_dataset("https://radar.kit.edu/id/aaa")]

    with patch(
        "radar_jupyter.stats.list_datasets_by_year", return_value=datasets
    ), patch(
        "radar_jupyter.stats.RadarApiClient.get_statistics",
        return_value={"totalAccess": 1, "totalDownloads": 1},
    ) as get_stats:
        aggregate_access_download_stats(2024, progress=False)
        clear_stats_cache()
        aggregate_access_download_stats(2024, progress=False)

    assert get_stats.call_count == 2


def test_transient_errors_are_not_cached_but_http_errors_are():
    datasets = [
        _dataset("https://radar.kit.edu/id/flaky"),
        _dataset("https://radar.kit.edu/id/missing"),
    ]
    calls = {"flaky": 0, "missing": 0}

    def _get(rid):
        calls[rid] += 1
        if rid == "flaky":
            # Times out the first time, succeeds the second.
            if calls["flaky"] == 1:
                raise requests.Timeout("slow host")
            return {"totalAccess": 5, "totalDownloads": 1}
        raise requests.HTTPError("404")  # stable "no statistics"

    with patch(
        "radar_jupyter.stats.list_datasets_by_year", return_value=datasets
    ), patch(
        "radar_jupyter.stats.RadarApiClient.get_statistics", side_effect=_get
    ):
        first = aggregate_access_download_stats(2024, progress=False)
        second = aggregate_access_download_stats(2024, progress=False)

    # First pass: flaky timed out (skipped, not cached), missing 404'd (cached).
    assert first.counted_datasets == 0
    # Second pass: flaky is retried and now succeeds; missing stays cached (no
    # second request), so exactly one more call to the flaky host was made.
    assert second.counted_datasets == 1
    assert second.total_access == 5
    assert calls == {"flaky": 2, "missing": 1}


# --------------------------------------------------------------------------- #
# plot_access_download_ratio (mocked aggregation)
# --------------------------------------------------------------------------- #

def test_plot_draws_two_wedges_with_absolute_and_percent():
    stats = YearAccessStats(2024, 300, 100, dataset_count=5, counted_datasets=5)
    with patch(
        "radar_jupyter.stats.aggregate_access_download_stats", return_value=stats
    ):
        ax = plot_access_download_ratio(2024, progress=False)

    wedges = [p for p in ax.patches]
    assert len(wedges) == 2  # Access + Downloads

    texts = " ".join(t.get_text() for t in ax.texts)
    assert "Access" in texts and "Downloads" in texts
    # absolute counts and their percentages both appear
    assert "300" in texts and "100" in texts
    assert "75.0%" in texts and "25.0%" in texts
    # total dataset count shown in the title
    assert "5 datasets" in ax.get_title()
    plt.close(ax.figure)


def test_plot_notes_partial_statistics_coverage():
    stats = YearAccessStats(2024, 10, 5, dataset_count=8, counted_datasets=6)
    with patch(
        "radar_jupyter.stats.aggregate_access_download_stats", return_value=stats
    ):
        ax = plot_access_download_ratio(2024, progress=False)
    assert "8 datasets" in ax.get_title()
    assert "6 with statistics" in ax.get_title()
    plt.close(ax.figure)


def test_plot_raises_when_no_datasets():
    stats = YearAccessStats(1999, 0, 0, dataset_count=0, counted_datasets=0)
    with patch(
        "radar_jupyter.stats.aggregate_access_download_stats", return_value=stats
    ):
        with pytest.raises(ValueError, match="No datasets"):
            plot_access_download_ratio(1999, progress=False)


def test_plot_raises_when_no_access_or_downloads():
    stats = YearAccessStats(2024, 0, 0, dataset_count=3, counted_datasets=3)
    with patch(
        "radar_jupyter.stats.aggregate_access_download_stats", return_value=stats
    ):
        with pytest.raises(ValueError, match="no recorded access"):
            plot_access_download_ratio(2024, progress=False)


# --------------------------------------------------------------------------- #
# Live integration test against the real endpoints
# --------------------------------------------------------------------------- #

@pytest.mark.network
def test_live_aggregate_access_download_stats():
    stats = aggregate_access_download_stats(2024, progress=False)
    assert stats.dataset_count > 0
    assert stats.counted_datasets > 0
    assert stats.total_access > 0
    assert stats.total_access >= stats.total_downloads
