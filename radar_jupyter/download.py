import sys
from pathlib import Path
from typing import Optional

import requests

from .client import (
    DOWNLOAD_DIR,
    RadarApiClient,
    RadarMetadataType,
    _get_filename_from_content_disposition_header,
    _get_metadata_export_url,
    resolve_dataset_id,
)


def _find_completed_download(download_dir: Path, radar_id: str) -> Optional[Path]:
    """Return a previously completed (non ``.part``) download for ``radar_id``, if any."""
    return next(
        (p for p in download_dir.glob(f"*{radar_id}*") if p.suffix != ".part"),
        None,
    )


def _resolve_output_path(
    response: requests.Response, download_dir: Path, radar_id: str, partial: Optional[Path]
) -> Path:
    """Determine the final file path for the download.

    When resuming, the name is taken from the existing ``.part`` file; otherwise
    it is derived from the response's ``Content-Disposition`` header.
    """
    if partial is not None:
        return partial.with_suffix("")
    filename = _get_filename_from_content_disposition_header(
        response.headers.get("content-disposition"), f"{radar_id}.tar"
    )
    return download_dir / filename


def _total_download_size(response: requests.Response, already_downloaded: int) -> int:
    """Total archive size in bytes, or 0 if the server did not report it.

    For a resumed (``206``) response the total is read from the ``Content-Range``
    header; otherwise it is the bytes already on disk plus ``Content-Length``.
    """
    content_range = response.headers.get("content-range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[-1]
        if total.isdigit():
            return int(total)
    return already_downloaded + int(response.headers.get("content-length", 0))


def _print_progress(downloaded: int, total_size: int) -> None:
    """Print a single carriage-return-prefixed download-progress line."""
    if total_size:
        percent = downloaded * 100 / total_size
        progress = (
            f"\rDownloaded {downloaded / 1024 / 1024:.2f} MB "
            f"of {total_size / 1024 / 1024:.2f} MB "
            f"({percent:.1f}%)"
        )
    else:
        progress = f"\rDownloaded {downloaded / 1024 / 1024:.2f} MB"

    sys.stdout.write(progress)
    sys.stdout.flush()


def _stream_to_file(
    response: requests.Response,
    partial_path: Path,
    chunk_size: int,
    downloaded: int,
    total_size: int,
    append: bool,
) -> None:
    """Stream the response body into ``partial_path``, printing progress per chunk.

    ``append`` opens the file for appending (resuming); otherwise it is truncated.
    ``downloaded`` is the number of bytes already present when appending.
    """
    with partial_path.open("ab" if append else "wb") as file:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue

            file.write(chunk)
            downloaded += len(chunk)
            _print_progress(downloaded, total_size)


def _download_file(
    identifier: str,
    client: RadarApiClient,
    chunk_size: int = 1024 * 1024,
    timeout: Optional[float] = 30.0,
    download_url: Optional[str] = None,
) -> Path:
    """
    Download a dataset TAR archive from RADAR, printing progress to stdout.

    When ``download_url`` is given (e.g. a DOI ``rel="item"`` link) it is used
    directly; otherwise the URL is resolved via the RADAR API, which handles
    redirects to the actual storage location.

    :param identifier: Dataset ID, RADAR ID (``RADAR/<id>``), or DOI.
    :param client: A RadarApiClient instance.
    :param chunk_size: Number of bytes to read per chunk.
    :param timeout: Stream request timeout in seconds. Use None to disable.
    :param download_url: Explicit download URL. If omitted, it is resolved via the API.
    :return: Path to the downloaded TAR file.

    The archive is streamed into a temporary ``.part`` file which is renamed to
    its final name only once the download completes. If the download is aborted
    or fails, the ``.part`` file is left in place; a later call resumes it via an
    HTTP ``Range`` request (or restarts from scratch if the server ignores it).
    """
    radar_id = resolve_dataset_id(identifier)
    download_dir = Path(DOWNLOAD_DIR)

    cached = _find_completed_download(download_dir, radar_id)
    if cached is not None:
        sys.stdout.write(f"Skipping download, using cached file: {cached.name}\n")
        sys.stdout.flush()
        return cached

    # A ``.part`` file from an interrupted download; resume from where it stopped.
    partial = next(download_dir.glob(f"*{radar_id}*.part"), None)
    resume_from = partial.stat().st_size if partial is not None else 0

    url = download_url if download_url is not None else client.get_download_url(radar_id)
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
        output_path = _resolve_output_path(response, download_dir, radar_id, partial)

        # The partial file is already complete; the server has nothing more to
        # send. Finalise it and skip the download.
        if partial is not None and response.status_code == 416:
            partial.rename(output_path)
            sys.stdout.write(f"Download already complete: {output_path.name}\n")
            sys.stdout.flush()
            return output_path

        response.raise_for_status()

        # A 206 means the server honoured the Range request and we can append;
        # anything else (typically 200) means we must download the file afresh.
        resuming = response.status_code == 206 and resume_from > 0
        downloaded = resume_from if resuming else 0
        total_size = _total_download_size(response, downloaded)

        partial_path = output_path.with_name(output_path.name + ".part")
        partial_path.parent.mkdir(parents=True, exist_ok=True)

        if resuming:
            sys.stdout.write(
                f"Resuming download from {resume_from / 1024 / 1024:.2f} MB\n"
            )
            sys.stdout.flush()

        _stream_to_file(
            response, partial_path, chunk_size, downloaded, total_size, resuming
        )

    partial_path.rename(output_path)

    sys.stdout.write("\n")
    sys.stdout.flush()
    return output_path


def download_radar_metadata(
    identifier: str,
    metadata_type: RadarMetadataType = RadarMetadataType.JSON,
    chunk_size: int = 1024 * 1024,
    timeout: Optional[float] = 30,
) -> Path:
    """
    Download RADAR dataset metadata in a specific format and save it to a file.

    Use this when you need the raw metadata file (JSON-LD, RADAR XML, etc.).
    For plain JSON metadata, prefer ``RadarApiClient.get_dataset_metadata()`` instead —
    it returns the data directly without writing a file.

    :param identifier: Dataset ID, RADAR ID (``RADAR/<id>``), or DOI.
    :param metadata_type: The schema type of the metadata to download.
    :param chunk_size: Number of bytes to read per chunk.
    :param timeout: Request timeout in seconds. Use None to disable.
    :return: Path to the downloaded metadata file.
    """
    radar_id = resolve_dataset_id(identifier)
    url = _get_metadata_export_url(radar_id, metadata_type)
    output_path = Path(DOWNLOAD_DIR)

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()

        filename = _get_filename_from_content_disposition_header(
            response.headers.get("content-disposition"), f"{radar_id}.json"
        )
        output_path = output_path / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                file.write(chunk)

    return output_path
