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
)


def _download_file(
    radar_id: str,
    client: RadarApiClient,
    chunk_size: int = 1024 * 1024,
    timeout: Optional[float] = 30,
) -> Path:
    """
    Download a dataset TAR archive from RADAR, printing progress to stdout.

    The download URL is resolved via the RADAR API, which handles redirects to
    the actual storage location.

    :param radar_id: The RADAR dataset ID.
    :param client: An authenticated RadarApiClient instance.
    :param chunk_size: Number of bytes to read per chunk.
    :param timeout: Stream request timeout in seconds. Use None to disable.
    :return: Path to the downloaded TAR file.
    """
    download_dir = Path(DOWNLOAD_DIR)
    existing = next(download_dir.glob(f"*{radar_id}*"), None)
    if existing is not None:
        sys.stdout.write(f"Skipping download, using cached file: {existing.name}\n")
        sys.stdout.flush()
        return existing

    url = client.get_download_url(radar_id)
    output_path = download_dir

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        filename = _get_filename_from_content_disposition_header(
            response.headers.get("content-disposition"), f"{radar_id}.tar"
        )
        output_path = output_path / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue

                file.write(chunk)
                downloaded += len(chunk)

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

    sys.stdout.write("\n")
    sys.stdout.flush()
    return output_path


def download_radar_metadata(
    radar_id: str,
    metadata_type: RadarMetadataType = RadarMetadataType.JSON,
    chunk_size: int = 1024 * 1024,
    timeout: Optional[float] = 30,
) -> Path:
    """
    Download RADAR dataset metadata in a specific format and save it to a file.

    Use this when you need the raw metadata file (JSON-LD, RADAR XML, etc.).
    For plain JSON metadata, prefer ``RadarApiClient.get_dataset()`` instead —
    it returns the data directly without writing a file.

    :param radar_id: The RADAR dataset ID.
    :param metadata_type: The schema type of the metadata to download.
    :param chunk_size: Number of bytes to read per chunk.
    :param timeout: Request timeout in seconds. Use None to disable.
    :return: Path to the downloaded metadata file.
    """
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
