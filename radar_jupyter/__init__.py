import sys
from pathlib import Path

from .client import (
    DOWNLOAD_DIR,
    EXTRACTION_DIR,
    RadarApiClient,
    RadarMetadataType,
    is_doi,
    is_verifiable_download_host,
    resolve_dataset_id,
    resolve_doi_download_url,
)
from .download import _download_file, download_radar_metadata
from .extract import _safe_extract_tar_with_progress
from .verify import verify_tar_checksum

__all__ = [
    "RadarApiClient",
    "RadarMetadataType",
    "download_and_extract",
    "download_radar_metadata",
]


def download_and_extract(
    identifier: str, client: RadarApiClient | None = None
) -> Path:
    """
    Download, verify, and extract a RADAR dataset.

    Dataset metadata is fetched from the RADAR API and the downloaded TAR archive
    is verified against the recorded MD5 checksum before extraction proceeds.
    If the dataset was previously downloaded and extracted, the cached result is
    returned after re-verifying the TAR checksum (if the TAR file is still present).

    :param identifier: Dataset ID, RADAR ID (``RADAR/<id>``), or DOI.
    :param client: Optional ``RadarApiClient`` instance. A default client is created if omitted.
    :return: Path to the extracted dataset directory.
    :raises ValueError: If the identifier is invalid or checksum verification fails.

    Example::

        from radar_jupyter import download_and_extract
        data_path = download_and_extract("https://doi.org/10.2222/your-dataset-id")
        data_path = download_and_extract("RADAR/your-dataset-id")
        data_path = download_and_extract("your-dataset-id")
    """
    if client is None:
        client = RadarApiClient()

    radar_id = resolve_dataset_id(identifier)

    # For a DOI, resolve the TAR download URL from its ``rel="item"`` Link header.
    # The checksum can only be verified for downloads served by a known RADAR host.
    download_url: str | None = None
    verify_checksum = True
    if is_doi(identifier):
        download_url = resolve_doi_download_url(identifier)
        verify_checksum = is_verifiable_download_host(download_url)
        if not verify_checksum:
            sys.stdout.write(
                "Warning: the download is not served by a known RADAR host; "
                "the file hash cannot be verified.\n"
            )
            sys.stdout.flush()

    extracted_dir = (Path(EXTRACTION_DIR) / radar_id).resolve()
    if extracted_dir.exists():
        if verify_checksum:
            tar_path = next(Path(DOWNLOAD_DIR).glob(f"*{radar_id}*"), None)
            if tar_path is not None:
                metadata = client.get_dataset_metadata(radar_id)
                verify_tar_checksum(tar_path, metadata)

        subdirs = [p for p in extracted_dir.iterdir() if p.is_dir()]
        cached_path = subdirs[0] if len(subdirs) == 1 else extracted_dir
        sys.stdout.write(f"Dataset already extracted at: {cached_path}\n")
        sys.stdout.flush()
        return cached_path

    dataset_tar_path = _download_file(identifier, client, download_url=download_url)
    if verify_checksum:
        metadata = client.get_dataset_metadata(radar_id)
        verify_tar_checksum(dataset_tar_path, metadata)
    return _safe_extract_tar_with_progress(dataset_tar_path, radar_id)
