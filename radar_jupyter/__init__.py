import sys
from pathlib import Path

from .client import DOWNLOAD_DIR, EXTRACTION_DIR, RadarApiClient, RadarMetadataType
from .download import _download_file, download_radar_metadata
from .extract import _safe_extract_tar_with_progress
from .verify import verify_tar_checksum

__all__ = [
    "RadarApiClient",
    "RadarMetadataType",
    "download_and_extract",
    "download_radar_metadata",
]


def download_and_extract(radar_id: str, client: RadarApiClient | None = None) -> Path:
    """
    Download, verify, and extract a RADAR dataset by its dataset ID.

    Dataset metadata is fetched from the RADAR API and the downloaded TAR archive
    is verified against the recorded MD5 checksum before extraction proceeds.

    :param radar_id: The RADAR dataset ID.
    :param client: Optional ``RadarApiClient`` instance. A default client is created if omitted.
    :return: Path to the extracted dataset directory.
    :raises ValueError: If checksum verification fails.

    Example::

        from radar_jupyter import download_and_extract
        data_path = download_and_extract("your-radar-id")
    """
    if client is None:
        client = RadarApiClient()

    extracted_dir = (Path(EXTRACTION_DIR) / radar_id).resolve()
    if extracted_dir.exists():
        tar_path = next(Path(DOWNLOAD_DIR).glob(f"*{radar_id}*"), None)
        if tar_path is not None:
            metadata = client.get_dataset_metadata(radar_id)
            verify_tar_checksum(tar_path, metadata)

        subdirs = [p for p in extracted_dir.iterdir() if p.is_dir()]
        cached_path = subdirs[0] if len(subdirs) == 1 else extracted_dir
        sys.stdout.write(f"Dataset already extracted at: {cached_path}\n")
        sys.stdout.flush()
        return cached_path

    dataset_tar_path = _download_file(radar_id, client)
    metadata = client.get_dataset_metadata(radar_id)
    verify_tar_checksum(dataset_tar_path, metadata)
    return _safe_extract_tar_with_progress(dataset_tar_path, radar_id)
