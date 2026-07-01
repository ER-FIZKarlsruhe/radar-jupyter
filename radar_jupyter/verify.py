import hashlib
from pathlib import Path


def _get_md5_hash_from_file(file_path: Path, block_size: int = 4096) -> str:
    """
    Calculate the MD5 checksum of a file, reading it in chunks to avoid memory issues.

    :param file_path: Path to the file.
    :param block_size: Chunk size in bytes.
    :return: MD5 checksum as a hex string.
    """
    with open(file_path, "rb") as file:
        md5_hasher = hashlib.md5()
        while chunk := file.read(block_size):
            md5_hasher.update(chunk)
    return md5_hasher.hexdigest()


def verify_tar_checksum(tar_path: Path, metadata: dict) -> None:
    """
    Verify the MD5 checksum of a downloaded RADAR TAR archive against the checksum
    recorded in the dataset metadata returned by the RADAR API.

    :param tar_path: Path to the downloaded TAR file.
    :param metadata: Dataset metadata dict as returned by ``RadarApiClient.get_dataset()``.
    :raises ValueError: If the checksum type is unsupported or the checksums do not match.
    """
    checksum_type = metadata["archive"]["checksumType"]
    if checksum_type != "MD5":
        raise ValueError(
            f"Unsupported checksum type '{checksum_type}' — only MD5 is supported."
        )

    expected = metadata["archive"]["checksum"]
    actual = _get_md5_hash_from_file(tar_path)

    if expected != actual:
        raise ValueError(
            f"Checksum mismatch for '{tar_path.name}': "
            f"expected {expected}, got {actual}."
        )
