import sys
import tarfile
from pathlib import Path
from typing import Optional

from .client import EXTRACTION_DIR


def _safe_extract_tar_with_progress(
    tar_path: Path, extraction_folder: Optional[str] = None
) -> Path:
    """
    Extract a TAR file, printing progress to stdout.

    Extraction is path-traversal safe: any member whose resolved path would escape
    the output directory raises a ValueError before extraction begins.

    :param tar_path: Path to the TAR file to extract.
    :param extraction_folder: Optional sub-folder beneath EXTRACTION_DIR.
    :return: Path to the extraction root. If the archive contains a single top-level
             directory, that directory's path is returned instead of the extraction root.
    """
    subfolder = (
        ""
        if extraction_folder is None or len(extraction_folder) == 0
        else f"/{extraction_folder}"
    )
    output_dir = Path(f"{EXTRACTION_DIR}{subfolder}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:*") as tar:
        members = tar.getmembers()
        total_members = len(members)
        root_names: set[str] = set()

        for member in members:
            member_path = (output_dir / member.name).resolve()
            if output_dir not in member_path.parents and member_path != output_dir:
                raise ValueError(f"Unsafe path in tar file: {member.name}")
            parts = Path(member.name).parts
            if parts:
                root_names.add(parts[0])

        for index, member in enumerate(members, start=1):
            tar.extract(member, path=output_dir)

            percent = index * 100 / total_members if total_members else 100
            sys.stdout.write(
                f"\rExtracted {index}/{total_members} files ({percent:.1f}%)"
            )
            sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()

    if len(root_names) == 1:
        root_path = output_dir / next(iter(root_names))
        if root_path.is_dir():
            return root_path
    return output_dir
