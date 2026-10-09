


from __future__ import annotations

import hashlib
import os
from pathlib import Path


_CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(path_arg: str | Path) -> str:

    return file_identity(path_arg)["sha256"]


def file_identity(path_arg: str | Path) -> dict[str, str | int]:


    path = Path(path_arg).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a regular file, got {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        path_before = path.stat()
        file_before = os.fstat(handle.fileno())
        if (path_before.st_dev, path_before.st_ino) != (
            file_before.st_dev,
            file_before.st_ino,
        ):
            raise RuntimeError(f"File changed while opening it for hashing: {path}")
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
        file_after = os.fstat(handle.fileno())
    path_after = path.stat()
    for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns"):
        if getattr(file_before, field) != getattr(file_after, field) or getattr(
            file_before, field
        ) != getattr(path_after, field):
            raise RuntimeError(f"File changed while hashing it: {path}")
    return {
        "path": str(path),
        "size_bytes": file_after.st_size,
        "sha256": digest.hexdigest(),
    }


def content_identity(path_arg: str | Path) -> dict[str, str | int]:

    identity = file_identity(path_arg)
    return {"size_bytes": identity["size_bytes"], "sha256": identity["sha256"]}
