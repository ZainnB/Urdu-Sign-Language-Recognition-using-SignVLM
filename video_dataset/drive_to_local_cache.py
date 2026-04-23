#!/usr/bin/env python
"""
Mirror samples from a slow data_root (e.g. Google Drive) to local_cache (e.g. /content) on
first use. Idempotent: later passes skip the source and read from local only.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import stat
import sys
import time
import tempfile
from pathlib import Path
from typing import Generator

try:
    from filelock import FileLock
except Exception:  # optional dependency
    FileLock = None  # type: ignore[assignment, misc]

# --- cross-process lock (DataLoader workers) ------------------------------------


def _lock_file_for_relpath(local_cache_dir: str, key: str) -> Path:
    h = hashlib.sha256(key.encode("utf-8", errors="surrogateescape")).hexdigest()
    d = Path(local_cache_dir) / ".signvlm_cache_locks"
    return d / f"{h}.flock"


@contextlib.contextmanager
def _lock_ctx(lock_path: Path) -> Generator[None, None, None]:
    if FileLock is not None:
        with FileLock(str(lock_path), timeout=3600):
            yield
        return
    if sys.platform == "win32":
        from msvcrt import LK_NBLCK, LK_UNLCK, locking  # type: ignore

        lock_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(lock_path, "a+")
        deadline = time.monotonic() + 3600.0
        try:
            while True:
                try:
                    f.seek(0)
                    locking(f.fileno(), LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"cache lock: {lock_path}") from None
                    time.sleep(0.05)
            yield
        finally:
            try:
                f.seek(0)
                locking(f.fileno(), LK_UNLCK, 1)
            except OSError:
                pass
            f.close()
    else:
        import fcntl

        lock_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(lock_path, "a+")
        try:
            fcntl.flock(f, fcntl.LOCK_EX)  # blocking; serializes copy workers
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()


# --- copy helpers ----------------------------------------------------------------


def _is_nonempty_frame_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    for e in p.iterdir():
        if e.is_file() and e.suffix.lower() in (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".webp",
        ):
            return True
    return False


def _atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, partial = tempfile.mkstemp(
        dir=dst.parent, prefix=dst.name + ".", suffix=".partial"
    )
    os.close(fd)
    ppath = Path(partial)
    try:
        shutil.copy2(src, ppath, follow_symlinks=True)
        os.chmod(ppath, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        os.replace(str(ppath), str(dst))
    except Exception:
        if ppath.exists():
            try:
                ppath.unlink()
            except OSError:
                pass
        raise


def _atomic_copytree_to_dst(src: Path, dst: Path) -> None:
    parent = dst.parent
    parent.mkdir(parents=True, exist_ok=True)
    part = parent / (dst.name + ".partial")
    if part.exists():
        if part.is_dir():
            shutil.rmtree(part, ignore_errors=True)
        else:
            if part.is_file():
                part.unlink()
    shutil.copytree(src, part, dirs_exist_ok=True, symlinks=True)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst, ignore_errors=True)
        else:
            if dst.is_file():
                dst.unlink()
    os.replace(str(part), str(dst))


def _sanitize_rel(rel: str) -> str:
    rel = rel.strip()
    p = os.path.normpath(rel)
    if p.startswith("..") or os.path.isabs(p):
        raise ValueError(f"Invalid relative path: {rel!r}")
    return p


def resolve_cached_path(
    frames_available: int,
    data_root: str,
    relpath: str,
    local_cache_dir: str,
) -> str:
    """
    Return full path: same relpath under local_cache_dir after a one-time copy
    from data_root. Reuses local for all later epochs/indices (idempotent).
    """
    if not local_cache_dir or not str(local_cache_dir).strip():
        return os.path.join(data_root, relpath)
    rel = _sanitize_rel(relpath)
    source_file = Path(data_root) / rel
    dest_file = Path(local_cache_dir) / rel
    lock_key = f"{int(bool(frames_available))}\0{rel}"

    if not frames_available:
        if dest_file.is_file() and dest_file.stat().st_size > 0:
            return str(dest_file)
        if not source_file.is_file():
            raise FileNotFoundError(
                f"Source video not found: {source_file} (relpath={relpath!r})"
            )
        with _lock_ctx(_lock_file_for_relpath(local_cache_dir, lock_key)):
            if dest_file.is_file() and dest_file.stat().st_size > 0:
                return str(dest_file)
            _atomic_copy_file(source_file, dest_file)
        return str(dest_file)

    sframe = source_file.parent / source_file.stem
    dframe = dest_file.parent / dest_file.stem
    if dframe.is_dir() and _is_nonempty_frame_dir(dframe):
        if source_file.is_file() and not (dest_file.is_file() and dest_file.stat().st_size > 0):
            with _lock_ctx(
                _lock_file_for_relpath(local_cache_dir, lock_key + "\0side")
            ):
                if source_file.is_file() and not (
                    dest_file.is_file() and dest_file.stat().st_size > 0
                ):
                    _atomic_copy_file(source_file, dest_file)
        return str(dest_file)

    with _lock_ctx(_lock_file_for_relpath(local_cache_dir, lock_key)):
        if dframe.is_dir() and _is_nonempty_frame_dir(dframe):
            if source_file.is_file() and not (
                dest_file.is_file() and dest_file.stat().st_size > 0
            ):
                _atomic_copy_file(source_file, dest_file)
            return str(dest_file)
        if not sframe.is_dir():
            raise FileNotFoundError(
                f"Source frame dir not found: {sframe} (relpath={relpath!r})"
            )
        dframe.parent.mkdir(parents=True, exist_ok=True)
        _atomic_copytree_to_dst(sframe, dframe)
        if source_file.is_file():
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            if not (dest_file.is_file() and dest_file.stat().st_size > 0):
                _atomic_copy_file(source_file, dest_file)

    return str(dest_file)
