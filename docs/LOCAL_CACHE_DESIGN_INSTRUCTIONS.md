# Instructions: Building a Local Disk Cache in Front of a Slow Source (Drive/S3/NFS/etc.)

Hand this document to an LLM implementing data loading for a training pipeline where the dataset
lives on slow, possibly network-mounted storage (Google Drive, S3, NFS, a mounted network share)
and there is a fast local scratch disk available (e.g. Colab's `/content`, a local SSD, a
container's ephemeral filesystem). It specifies the exact cache design used successfully in the
SignVLM project — implement this pattern, not a bespoke one, unless there's a specific reason this
project's constraints don't apply.

## Goal

Make every file/sample read hit the slow source **at most once**, ever (per runtime lifetime),
regardless of how many epochs, how many DataLoader worker processes, or how many different dataset
splits are reading concurrently — with zero risk of two different logical datasets silently
aliasing each other's cached files, and zero risk of a crash mid-copy leaving a corrupted-but-complete-looking cache entry.

Do **not** build a bulk "sync/mirror everything up front" step. Build a **lazy, on-first-use**
cache: a sample is copied into the cache the first time something actually asks for it, not before.
This lets training/inference start immediately instead of waiting for a multi-minute upfront copy,
and it means you never pay to cache samples that end up unused (e.g. an aborted run, a capped
`max_samples` debug run).

## 1. Directory structure

```
<cache_root>/
├── .cache_locks/                      # per-key lock files, one per in-flight copy
│   └── <sha256(lock_key)>.flock
├── <namespace_A>/                     # one subtree per distinct source root (see §3)
│   └── <relpath mirrors source layout exactly>
│       ├── some/class/clip001.mp4
│       └── some/class/clip001/            # if caching extracted frames alongside raw files
│           ├── 00000.jpg
│           └── 00001.jpg
└── <namespace_B>/
    └── ...
```

Rules:
- The relative path layout **inside** a namespace must exactly mirror the relative path layout of
  the source root it was copied from. Do not flatten, rename, or hash filenames — this keeps the
  cache debuggable (a human can `ls` it and recognize what's there) and lets any code elsewhere
  that already knows the relpath convention work unmodified against either the source root or the
  cache root.
- Everything under `cache_root` is treated as **fully disposable and rebuildable from the source**
  at any time. Never write anything to the cache that doesn't also exist (or is derivable from
  something that exists) on the source. Never treat the cache as a place to persist results,
  checkpoints, logs, or anything else that matters if lost.

## 2. Public API surface

Implement one function that every read path calls instead of touching the source root directly:

```python
def resolve_cached_path(source_root: str, relpath: str, cache_root: str, *, is_dir: bool = False) -> str:
    """
    Return a local filesystem path to `relpath` under `source_root`, guaranteed to exist and be
    fully readable, mirrored into `cache_root` on first call for this (source_root, relpath) pair.
    Every later call for the same pair returns the cached path immediately with no I/O against
    source_root. Raises FileNotFoundError if the source itself doesn't have this file/dir.
    """
```

`is_dir=True` selects the "mirror a whole directory" mode (e.g. a folder of extracted frame
images) instead of "copy a single file" mode — both need the same namespacing, locking, and
atomicity guarantees, but the copy primitive differs (`shutil.copy2` vs `shutil.copytree`).

If your pipeline reads from more than one logical dataset split (train/val/test/unseen, or
multiple datasets entirely) that might share a single `cache_root`, this same function must be
called with each split's own `source_root` — never assume a single global source root.

## 3. Namespacing — the single most important rule

**Never key a cache path by the relative path alone.** Two different source roots (e.g.
`train_split/` and `val_split/`) routinely contain files with identical relative paths (every
split's `ClassA/` folder has its own independently-numbered `1.mp4`). If the cache path is just
`cache_root/<relpath>`, the second split to touch that relpath will resolve to the **same file
the first split already cached** — silently serving split A's content under split B's label, with
**no error, no warning, nothing visibly wrong** until someone notices metrics look suspicious.

Fix: derive a namespace from a hash of the **resolved absolute path** of `source_root`, and put
every namespace in its own subtree:

```python
import hashlib
from pathlib import Path

def _namespace_for(source_root: str) -> str:
    resolved = str(Path(source_root).resolve())
    h = hashlib.sha256(resolved.encode("utf-8", errors="surrogateescape")).hexdigest()[:8]
    name = Path(source_root).name or "root"
    return f"{name}_{h}"          # human-readable prefix + collision-proof suffix
```

Then every cached path is `cache_root / _namespace_for(source_root) / relpath`. Do this **even if
you currently only have one source root** — it costs nothing and it means the moment a second
split/dataset is added later, there is no silent data-corruption bug waiting to happen. Retrofit
is much more expensive: it requires proving old cache entries are safe to keep or must be nuked and
rebuilt, and any experiment run before the fix is now suspect.

## 4. Atomic writes — never leave a half-copied entry

Every write into the cache — whether one file or a whole directory — must go through a
temp-name-then-atomic-rename sequence, so a process crash, OOM kill, or Colab runtime recycle
mid-copy can never leave something that *looks* complete (and would pass the idempotency check
below) but is actually truncated or empty.

Single file:
```python
import os, shutil, stat, tempfile
from pathlib import Path

def _atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=dst.name + ".", suffix=".partial")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        shutil.copy2(src, tmp_path, follow_symlinks=True)
        os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        os.replace(str(tmp_path), str(dst))   # atomic on POSIX and Windows (NTFS)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
```

Whole directory (e.g. a folder of extracted frames):
```python
def _atomic_copytree(src: Path, dst: Path) -> None:
    part = dst.parent / (dst.name + ".partial")
    if part.exists():
        shutil.rmtree(part, ignore_errors=True) if part.is_dir() else part.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, part, dirs_exist_ok=True, symlinks=True)
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True) if dst.is_dir() else dst.unlink()
    os.replace(str(part), str(dst))
```

The idempotency check elsewhere (§5) must only ever treat a cache entry as "done" by checking the
**final** name, never the `.partial` name — that's what makes this safe.

## 5. Idempotency check ("is this already cached?")

Before copying anything, check whether the destination is already a complete, valid cache entry,
and if so, return immediately with **zero** source-root I/O:

- File mode: `dest_file.is_file() and dest_file.stat().st_size > 0`
- Directory mode: directory exists **and** contains at least one file of the expected type (e.g.
  `any(dest_dir.glob("*.jpg"))`) — an empty directory must not count as "cached," because some
  filesystem APIs (e.g. `Path.glob()` on a nonexistent/empty dir) return an empty result rather
  than raising, which can make a broken or not-yet-populated cache look silently "fine."

Recheck this same condition **again after acquiring the lock** (§6) — another process may have
finished the copy while this one was waiting for the lock, and repeating the copy would be wasted
work (or, worse, race with the atomic rename above).

## 6. Cross-process locking

If more than one process (e.g. DataLoader worker processes) can request the same never-yet-cached
key concurrently, guard the copy with a per-key lock so they don't duplicate the copy or race on
the atomic rename. Use a real file lock (`filelock` package if available) so it works across
separate processes, not just threads within one process:

```python
import hashlib
from pathlib import Path

def _lock_path(cache_root: str, key: str) -> Path:
    h = hashlib.sha256(key.encode("utf-8", errors="surrogateescape")).hexdigest()
    return Path(cache_root) / ".cache_locks" / f"{h}.flock"
```

The lock key must include the namespace (§3), not just the relpath, for the same collision reason
as everything else here. Prefer a library (`filelock.FileLock(path, timeout=...)`) over hand-rolled
locking; if you must hand-roll for a platform without it, implement both a POSIX path (`fcntl.flock`)
and a Windows path (`msvcrt.locking`) — don't assume one OS.

Always set a generous but finite timeout (e.g. an hour) rather than blocking forever, so a lock
left behind by a killed process doesn't wedge the entire pipeline; surface a clear `TimeoutError`
naming the lock path if it's ever hit.

## 7. Path sanitization

Before treating any `relpath` as a filesystem path (whether it came from a manifest file, a
DataLoader, or anywhere else), reject anything that could escape the intended tree:

```python
import os

def _sanitize_rel(rel: str) -> str:
    rel = rel.strip()
    p = os.path.normpath(rel)
    if p.startswith("..") or os.path.isabs(p):
        raise ValueError(f"Invalid relative path: {rel!r}")
    return p
```

This matters even for "trusted" internal manifests — a bug elsewhere that writes a bad relpath
into a TSV/JSON list should fail loudly here, not silently write outside the cache tree.

## 8. Full reference algorithm (single-file mode)

```python
def resolve_cached_path(source_root, relpath, cache_root, is_dir=False):
    if not cache_root:
        return os.path.join(source_root, relpath)      # caching disabled entirely

    rel = _sanitize_rel(relpath)
    ns = _namespace_for(source_root)
    src = Path(source_root) / rel
    dst = Path(cache_root) / ns / rel
    lock_key = f"{ns}\0{rel}"

    if dst.is_file() and dst.stat().st_size > 0:
        return str(dst)                                  # already cached, zero source I/O

    if not src.is_file():
        raise FileNotFoundError(f"Source not found: {src} (relpath={relpath!r})")

    with _lock_ctx(_lock_path(cache_root, lock_key)):
        if dst.is_file() and dst.stat().st_size > 0:      # re-check: another process may have won
            return str(dst)
        _atomic_copy_file(src, dst)

    return str(dst)
```

Directory mode follows the same shape, swapping the file existence/size check for the
non-empty-directory check (§5) and `_atomic_copy_file` for `_atomic_copytree`.

## 9. Testing checklist before trusting this in a real run

- Two different source roots with an identically-named relpath, sharing one `cache_root`: confirm
  both resolve to distinct cached files with the correct content (this is the collision bug from
  §3 — write an explicit test for it, don't just eyeball the code).
- Kill the process mid-copy (e.g. `SIGKILL` during a large file copy) and confirm the next call
  either has no cache entry (clean retry) or a correctly-restarted one — never a truncated file
  that looks "cached."
- Call `resolve_cached_path` concurrently from multiple processes for the same never-cached key;
  confirm exactly one copy happens and both callers get a valid, complete result.
- Call it for a relpath that doesn't exist on the source; confirm a clear `FileNotFoundError`,
  not a confusing downstream failure several layers away.
- If running on both Linux and Windows, confirm the locking fallback actually works on both — test
  on Windows explicitly if that's a target platform; don't assume the POSIX path was "close enough."

## 10. What this pattern deliberately does NOT do (scope boundaries)

- **No eviction/LRU.** The cache only grows. If the source dataset is larger than local disk, add
  an eviction policy on top of this — this base pattern assumes the full dataset fits in local
  scratch space, which was true for the reference project. Flag this explicitly to whoever's
  driving the project if disk budget is a real constraint.
- **No cache invalidation on source change.** If a source file changes after being cached, the
  cache will keep serving the old copy forever (no timestamp/hash comparison). If the source
  dataset can mutate mid-project (relabeling, fixing corrupt files), either bump the namespace
  (e.g. include a dataset version string in `_namespace_for`) or add an explicit "clear this
  namespace" utility rather than trying to detect staleness automatically.
- **Not a replacement for keeping the source as ground truth.** Never write anything back from the
  cache to the source, and never treat the cache as safe to be the only copy of anything.
