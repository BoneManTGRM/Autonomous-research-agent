"""
Safe storage utilities for long running autonomous agents.

This module centralizes low‑level file I/O so that:
 - Writes are atomic
 - Reads are safe even if a previous write was interrupted
 - Simple file locks reduce the chance of concurrent corruption
 - Render or other slow platforms cannot easily break the memory file

It is intentionally lightweight and has zero external dependencies.

Changes in this version:
 - Ensure file contents are flushed and synced to disk before replacing the
   original file. This reduces the risk of data loss due to hardware or
   operating system crashes during a write operation. The directory entry is
   also synced after a successful replace.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union


JsonType = Union[Dict[str, Any], list, int, float, str, bool, None]


class FileLock:
    """
    Very simple in‑process lock for a path.

    This does not protect against multi‑process access in all cases, but it
    greatly reduces risk for the common case where only one worker is writing
    frequently.

    For true cross‑process locking you can extend this later with platform
    specific primitives or a lock file protocol.
    """

    _locks: Dict[Path, threading.Lock] = {}
    _global_lock = threading.Lock()

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path).absolute()

        # Use a global lock to safely create one lock per path
        with FileLock._global_lock:
            if self.path not in FileLock._locks:
                FileLock._locks[self.path] = threading.Lock()
            self._lock = FileLock._locks[self.path]

    def acquire(self) -> None:
        """Acquire the underlying lock."""
        self._lock.acquire()

    def release(self) -> None:
        """Release the underlying lock."""
        self._lock.release()

    @contextmanager
    def hold(self):
        """
        Context manager to acquire and release the lock around a block.

        Usage:

            with file_lock.hold():
                # do work exclusively
                ...
        """
        self.acquire()
        try:
            yield
        finally:
            self.release()


class Storage:
    """
    Convenience wrapper that provides safe JSON and text file operations.

    Features:
    - Ensures parent directories exist
    - Atomic write pattern: write to a temporary file and then replace
    - Optional in‑process lock for thread safety
    - Silent failure mode on write errors (for resiliency)

    This implementation flushes and fsyncs the temporary file before
    replacement, and fsyncs the parent directory after replacement, to
    increase reliability on platforms that support it.
    """

    def __init__(self, path: Union[str, Path], use_lock: bool = True) -> None:
        self.path = Path(path)
        self.use_lock = use_lock
        self._lock = FileLock(self.path) if use_lock else None

        # Ensure parent directory exists when possible
        parent = self.path.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # JSON helpers
    # -----------------------------
    def load_json(self, *, default: Optional[JsonType] = None) -> JsonType:
        """
        Load JSON from disk.

        If the file does not exist or cannot be decoded, return `default`
        or a sensible fallback (an empty dict).
        """
        if not self.path.exists():
            return {} if default is None else default

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {} if default is None else default

    def save_json(self, data: JsonType) -> None:
        """
        Persist JSON to disk using an atomic write pattern.

        Writes to `<path>.tmp` first, flushes and fsyncs the temporary file, then
        replaces the main file. Errors are swallowed so that the agent is not
        terminated during long unattended runs.
        """
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")

        def _write() -> None:
            try:
                # Write JSON to temporary file and ensure it is flushed to disk
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    # Flush Python's internal buffers
                    f.flush()
                    # Synchronize OS buffers to disk if supported
                    try:
                        os.fsync(f.fileno())
                    except Exception:
                        # If fsync is unavailable (e.g., on some platforms), ignore
                        pass
                # Atomically replace the target file
                os.replace(tmp_path, self.path)
                # Sync the directory entry to ensure the rename is persisted
                try:
                    dir_fd = os.open(str(self.path.parent), os.O_DIRECTORY)
                except Exception:
                    dir_fd = None
                if dir_fd is not None:
                    try:
                        os.fsync(dir_fd)
                    except Exception:
                        pass
                    finally:
                        try:
                            os.close(dir_fd)
                        except Exception:
                            pass
            except Exception:
                # Clean up temporary file on error
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass

        # Acquire lock if configured
        if self._lock is not None:
            with self._lock.hold():
                _write()
        else:
            _write()

    # -----------------------------
    # Text helpers (for logs, reports)
    # -----------------------------
    def load_text(self, *, default: str = "") -> str:
        """
        Load plain text from disk.

        If the file does not exist or cannot be read, return `default`.
        """
        if not self.path.exists():
            return default
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return default

    def save_text(self, text: str) -> None:
        """
        Save plain text using the same atomic pattern as JSON.

        Writes to `<path>.tmp`, flushes and fsyncs, then replaces the main file.
        Errors are swallowed to maintain resiliency during unattended runs.
        """
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")

        def _write() -> None:
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(text)
                    # Flush and sync the temporary file
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except Exception:
                        pass
                os.replace(tmp_path, self.path)
                # Sync directory entry
                try:
                    dir_fd = os.open(str(self.path.parent), os.O_DIRECTORY)
                except Exception:
                    dir_fd = None
                if dir_fd is not None:
                    try:
                        os.fsync(dir_fd)
                    except Exception:
                        pass
                    finally:
                        try:
                            os.close(dir_fd)
                        except Exception:
                            pass
            except Exception:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass

        if self._lock is not None:
            with self._lock.hold():
                _write()
        else:
            _write()


# Convenience functions if you do not want to instantiate Storage manually

def load_json_file(path: Union[str, Path], *, default: Optional[JsonType] = None) -> JsonType:
    """One‑line helper to load JSON from a file path."""
    return Storage(path).load_json(default=default)


def save_json_file(path: Union[str, Path], data: JsonType) -> None:
    """One‑line helper to save JSON to a file path."""
    Storage(path).save_json(data)


def load_text_file(path: Union[str, Path], *, default: str = "") -> str:
    """One‑line helper to load text from a file path."""
    return Storage(path).load_text(default=default)


def save_text_file(path: Union[str, Path], text: str) -> None:
    """One‑line helper to save text to a file path."""
    Storage(path).save_text(text)
