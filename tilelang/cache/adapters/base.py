"""Base utilities for cache adapters."""

from __future__ import annotations

import os
from typing import Any

from tilelang.cache.adapter import CacheAdapter


class BaseCacheAdapter(CacheAdapter):
    """Base cache adapter with common utilities."""

    def _read_text_file(self, path: str) -> str | None:
        """Read a text file from cache.

        Args:
            path: Full path to the file

        Returns:
            File content as string, or None if file doesn't exist
        """
        try:
            with open(path) as f:
                return f.read()
        except OSError:
            return None

    def _read_binary_file(self, path: str) -> bytes | None:
        """Read a binary file from cache.

        Args:
            path: Full path to the file

        Returns:
            File content as bytes, or None if file doesn't exist
        """
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            return None

    def _load_common_files(self, cache_path: str) -> dict[str, Any]:
        """Load common files that most backends need.

        Args:
            cache_path: Path to cache directory

        Returns:
            Dictionary with loaded common files
        """
        result = {}

        # Try to load device kernel source
        device_kernel_path = os.path.join(cache_path, "device_kernel.cu")
        device_source = self._read_text_file(device_kernel_path)
        if device_source is not None:
            result["device_kernel_source"] = device_source

        # Try to load host kernel source
        host_kernel_path = os.path.join(cache_path, "host_kernel.cu")
        host_source = self._read_text_file(host_kernel_path)
        if host_source is not None:
            result["host_kernel_source"] = host_source

        return result
