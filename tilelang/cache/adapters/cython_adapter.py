"""Cache adapter for Cython backend."""

from __future__ import annotations

import os
from typing import Any

from tilelang.cache.adapter import CacheAdapterRegistry, CacheFileInfo
from tilelang.cache.adapters.base import BaseCacheAdapter
from tilelang.jit import JITKernel


@CacheAdapterRegistry.register("cython")
class CythonCacheAdapter(BaseCacheAdapter):
    """Cache adapter for Cython execution backend."""

    def get_cache_files(self, kernel: JITKernel) -> dict[str, CacheFileInfo]:
        """Get files to cache for Cython kernel.

        Cython backend caches:
        - device_kernel.cu: Device kernel source code
        - host_kernel.cu: Host wrapper source code
        - kernel_lib.so: Compiled shared library
        """
        files = {}

        # Device kernel source
        if kernel.kernel_source is not None:
            files["device_kernel.cu"] = CacheFileInfo(
                path="device_kernel.cu", content=kernel.kernel_source, mode="w", is_required=False, description="Device kernel source code"
            )

        # Host kernel source
        if hasattr(kernel.adapter, "get_kernel_source"):
            host_source = kernel.adapter.get_kernel_source()
            if host_source:
                files["host_kernel.cu"] = CacheFileInfo(
                    path="host_kernel.cu", content=host_source, mode="w", is_required=False, description="Host wrapper source code"
                )

        # Shared library
        if hasattr(kernel.adapter, "libpath") and kernel.adapter.libpath:
            try:
                with open(kernel.adapter.libpath, "rb") as f:
                    lib_content = f.read()
                files["kernel_lib.so"] = CacheFileInfo(
                    path="kernel_lib.so", content=lib_content, mode="wb", is_required=True, description="Compiled shared library"
                )
            except OSError:
                pass

        return files

    def load_from_cache(self, cache_path: str) -> dict[str, Any] | None:
        """Load Cython kernel data from cache.

        Args:
            cache_path: Path to cache directory

        Returns:
            Dictionary with loaded data:
            - device_kernel_source: Device kernel source (optional)
            - host_kernel_source: Host wrapper source (optional)
            - kernel_lib_path: Path to shared library
        """
        # Load common files
        result = self._load_common_files(cache_path)

        # Add library path
        lib_path = os.path.join(cache_path, "kernel_lib.so")
        if os.path.exists(lib_path):
            result["kernel_lib_path"] = lib_path
        else:
            return None

        return result

    def get_required_files(self) -> list[str]:
        """Get required files for Cython cache.

        Returns:
            List of required file paths
        """
        return ["kernel_lib.so"]
