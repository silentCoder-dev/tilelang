"""Cache adapter for TVM FFI backend."""

from __future__ import annotations

import os
from typing import Any

from tilelang.cache.adapter import CacheAdapterRegistry, CacheFileInfo
from tilelang.cache.adapters.base import BaseCacheAdapter
from tilelang.jit import JITKernel


@CacheAdapterRegistry.register("tvm_ffi")
class TVMFFICacheAdapter(BaseCacheAdapter):
    """Cache adapter for TVM FFI execution backend."""

    def get_cache_files(self, kernel: JITKernel) -> dict[str, CacheFileInfo]:
        """Get files to cache for TVM FFI kernel.

        TVM FFI backend caches:
        - device_kernel.cu: Device kernel source code
        - host_kernel.cu: Host wrapper source code
        - executable.so: Compiled executable library
        """
        files = {}

        # Device kernel source
        if kernel.kernel_source is not None:
            files["device_kernel.cu"] = CacheFileInfo(
                path="device_kernel.cu", content=kernel.kernel_source, mode="w", is_required=False, description="Device kernel source code"
            )

        # Host kernel source
        if hasattr(kernel.adapter, "get_host_source"):
            host_source = kernel.adapter.get_host_source()
            if host_source:
                files["host_kernel.cu"] = CacheFileInfo(
                    path="host_kernel.cu", content=host_source, mode="w", is_required=False, description="Host wrapper source code"
                )

        # Executable library - we'll store the executable object
        # and let KernelCache handle the actual export
        if hasattr(kernel.adapter, "executable") and kernel.adapter.executable:
            files["executable.so"] = CacheFileInfo(
                path="executable.so",
                content=kernel.adapter.executable,
                mode="wb",
                is_required=True,
                description="Compiled executable library",
                # Add a flag to indicate this is an Executable object
                # that needs special handling
                is_executable=True,
            )

        return files

    def load_from_cache(self, cache_path: str) -> dict[str, Any] | None:
        """Load TVM FFI kernel data from cache.

        Args:
            cache_path: Path to cache directory

        Returns:
            Dictionary with loaded data:
            - device_kernel_source: Device kernel source (optional)
            - host_kernel_source: Host wrapper source (optional)
            - kernel_lib_path: Path to executable library
        """
        # Load common files
        result = self._load_common_files(cache_path)

        # Add executable path
        executable_path = os.path.join(cache_path, "executable.so")
        if os.path.exists(executable_path):
            result["kernel_lib_path"] = executable_path
        else:
            return None

        return result

    def get_required_files(self) -> list[str]:
        """Get required files for TVM FFI cache.

        Returns:
            List of required file paths
        """
        return ["executable.so"]
