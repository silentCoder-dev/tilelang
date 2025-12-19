"""Cache adapter for NVRTC backend."""

from __future__ import annotations

import os
from typing import Any

from tilelang.cache.adapter import CacheAdapterRegistry, CacheFileInfo
from tilelang.cache.adapters.base import BaseCacheAdapter
from tilelang.jit import JITKernel


@CacheAdapterRegistry.register("nvrtc")
class NVRTCCacheAdapter(BaseCacheAdapter):
    """Cache adapter for NVRTC execution backend."""

    def get_cache_files(self, kernel: JITKernel) -> dict[str, CacheFileInfo]:
        """Get files to cache for NVRTC kernel.

        NVRTC backend caches:
        - device_kernel.cu: Device kernel source code
        - host_kernel.cu: Host wrapper source code
        - kernel.cubin: Compiled CUBIN
        - kernel.py: Python wrapper code
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

        # CUBIN file
        if hasattr(kernel.adapter, "libpath") and kernel.adapter.libpath:
            # Read the CUBIN file
            try:
                with open(kernel.adapter.libpath, "rb") as f:
                    cubin_content = f.read()
                files["kernel.cubin"] = CacheFileInfo(
                    path="kernel.cubin", content=cubin_content, mode="wb", is_required=True, description="Compiled CUBIN"
                )
            except OSError:
                pass

        # Python wrapper file
        if hasattr(kernel.adapter, "libpath") and kernel.adapter.libpath:
            # Try to find corresponding .py file
            py_path = kernel.adapter.libpath.replace(".cubin", ".py")
            if os.path.exists(py_path):
                try:
                    with open(py_path, "rb") as f:
                        py_content = f.read()
                    files["kernel.py"] = CacheFileInfo(
                        path="kernel.py", content=py_content, mode="wb", is_required=False, description="Python wrapper code"
                    )
                except OSError:
                    pass

        return files

    def load_from_cache(self, cache_path: str) -> dict[str, Any] | None:
        """Load NVRTC kernel data from cache.

        Args:
            cache_path: Path to cache directory

        Returns:
            Dictionary with loaded data:
            - device_kernel_source: Device kernel source (optional)
            - host_kernel_source: Host wrapper source (optional)
            - kernel_lib_path: Path to CUBIN file
            - python_wrapper_path: Path to Python wrapper (optional)
        """
        # Load common files
        result = self._load_common_files(cache_path)

        # Add CUBIN path
        cubin_path = os.path.join(cache_path, "kernel.cubin")
        if os.path.exists(cubin_path):
            result["kernel_lib_path"] = cubin_path
        else:
            return None

        # Add Python wrapper path if exists
        py_path = os.path.join(cache_path, "kernel.py")
        if os.path.exists(py_path):
            result["python_wrapper_path"] = py_path

        return result

    def get_required_files(self) -> list[str]:
        """Get required files for NVRTC cache.

        Returns:
            List of required file paths
        """
        return ["kernel.cubin"]
