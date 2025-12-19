"""Cache adapter for Torch backend."""

from __future__ import annotations

import os
from typing import Any

from tilelang.cache.adapter import CacheAdapterRegistry, CacheFileInfo
from tilelang.cache.adapters.base import BaseCacheAdapter
from tilelang.jit import JITKernel


@CacheAdapterRegistry.register("torch")
class TorchCacheAdapter(BaseCacheAdapter):
    """Cache adapter for Torch execution backend.

    Note: Torch backend may have different caching requirements
    depending on the specific implementation (e.g., Metal, CPU, etc.).
    This is a generic adapter that can be extended for specific
    Torch backends.
    """

    def get_cache_files(self, kernel: JITKernel) -> dict[str, CacheFileInfo]:
        """Get files to cache for Torch kernel.

        Torch backend caches:
        - device_kernel.cu/metal/etc: Device kernel source code
        - host_kernel.cu: Host wrapper source code
        - kernel_lib.so: Compiled shared library
        """
        files = {}

        # Determine file extension based on target
        device_ext = self._get_device_extension(kernel)

        # Device kernel source
        if kernel.kernel_source is not None:
            files[f"device_kernel.{device_ext}"] = CacheFileInfo(
                path=f"device_kernel.{device_ext}",
                content=kernel.kernel_source,
                mode="w",
                is_required=False,
                description=f"Device kernel source code ({device_ext})",
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
        """Load Torch kernel data from cache.

        Args:
            cache_path: Path to cache directory

        Returns:
            Dictionary with loaded data:
            - device_kernel_source: Device kernel source (optional)
            - host_kernel_source: Host wrapper source (optional)
            - kernel_lib_path: Path to shared library
        """
        # Try to find device kernel file with any extension
        device_source = None
        for ext in ["cu", "metal", "cpp", "c"]:
            device_path = os.path.join(cache_path, f"device_kernel.{ext}")
            source = self._read_text_file(device_path)
            if source is not None:
                device_source = source
                break

        # Load host kernel source
        host_path = os.path.join(cache_path, "host_kernel.cu")
        host_source = self._read_text_file(host_path)

        result = {}
        if device_source is not None:
            result["device_kernel_source"] = device_source
        if host_source is not None:
            result["host_kernel_source"] = host_source

        # Add library path
        lib_path = os.path.join(cache_path, "kernel_lib.so")
        if os.path.exists(lib_path):
            result["kernel_lib_path"] = lib_path
        else:
            return None

        return result

    def get_required_files(self) -> list[str]:
        """Get required files for Torch cache.

        Returns:
            List of required file paths
        """
        return ["kernel_lib.so"]

    def _get_device_extension(self, kernel: JITKernel) -> str:
        """Get device kernel file extension based on target.

        Args:
            kernel: JITKernel instance

        Returns:
            File extension (e.g., 'cu', 'metal', 'cpp')
        """
        # Check if adapter has target information
        if hasattr(kernel.adapter, "target"):
            target_str = str(kernel.adapter.target).lower()
            if "metal" in target_str:
                return "metal"
            elif "cuda" in target_str:
                return "cu"
            elif "hip" in target_str:
                return "cpp"  # HIP typically uses .cpp files
            elif "cpu" in target_str or "llvm" in target_str:
                return "cpp"

        # Default to CUDA
        return "cu"
