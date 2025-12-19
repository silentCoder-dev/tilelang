"""Cache adapter for CuTeDSL backend."""

from __future__ import annotations

import os
from typing import Any

from tilelang.cache.adapter import CacheAdapterRegistry, CacheFileInfo
from tilelang.cache.adapters.base import BaseCacheAdapter
from tilelang.jit import JITKernel


@CacheAdapterRegistry.register("cutedsl")
class CuTeDSLCacheAdapter(BaseCacheAdapter):
    """Cache adapter for CuTeDSL execution backend."""

    def get_cache_files(self, kernel: JITKernel) -> dict[str, CacheFileInfo]:
        """Get files to cache for CuTeDSL kernel.

        CuTeDSL backend caches:
        - kernel.py: Python module with kernel implementation
        - launcher_lib.so: C++ launcher library
        - launcher.cpp: C++ launcher source code (for debugging)
        """
        files = {}

        # Python kernel module
        if hasattr(kernel.adapter, "get_kernel_source"):
            kernel_source = kernel.adapter.get_kernel_source()
            if kernel_source:
                files["kernel.py"] = CacheFileInfo(
                    path="kernel.py", content=kernel_source, mode="w", is_required=True, description="Python kernel module"
                )

        # C++ launcher library
        lib_gen = getattr(kernel.adapter, "lib_generator", None)
        if lib_gen and hasattr(lib_gen, "launcher_libpath") and lib_gen.launcher_libpath:
            launcher_libpath = lib_gen.launcher_libpath
            if os.path.exists(launcher_libpath):
                try:
                    with open(launcher_libpath, "rb") as f:
                        lib_content = f.read()
                    files["launcher_lib.so"] = CacheFileInfo(
                        path="launcher_lib.so", content=lib_content, mode="wb", is_required=True, description="C++ launcher library"
                    )
                except OSError:
                    pass

        # C++ launcher source code (for debugging)
        if hasattr(kernel.adapter, "launcher_cpp_code") and kernel.adapter.launcher_cpp_code:
            files["launcher.cpp"] = CacheFileInfo(
                path="launcher.cpp",
                content=kernel.adapter.launcher_cpp_code,
                mode="w",
                is_required=False,
                description="C++ launcher source code (debugging)",
            )

        return files

    def load_from_cache(self, cache_path: str) -> dict[str, Any] | None:
        """Load CuTeDSL kernel data from cache.

        Args:
            cache_path: Path to cache directory

        Returns:
            Dictionary with loaded data:
            - kernel_lib_path: Path to Python kernel module
            - launcher_lib_path: Path to C++ launcher library
            - launcher_cpp_code: C++ launcher source code (optional)
        """
        result = {}

        # Python kernel module path
        kernel_py_path = os.path.join(cache_path, "kernel.py")
        if os.path.exists(kernel_py_path):
            result["kernel_lib_path"] = kernel_py_path
        else:
            return None

        # C++ launcher library path
        launcher_lib_path = os.path.join(cache_path, "launcher_lib.so")
        if os.path.exists(launcher_lib_path):
            result["launcher_lib_path"] = launcher_lib_path
        else:
            return None

        # C++ launcher source code (optional)
        launcher_cpp_path = os.path.join(cache_path, "launcher.cpp")
        launcher_cpp_code = self._read_text_file(launcher_cpp_path)
        if launcher_cpp_code is not None:
            result["launcher_cpp_code"] = launcher_cpp_code

        # For CuTeDSL, we don't load device/host kernel sources from cache
        # as they are embedded in the Python module
        result["device_kernel_source"] = ""
        result["host_kernel_source"] = ""

        return result

    def get_required_files(self) -> list[str]:
        """Get required files for CuTeDSL cache.

        Returns:
            List of required file paths
        """
        return ["kernel.py", "launcher_lib.so"]
