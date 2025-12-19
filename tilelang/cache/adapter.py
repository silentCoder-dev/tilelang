"""Cache adapter interface for extensible kernel caching system.

This module provides a pluggable architecture for kernel cache adapters,
allowing different execution backends to define their own caching logic
without modifying the core KernelCache implementation.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from tilelang.jit import JITKernel

# Type alias for content that can be string, bytes, or any other object
from typing import Any


@dataclass
class CacheFileInfo:
    """Information about a file to be cached.

    Attributes:
        path: Relative path within the cache directory
        content: File content (string, bytes, or special object like Executable)
        mode: File write mode ('w' for text, 'wb' for binary)
        is_required: Whether this file is required for loading from cache
        description: Optional description of the file's purpose
        is_executable: Whether content is a TVM Executable that needs special handling
    """

    path: str
    content: str | bytes | Any
    mode: str  # 'w' or 'wb'
    is_required: bool = True
    description: str | None = None
    is_executable: bool = False


class CacheAdapter(ABC):
    """Base class for backend-specific cache adapters.

    Each execution backend should implement this interface to define
    how its kernels are serialized to and deserialized from cache.
    """

    @abstractmethod
    def get_cache_files(self, kernel: JITKernel) -> dict[str, CacheFileInfo]:
        """Get information about files to cache for a kernel.

        Args:
            kernel: The JITKernel instance to cache

        Returns:
            Dictionary mapping file paths to CacheFileInfo objects
        """
        pass

    @abstractmethod
    def load_from_cache(self, cache_path: str) -> dict[str, Any] | None:
        """Load kernel data from cache directory.

        Args:
            cache_path: Path to the cache directory

        Returns:
            Dictionary of loaded data, or None if loading failed
        """
        pass

    @abstractmethod
    def get_required_files(self) -> list[str]:
        """Get list of required file paths for cache validation.

        Returns:
            List of relative file paths that must exist for cache to be valid
        """
        pass

    def validate_cache(self, cache_path: str) -> bool:
        """Validate that cache directory contains all required files.

        Args:
            cache_path: Path to the cache directory

        Returns:
            True if cache is valid, False otherwise
        """
        required_files = self.get_required_files()
        return all(os.path.exists(os.path.join(cache_path, f)) for f in required_files)


class CacheAdapterRegistry:
    """Registry for cache adapters.

    Provides a central registry for backend cache adapters,
    allowing dynamic registration and retrieval.
    """

    _adapters: dict[str, type[CacheAdapter]] = {}

    @classmethod
    def register(cls, backend_name: str) -> callable:
        """Decorator to register a cache adapter for a backend.

        Args:
            backend_name: Name of the execution backend

        Returns:
            Decorator function
        """

        def decorator(adapter_cls: type[CacheAdapter]) -> type[CacheAdapter]:
            if backend_name in cls._adapters:
                raise ValueError(f"Cache adapter already registered for backend: {backend_name}")
            cls._adapters[backend_name] = adapter_cls
            return adapter_cls

        return decorator

    @classmethod
    def get_adapter(cls, backend_name: str) -> CacheAdapter:
        """Get cache adapter instance for a backend.

        Args:
            backend_name: Name of the execution backend

        Returns:
            CacheAdapter instance

        Raises:
            ValueError: If no adapter is registered for the backend
        """
        if backend_name not in cls._adapters:
            # Try to import default adapters
            cls._import_default_adapters()

        if backend_name not in cls._adapters:
            raise ValueError(f"No cache adapter registered for backend: {backend_name}. Available backends: {list(cls._adapters.keys())}")

        return cls._adapters[backend_name]()

    @classmethod
    def has_adapter(cls, backend_name: str) -> bool:
        """Check if an adapter is registered for a backend.

        Args:
            backend_name: Name of the execution backend

        Returns:
            True if adapter exists, False otherwise
        """
        if not cls._adapters:
            cls._import_default_adapters()
        return backend_name in cls._adapters

    @classmethod
    def list_backends(cls) -> list[str]:
        """List all registered backend names.

        Returns:
            List of backend names
        """
        # Ensure default adapters are imported
        if not cls._adapters:
            cls._import_default_adapters()
        return list(cls._adapters.keys())

    @classmethod
    def _import_default_adapters(cls) -> None:
        """Import default cache adapters.

        This method imports adapters for built-in backends to ensure
        they are available when needed.
        """
        # Import each adapter module to trigger decorator registration
        adapter_modules = [
            "tvm_ffi_adapter",
            "nvrtc_adapter",
            "cutedsl_adapter",
            "ctypes_adapter",
            "cython_adapter",
            "torch_adapter",
        ]

        for module_name in adapter_modules:
            try:
                __import__(f"tilelang.cache.adapters.{module_name}", fromlist=[""])
                # The decorator should have registered the adapter when the module was imported
            except ImportError:
                # Adapter module not found, skip it
                continue
