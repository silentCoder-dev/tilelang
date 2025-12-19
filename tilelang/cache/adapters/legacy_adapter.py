"""Legacy cache adapter for backward compatibility.

This adapter provides backward compatibility with the old hardcoded
cache logic until all backends are migrated to the new adapter system.
"""

from __future__ import annotations

from typing import Any

from tilelang.cache.adapter import CacheAdapterRegistry, CacheFileInfo
from tilelang.cache.adapters.base import BaseCacheAdapter
from tilelang.jit import JITKernel


@CacheAdapterRegistry.register("legacy")
class LegacyCacheAdapter(BaseCacheAdapter):
    """Legacy cache adapter that mimics the old hardcoded logic.

    This adapter is used as a fallback for backends that haven't been
    migrated to the new adapter system yet.
    """

    def get_cache_files(self, kernel: JITKernel) -> dict[str, CacheFileInfo]:
        """Get files to cache using legacy logic.

        This mimics the old hardcoded logic in KernelCache._save_kernel_to_disk.
        """
        # This adapter doesn't actually save files - it's just a placeholder
        # for backward compatibility. The actual saving is done by the
        # legacy code path in KernelCache.
        return {}

    def load_from_cache(self, cache_path: str) -> dict[str, Any] | None:
        """Load kernel data using legacy logic.

        This mimics the old hardcoded logic in KernelCache._load_kernel_from_disk.
        """
        # This adapter doesn't actually load files - it's just a placeholder
        # for backward compatibility. The actual loading is done by the
        # legacy code path in KernelCache.
        return None

    def get_required_files(self) -> list[str]:
        """Get required files using legacy logic.

        Returns an empty list to indicate that this adapter doesn't
        handle cache validation.
        """
        return []

    def validate_cache(self, cache_path: str) -> bool:
        """Always return True for legacy adapter.

        Legacy validation is handled by the old code path.
        """
        return True
