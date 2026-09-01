"""Memory-only cache policy for TT-Lang Python source artifacts."""

from __future__ import annotations

from tilelang.cache.kernel_cache import KernelCache


class TTNNKernelCache(KernelCache):
    """Keep process-local hits while avoiding an unsupported binary contract."""

    def _load_kernel_from_disk(self, *args, **kwargs):
        return None

    def _save_kernel_to_disk(self, *args, **kwargs):
        return None
