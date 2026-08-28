"""Tenstorrent-specific transformation frontends."""

from .. import _ffi_api


def ValidateTenstorrentKernelLaunch():
    """Validate the static 2-D Core grid and unit thread dimensions."""

    return _ffi_api.ValidateTenstorrentKernelLaunch()  # type: ignore


def LowerTenstorrentFrontendAnnotations():
    """Lower topology forms and stage per-buffer allocation metadata."""

    return _ffi_api.LowerTenstorrentFrontendAnnotations()  # type: ignore


def LowerTenstorrentBufferAllocations():
    """Attach staged metadata to AllocBuffer nodes by Var identity."""

    return _ffi_api.LowerTenstorrentBufferAllocations()  # type: ignore


__all__ = (
    "LowerTenstorrentBufferAllocations",
    "LowerTenstorrentFrontendAnnotations",
    "ValidateTenstorrentKernelLaunch",
)
