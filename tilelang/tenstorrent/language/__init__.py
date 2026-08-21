"""Tenstorrent language dialect: common TileLang plus TT topology primitives."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL

from . import tt as tt

__tilelang_dialect__ = "tenstorrent"
__all__ = tuple(dict.fromkeys((*_COMMON_ALL, "tt")))

del _COMMON_ALL
