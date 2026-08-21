# Tenstorrent frontend intrinsic registrations are hardware-independent.
file(GLOB TILE_LANG_TENSTORRENT_SRCS
  src/tenstorrent/op/*.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_TENSTORRENT_SRCS})
