"""Experimental source-only codegen for the first Tenstorrent TTL subset."""

from __future__ import annotations

import keyword
from dataclasses import dataclass

from tvm import IRModule, ir, tirx
from tvm.runtime import _ffi_api as _runtime_ffi_api
from tvm.target import Target


_DEVICE_KERNEL_LAUNCH = int(ir.CallingConv.DEVICE_KERNEL_LAUNCH)
_THREAD_TAGS = ("blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y", "threadIdx.z")
_SUPPORTED_ARCHES = {"wormhole_b0", "blackhole"}


def _unsupported(detail: str) -> NotImplementedError:
    return NotImplementedError(f"Tenstorrent TTL source v1 does not support {detail}")


def _as_static_int(value, detail: str) -> int:
    if not isinstance(value, tirx.IntImm):
        raise _unsupported(f"dynamic {detail}: {value}")
    return int(value)


@dataclass(frozen=True)
class _Transfer:
    kind: str
    core: tuple[int, int]
    peer: tuple[int, int]
    extent: tuple[int, int]
    buffer_data: tirx.Var


class _TTLSourceEmitter:
    """Analyze a deliberately small lowered TIR subset and emit TT-Lang Python."""

    def __init__(self, mod: IRModule, target: Target) -> None:
        self.mod = mod
        self.target = target
        self.block_vars: dict[str, tirx.Var] = {}
        self.block_extents: dict[str, int] = {}
        self.allocations: list[tirx.AllocBuffer] = []
        self.transfers: list[_Transfer] = []

    def emit(self) -> tuple[str, str]:
        entrypoint, func = self._device_function()
        arch = str(self.target.attrs.get("arch", ""))
        if arch not in _SUPPORTED_ARCHES:
            raise ValueError(f"Unsupported Tenstorrent architecture {arch!r}")
        if not entrypoint.isidentifier() or keyword.iskeyword(entrypoint):
            raise ValueError(f"TTL entrypoint {entrypoint!r} is not a valid Python identifier")
        if func.params or func.buffer_map:
            raise _unsupported("runtime parameters or global tensor access")

        grid = self._validate_thread_extent(func)
        self._visit_stmt(func.body, {})
        allocation, block_count = self._validate_allocation()
        source, destination, extent = self._validate_transfer(allocation)
        source_text = self._render(
            entrypoint=entrypoint,
            arch=arch,
            grid=grid,
            block_count=block_count,
            ttl_shape=tuple(value // 32 for value in extent),
            source=source,
            destination=destination,
        )
        return entrypoint, source_text

    def _device_function(self) -> tuple[str, tirx.PrimFunc]:
        candidates: list[tuple[str, tirx.PrimFunc]] = []
        for global_var, base_func in self.mod.functions.items():
            if not isinstance(base_func, tirx.PrimFunc):
                continue
            calling_conv = base_func.attrs.get("calling_conv") if base_func.attrs else None
            if calling_conv is not None and int(calling_conv) == _DEVICE_KERNEL_LAUNCH:
                candidates.append((global_var.name_hint, base_func))
        if len(candidates) != 1:
            raise ValueError(
                "Tenstorrent TTL source v1 requires exactly one DEVICE_KERNEL_LAUNCH PrimFunc, "
                f"but found {len(candidates)}"
            )
        return candidates[0]

    def _validate_thread_extent(self, func: tirx.PrimFunc) -> tuple[int, int]:
        extents = func.attrs.get("thread_extent") if func.attrs else None
        if extents is None:
            raise ValueError("Tenstorrent TTL source v1 requires static thread_extent metadata")
        values = {tag: _as_static_int(extents[tag], f"thread extent {tag}") for tag in _THREAD_TAGS if tag in extents}
        if set(values) != set(_THREAD_TAGS):
            missing = sorted(set(_THREAD_TAGS) - set(values))
            raise ValueError(f"Tenstorrent TTL source v1 is missing thread extents: {', '.join(missing)}")
        for tag in ("threadIdx.x", "threadIdx.y", "threadIdx.z"):
            if values[tag] != 1:
                raise _unsupported(f"{tag} extent {values[tag]}; all thread extents must be 1")
        grid = (values["blockIdx.x"], values["blockIdx.y"])
        if grid[0] <= 0 or grid[1] <= 0:
            raise ValueError(f"Tenstorrent TTL source v1 requires a non-empty 2-D grid, got {grid}")
        return grid

    def _visit_stmt(self, stmt: tirx.Stmt, guards: dict[str, int]) -> None:
        if isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                self._visit_stmt(child, guards)
            return
        if isinstance(stmt, tirx.AttrStmt):
            if stmt.attr_key != "thread_extent" or not isinstance(stmt.node, tirx.IterVar):
                raise _unsupported(f"statement {type(stmt).__name__} with attribute {stmt.attr_key!r}")
            tag = str(stmt.node.thread_tag)
            if tag not in _THREAD_TAGS:
                raise _unsupported(f"launch thread {tag!r}")
            extent = _as_static_int(stmt.value, f"launch extent {tag}")
            self.block_vars[tag] = stmt.node.var
            self.block_extents[tag] = extent
            self._visit_stmt(stmt.body, guards)
            return
        if isinstance(stmt, tirx.AllocBuffer):
            self.allocations.append(stmt)
            return
        if isinstance(stmt, tirx.DeclBuffer):
            return
        if isinstance(stmt, tirx.IfThenElse):
            if stmt.else_case is not None:
                raise _unsupported("an else branch in a core guard")
            next_guards = dict(guards)
            for tag, value in self._parse_core_guard(stmt.condition).items():
                previous = next_guards.get(tag)
                if previous is not None and previous != value:
                    raise ValueError(f"Contradictory Tenstorrent core guard for {tag}")
                next_guards[tag] = value
            self._visit_stmt(stmt.then_case, next_guards)
            return
        if isinstance(stmt, tirx.Evaluate):
            if isinstance(stmt.value, tirx.IntImm) and int(stmt.value) == 0:
                return
            if not isinstance(stmt.value, tirx.Call) or not isinstance(stmt.value.op, ir.Op):
                raise _unsupported(f"evaluate expression {stmt.value}")
            self._record_transfer(stmt.value, guards)
            return
        if isinstance(stmt, tirx.BufferStore):
            raise _unsupported("global or shared buffer stores")
        raise _unsupported(f"statement node {type(stmt).__name__}: {stmt}")

    def _parse_core_guard(self, condition: tirx.PrimExpr) -> dict[str, int]:
        if isinstance(condition, tirx.And):
            result = self._parse_core_guard(condition.a)
            for tag, value in self._parse_core_guard(condition.b).items():
                if tag in result and result[tag] != value:
                    raise ValueError(f"Contradictory Tenstorrent core guard for {tag}")
                result[tag] = value
            return result
        if not isinstance(condition, tirx.EQ):
            raise _unsupported(f"non-equality core guard {condition}")
        variable, value = condition.a, condition.b
        if isinstance(value, tirx.Var) and isinstance(variable, tirx.IntImm):
            variable, value = value, variable
        if not isinstance(variable, tirx.Var):
            raise _unsupported(f"core guard {condition}")
        for tag in ("blockIdx.x", "blockIdx.y"):
            block_var = self.block_vars.get(tag)
            if block_var is not None and variable.same_as(block_var):
                return {tag: _as_static_int(value, f"core guard for {tag}")}
        raise _unsupported(f"guard on non-Core variable {variable}")

    def _guard_core(self, guards: dict[str, int]) -> tuple[int, int]:
        coordinates = []
        for tag in ("blockIdx.x", "blockIdx.y"):
            if tag in guards:
                coordinate = guards[tag]
                extent = self.block_extents[tag]
                if not 0 <= coordinate < extent:
                    raise ValueError(
                        f"Tenstorrent core coordinate {coordinate} is outside {tag} extent {extent}"
                    )
                coordinates.append(coordinate)
                continue
            block_var = self.block_vars.get(tag)
            if block_var is None:
                raise ValueError(f"Tenstorrent TTL source v1 did not find launch variable {tag}")
            extent = self.block_extents[tag]
            if extent != 1:
                raise _unsupported(f"a transfer not guarded by {tag}")
            coordinates.append(0)
        return coordinates[0], coordinates[1]

    def _record_transfer(self, call: tirx.Call, guards: dict[str, int]) -> None:
        op_name = call.op.name
        if op_name not in {"tl.tt.noc_send", "tl.tt.noc_recv"}:
            raise _unsupported(f"call {op_name}")
        expected_args = 5 if op_name == "tl.tt.noc_send" else 3
        if len(call.args) != expected_args:
            raise ValueError(f"Malformed {op_name}: expected {expected_args} arguments")
        extent, buffer_data = self._parse_region(call.args[0], 1 if op_name == "tl.tt.noc_send" else 2)
        core = self._guard_core(guards)
        if op_name == "tl.tt.noc_send":
            begin = (_as_static_int(call.args[1], "destination x"), _as_static_int(call.args[2], "destination y"))
            end = (
                _as_static_int(call.args[3], "destination end x"),
                _as_static_int(call.args[4], "destination end y"),
            )
            if end != (begin[0] + 1, begin[1] + 1):
                raise _unsupported(f"collective destination range [{begin}, {end})")
            peer, kind = begin, "send"
        else:
            peer = (_as_static_int(call.args[1], "source x"), _as_static_int(call.args[2], "source y"))
            kind = "recv"
        self.transfers.append(
            _Transfer(kind=kind, core=core, peer=peer, extent=extent, buffer_data=buffer_data)
        )

    def _parse_region(self, region: tirx.PrimExpr, expected_access: int) -> tuple[tuple[int, int], tirx.Var]:
        if (
            not isinstance(region, tirx.Call)
            or not isinstance(region.op, ir.Op)
            or region.op.name != "tl.tileop.region"
        ):
            raise ValueError(f"Malformed Tenstorrent transfer region: {region}")
        if len(region.args) != 4:
            raise _unsupported(f"a non-2-D transfer region {region}")
        load = region.args[0]
        if not isinstance(load, tirx.BufferLoad):
            raise ValueError(f"Tenstorrent transfer region must start from a BufferLoad, got {load}")
        if len(load.indices) != 1 or _as_static_int(load.indices[0], "transfer offset") != 0:
            raise _unsupported(f"a partial transfer starting at {load.indices}")
        access = _as_static_int(region.args[1], "region access type")
        if access != expected_access:
            raise ValueError(f"Tenstorrent transfer region access type must be {expected_access}, got {access}")
        extent = (
            _as_static_int(region.args[2], "region row extent"),
            _as_static_int(region.args[3], "region column extent"),
        )
        return extent, load.buffer.data

    def _validate_allocation(self) -> tuple[tirx.AllocBuffer, int]:
        if len(self.allocations) != 1:
            raise _unsupported(f"{len(self.allocations)} shared DFB allocations; exactly one is required")
        allocation = self.allocations[0]
        buffer = allocation.buffer
        if buffer.scope() != "shared.dyn":
            raise _unsupported(f"DFB storage scope {buffer.scope()!r}")
        if str(buffer.dtype) != "bfloat16":
            raise _unsupported(f"DFB dtype {buffer.dtype}; only bfloat16 is supported")
        annotations = allocation.annotations
        tile_shape = annotations.get("tt.tile_shape")
        if tile_shape is None or tuple(_as_static_int(value, "tt.tile_shape") for value in tile_shape) != (32, 32):
            raise _unsupported(f"tt.tile_shape {tile_shape}; only (32, 32) is supported")
        block_count_value = annotations.get("tt.dfb_block_count")
        if block_count_value is None:
            raise ValueError("Tenstorrent TTL source v1 requires tt.dfb_block_count metadata")
        block_count = _as_static_int(block_count_value, "tt.dfb_block_count")
        if not 1 <= block_count <= 32:
            raise ValueError(f"tt.dfb_block_count must be in [1, 32], got {block_count}")
        return allocation, block_count

    def _validate_transfer(
        self, allocation: tirx.AllocBuffer
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        sends = [transfer for transfer in self.transfers if transfer.kind == "send"]
        receives = [transfer for transfer in self.transfers if transfer.kind == "recv"]
        if len(sends) != 1 or len(receives) != 1:
            raise _unsupported(
                f"{len(sends)} sends and {len(receives)} receives; exactly one matched pair is required"
            )
        send, receive = sends[0], receives[0]
        if send.core != receive.peer or send.peer != receive.core:
            raise ValueError(f"Unmatched Tenstorrent transfer endpoints: send={send}, receive={receive}")
        if send.extent != receive.extent:
            raise ValueError(f"Mismatched Tenstorrent transfer regions: send={send.extent}, receive={receive.extent}")
        allocation_data = allocation.buffer.data
        if not send.buffer_data.same_as(allocation_data) or not receive.buffer_data.same_as(allocation_data):
            raise ValueError("Tenstorrent transfer region does not reference the declared DFB allocation")
        extent = send.extent
        if any(value <= 0 or value % 32 for value in extent):
            raise _unsupported(f"transfer extent {extent}; each dimension must be a positive multiple of 32")
        allocation_elements = 1
        for value in allocation.buffer.shape:
            allocation_elements *= _as_static_int(value, "DFB allocation extent")
        if allocation_elements != extent[0] * extent[1]:
            raise _unsupported(f"partial DFB region {extent} for a {allocation_elements}-element allocation")
        return send.core, send.peer, extent

    @staticmethod
    def _render(
        *,
        entrypoint: str,
        arch: str,
        grid: tuple[int, int],
        block_count: int,
        ttl_shape: tuple[int, int],
        source: tuple[int, int],
        destination: tuple[int, int],
    ) -> str:
        return f'''from __future__ import annotations

import torch
import ttl
import ttnn

__tilelang_ttl_artifact__ = {{
    "version": 1,
    "entrypoint": {entrypoint!r},
    "target_arch": {arch!r},
    "grid": {grid!r},
}}


class _TileLangBFloat16Template:
    dtype = torch.bfloat16


@ttl.operation(grid={grid!r})
def {entrypoint}():
    tilelang_dfb = ttl.make_dataflow_buffer_like(
        _TileLangBFloat16Template(), shape={ttl_shape!r}, block_count={block_count}
    )
    tilelang_pipe = ttl.PipeNet([
        ttl.Pipe(src={source!r}, dst={destination!r}),
    ])

    @ttl.datamovement()
    def tilelang_transfer():
        def tilelang_send(pipe):
            with tilelang_dfb.wait() as block:
                ttl.copy(block, pipe).wait()

        tilelang_pipe.if_src(tilelang_send)

        def tilelang_recv(pipe):
            with tilelang_dfb.reserve() as block:
                ttl.copy(pipe, block).wait()

        tilelang_pipe.if_dst(tilelang_recv)
'''


def build_ttl_without_compile(mod: IRModule, target: Target):
    """Return an importable TT-Lang Python source module without loading TTNN."""

    entrypoint, source = _TTLSourceEmitter(mod, target).emit()
    return _runtime_ffi_api.CSourceModuleCreate(source, "py", [entrypoint], None)
