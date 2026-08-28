from __future__ import annotations

import pytest

import tilelang
from tilelang import tvm
from tvm import tirx
from tvm.ir import Op
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody
from tilelang.tenstorrent.transform import (
    LowerTenstorrentBufferAllocations,
    LowerTenstorrentFrontendAnnotations,
    ValidateTenstorrentKernelLaunch,
)


_ALLOC_BUFFER_ANNOTATIONS = "tl.alloc_buffer_annotations"
_ALLOC_BUFFER_METADATA = "tl.tt.alloc_buffer_metadata"
_FRONTEND_TOPOLOGY_OPS = {
    "tl.tt.is_src",
    "tl.tt.is_dst",
    "tl.tt.is_active",
    "tl.tt.pipe_src",
    "tl.tt.pipe_dst",
    "tl.tt.pipe_dst_range",
    "tl.tt.pipe_send",
    "tl.tt.pipe_recv",
}
_FOREACH_ANNOTATIONS = {"tl.tt.foreach_src", "tl.tt.foreach_dst"}


def _module(func: tirx.PrimFunc) -> tvm.IRModule:
    return tvm.IRModule({"main": func.with_attr("global_symbol", "main")})


def _collect(func: tirx.PrimFunc, node_type: type) -> list:
    nodes = []

    def visit(node):
        if isinstance(node, node_type):
            nodes.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return nodes


def _op_calls(func: tirx.PrimFunc) -> list[tirx.Call]:
    return [
        call
        for call in _collect(func, tirx.Call)
        if isinstance(call.op, Op)
    ]


def _prepare_frontend(func: tirx.PrimFunc) -> tvm.IRModule:
    mod = ValidateTenstorrentKernelLaunch()(_module(func))
    return tilelang.transform.MaterializeKernelLaunch()(mod)


def _lower_frontend(func: tirx.PrimFunc) -> tvm.IRModule:
    return LowerTenstorrentFrontendAnnotations()(_prepare_frontend(func))


def _empty_kernel(blocks=(2, 1), threads=1):
    @T.prim_func
    def main():
        with T.Kernel(*blocks, threads=threads):
            T.evaluate(0)

    return main


def _buffer_kernel(annotations, shape=(64, 128)):
    @T.prim_func
    def main(A: T.Tensor(shape, T.bfloat16)):
        with T.Kernel(1, 1, threads=1):
            shared = T.alloc_shared(shape, T.bfloat16, annotations=annotations)
            if len(shape) == 1:
                A[0] = shared[0]
            else:
                A[0, 0] = shared[0, 0]

    return main


def test_validate_kernel_launch_accepts_static_2d_single_thread_grid():
    before = _module(_empty_kernel(blocks=(4, 2), threads=(1, 1, 1)))
    after = ValidateTenstorrentKernelLaunch()(before)

    assert tvm.ir.structural_equal(after, before)


@pytest.mark.parametrize("blocks", [(4,), (4, 2, 1)])
def test_validate_kernel_launch_rejects_non_2d_grid(blocks):
    with pytest.raises(ValueError, match="requires exactly a static 2-D Core grid"):
        ValidateTenstorrentKernelLaunch()(_module(_empty_kernel(blocks=blocks)))


def test_validate_kernel_launch_rejects_dynamic_grid():
    @T.prim_func
    def main(grid_x: T.int32):
        with T.Kernel(grid_x, 2, threads=1):
            T.evaluate(0)

    with pytest.raises(ValueError, match="requires exactly a static 2-D Core grid"):
        ValidateTenstorrentKernelLaunch()(_module(main))


@pytest.mark.parametrize("threads", [2, (1, 2), (1, 1, 2)])
def test_validate_kernel_launch_rejects_non_unit_thread_extents(threads):
    with pytest.raises(ValueError, match=r"requires threadIdx\.[xyz] extent to be 1"):
        ValidateTenstorrentKernelLaunch()(_module(_empty_kernel(threads=threads)))


def test_frontend_pass_consumes_topology_intrinsics_and_foreach_annotations():
    point_to_point = T.tt.PipeNet([T.tt.Pipe(src=(0, 0), dst=(1, 0))])
    collective = T.tt.PipeNet(
        [T.tt.Pipe(src=(0, 0), dst=T.tt.CoreRange(begin=(1, 0), end=(3, 2)))]
    )

    @T.prim_func
    def main():
        with T.Kernel(3, 2, threads=1):
            send_block = T.alloc_shared((32, 32), T.bfloat16)
            recv_block = T.alloc_shared((32, 32), T.bfloat16)
            if T.tt.is_src(point_to_point):
                T.evaluate(1)
            if T.tt.is_dst(point_to_point):
                T.evaluate(2)
            if T.tt.is_active(point_to_point):
                T.evaluate(3)
            for pipe in T.tt.foreach_src(point_to_point):
                src_x, src_y = T.tt.pipe_src(pipe)
                dst_x, dst_y = T.tt.pipe_dst(pipe)
                T.evaluate(src_x + src_y + dst_x + dst_y)
                T.copy(send_block, pipe)
            for pipe in T.tt.foreach_dst(point_to_point):
                T.copy(pipe, recv_block)
            for pipe in T.tt.foreach_src(collective):
                begin, end = T.tt.pipe_dst_range(pipe)
                T.evaluate(begin[0] + begin[1] + end[0] + end[1])

    lowered = _lower_frontend(main)["main"]
    calls = _op_calls(lowered)
    op_names = [call.op.name for call in calls]

    assert _FRONTEND_TOPOLOGY_OPS.isdisjoint(op_names)
    assert op_names.count("tl.tt.noc_send") == 1
    assert op_names.count("tl.tt.noc_recv") == 1
    assert all(
        _FOREACH_ANNOTATIONS.isdisjoint(loop.annotations)
        for loop in _collect(lowered, tirx.For)
    )

    noc_send = next(call for call in calls if call.op.name == "tl.tt.noc_send")
    noc_recv = next(call for call in calls if call.op.name == "tl.tt.noc_recv")
    assert [int(arg.value) for arg in noc_send.args[1:]] == [1, 0, 2, 1]
    assert [int(arg.value) for arg in noc_recv.args[1:]] == [0, 0]


def test_buffer_metadata_is_consumed_in_two_stages_for_multiple_buffers():
    @T.prim_func
    def main(A: T.Tensor((64, 128), T.bfloat16)):
        with T.Kernel(1, 1, threads=1):
            lhs = T.alloc_shared(
                (64, 128),
                T.bfloat16,
                annotations={"tt.dfb_block_count": 1},
            )
            rhs = T.alloc_shared(
                (64, 128),
                T.bfloat16,
                annotations={
                    "tt.dfb_block_count": 3,
                    "tt.tile_shape": (32, 32),
                },
            )
            A[0, 0] = lhs[0, 0] + rhs[0, 0]

    frontend = _lower_frontend(main)
    frontend_func = frontend["main"]
    assert _ALLOC_BUFFER_METADATA in frontend_func.attrs
    assert len(frontend_func.attrs[_ALLOC_BUFFER_METADATA]) == 2
    assert all(
        _ALLOC_BUFFER_ANNOTATIONS not in block.annotations
        for block in _collect(frontend_func, tirx.SBlock)
    )

    opaque = tilelang.transform.LowerOpaqueBlock()(frontend)
    lowered = LowerTenstorrentBufferAllocations()(opaque)["main"]
    assert _ALLOC_BUFFER_METADATA not in lowered.attrs

    allocations = [
        alloc
        for alloc in _collect(lowered, tirx.AllocBuffer)
        if "tt.dfb_block_count" in alloc.annotations
    ]
    assert len(allocations) == 2
    assert sorted(
        int(alloc.annotations["tt.dfb_block_count"].value)
        for alloc in allocations
    ) == [1, 3]
    assert all(
        [int(value.value) for value in alloc.annotations["tt.tile_shape"]]
        == [32, 32]
        for alloc in allocations
    )


def test_buffer_allocation_pass_rejects_dangling_metadata_key():
    dangling = tirx.Var("dangling", "handle")

    @T.prim_func
    def main():
        with T.Kernel(1, 1, threads=1):
            shared = T.alloc_shared((32, 32), T.bfloat16)
            T.evaluate(shared[0, 0])

    mod = tilelang.transform.LowerOpaqueBlock()(_lower_frontend(main))
    mod["main"] = mod["main"].with_attr(
        _ALLOC_BUFFER_METADATA,
        {
            dangling: {
                "tt.dfb_block_count": tirx.IntImm("int32", 1),
                "tt.tile_shape": [
                    tirx.IntImm("int32", 32),
                    tirx.IntImm("int32", 32),
                ],
            }
        },
    )
    with pytest.raises(ValueError, match="did not match an AllocBuffer"):
        LowerTenstorrentBufferAllocations()(mod)


def test_frontend_pass_rejects_unknown_allocation_annotation():
    with pytest.raises(ValueError, match="tt.unknown"):
        _lower_frontend(_buffer_kernel({"tt.unknown": 1}))


@pytest.mark.parametrize(
    "annotations",
    [
        {"tt.dfb_block_count": True},
        {"tt.dfb_block_count": "2"},
        {"tt.tile_shape": (32, "32")},
        {"tt.tile_shape": "32x32"},
    ],
)
def test_frontend_pass_rejects_invalid_allocation_metadata_types(annotations):
    with pytest.raises((TypeError, ValueError), match=r"tt\.(dfb_block_count|tile_shape)"):
        _lower_frontend(_buffer_kernel(annotations))


def test_frontend_pass_rejects_non_2d_buffer_metadata():
    with pytest.raises(ValueError, match="2-D"):
        _lower_frontend(
            _buffer_kernel({"tt.dfb_block_count": 1}, shape=(64,))
        )


def test_frontend_pass_rejects_shape_not_divisible_by_tile():
    with pytest.raises(ValueError, match="divisible"):
        _lower_frontend(
            _buffer_kernel({"tt.tile_shape": (32, 32)}, shape=(48, 128))
        )


@pytest.mark.parametrize("block_count", [0, 33])
def test_frontend_pass_rejects_out_of_range_dfb_block_count(block_count):
    with pytest.raises(ValueError, match="tt.dfb_block_count"):
        _lower_frontend(
            _buffer_kernel({"tt.dfb_block_count": block_count})
        )


@pytest.mark.parametrize("tile_shape", [(16, 32), (32,), (32, 0)])
def test_frontend_pass_rejects_unsupported_tile_shape(tile_shape):
    with pytest.raises((TypeError, ValueError), match="tt.tile_shape"):
        _lower_frontend(_buffer_kernel({"tt.tile_shape": tile_shape}))


def test_pipeline_does_not_leave_temporary_tenstorrent_annotations():
    kernel = _buffer_kernel(
        {
            "tt.dfb_block_count": 2,
            "tt.tile_shape": (32, 32),
        }
    )
    target = tilelang.tvm.target.Target(
        {"kind": "tenstorrent", "arch": "wormhole_b0"}
    )
    lowered = TenstorrentPassPipelineBody(_module(kernel), target)

    for func in lowered.functions.values():
        if not isinstance(func, tirx.PrimFunc):
            continue
        assert _ALLOC_BUFFER_METADATA not in func.attrs
        assert all(
            _ALLOC_BUFFER_ANNOTATIONS not in block.annotations
            for block in _collect(func, tirx.SBlock)
        )
        assert all(
            _FOREACH_ANNOTATIONS.isdisjoint(loop.annotations)
            for loop in _collect(func, tirx.For)
        )
