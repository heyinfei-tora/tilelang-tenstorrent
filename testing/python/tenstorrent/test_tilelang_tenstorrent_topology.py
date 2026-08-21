import json

import pytest
from tvm import tirx
from tvm.ir import Op

from tilelang.tenstorrent import language as T


def _collect_calls_and_foreach_loops(func):
    calls = []
    loops = []

    def visit(node):
        if isinstance(node, tirx.Call) and isinstance(node.op, Op) and node.op.name.startswith("tl.tt."):
            calls.append(node)
        if isinstance(node, tirx.For) and ("tl.tt.foreach_src" in node.annotations or "tl.tt.foreach_dst" in node.annotations):
            loops.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return calls, loops


def test_topology_primitives_construct_ordered_ir():
    gather = T.tt.PipeNet(
        [
            T.tt.Pipe(src=(1, 0), dst=(0, 0)),
            T.tt.Pipe(src=(2, 0), dst=(0, 0)),
            T.tt.Pipe(src=(2, 0), dst=(0, 0)),
        ]
    )

    @T.prim_func
    def main():
        with T.Kernel(4, 1, threads=1):
            if T.tt.is_src(gather):
                T.evaluate(0)
            if T.tt.is_dst(gather):
                T.evaluate(0)
            if T.tt.is_active(gather):
                T.evaluate(0)
            for pipe in T.tt.foreach_src(gather):
                src_x, src_y = T.tt.pipe_src(pipe)
                dst_x, dst_y = T.tt.pipe_dst(pipe)
                T.evaluate(src_x + src_y + dst_x + dst_y)
            for pipe in T.tt.foreach_dst(gather):
                src_x, _ = T.tt.pipe_src(pipe)
                T.evaluate(src_x)

    calls, loops = _collect_calls_and_foreach_loops(main)
    assert [call.op.name for call in calls].count("tl.tt.is_src") == 1
    assert [call.op.name for call in calls].count("tl.tt.is_dst") == 1
    assert [call.op.name for call in calls].count("tl.tt.is_active") == 1
    assert [call.op.name for call in calls].count("tl.tt.pipe_src") == 4
    assert [call.op.name for call in calls].count("tl.tt.pipe_dst") == 2
    assert len(loops) == 2

    descriptor = json.loads(str(loops[0].annotations["tl.tt.foreach_src"]))
    assert descriptor["kind"] == "point_to_point"
    assert descriptor["pipes"][1] == descriptor["pipes"][2]
    assert [pipe["src"] for pipe in descriptor["pipes"]] == [[1, 0], [2, 0], [2, 0]]


def test_collective_range_accessor_constructs_ir():
    broadcast = T.tt.PipeNet([T.tt.Pipe(src=(0, 0), dst=T.tt.CoreRange(begin=(1, 0), end=(4, 1)))])

    @T.prim_func
    def main():
        with T.Kernel(4, 1, threads=1):
            for pipe in T.tt.foreach_src(broadcast):
                begin, end = T.tt.pipe_dst_range(pipe)
                T.evaluate(begin[0] + begin[1] + end[0] + end[1])

    calls, loops = _collect_calls_and_foreach_loops(main)
    assert [call.op.name for call in calls].count("tl.tt.pipe_dst_range") == 4
    descriptor = json.loads(str(loops[0].annotations["tl.tt.foreach_src"]))
    assert descriptor["kind"] == "collective"
    assert descriptor["pipes"][0]["dst"] == {"begin": [1, 0], "end": [4, 1]}


@pytest.mark.parametrize(
    "factory, error, match",
    [
        (lambda: T.tt.CoreRange((0, 0), (0, 1)), ValueError, "non-empty"),
        (lambda: T.tt.Pipe(src=(0, 0), dst=(True, 1)), TypeError, "compile-time integers"),
        (lambda: T.tt.PipeNet([]), ValueError, "at least one"),
        (
            lambda: T.tt.PipeNet(
                [
                    T.tt.Pipe(src=(0, 0), dst=(1, 0)),
                    T.tt.Pipe(src=(0, 0), dst=T.tt.CoreRange((1, 0), (2, 1))),
                ]
            ),
            ValueError,
            "cannot mix",
        ),
    ],
)
def test_invalid_topologies_fail_at_construction(factory, error, match):
    with pytest.raises(error, match=match):
        factory()


def test_topology_must_fit_static_2d_kernel_grid():
    outside = T.tt.PipeNet([T.tt.Pipe(src=(0, 0), dst=(2, 0))])

    with pytest.raises(ValueError, match="outside T.Kernel grid"):

        @T.prim_func
        def outside_grid():
            with T.Kernel(2, 1, threads=1):
                T.evaluate(T.tt.is_dst(outside))

    with pytest.raises(ValueError, match="require a 2-D"):

        @T.prim_func
        def one_dimensional_grid():
            with T.Kernel(2, threads=1):
                T.evaluate(T.tt.is_dst(outside))


def test_selected_pipe_accessors_validate_region_and_contract():
    point_to_point = T.tt.PipeNet([T.tt.Pipe(src=(0, 0), dst=(1, 0))])
    collective = T.tt.PipeNet([T.tt.Pipe(src=(0, 0), dst=T.tt.CoreRange(begin=(1, 0), end=(2, 1)))])

    with pytest.raises(ValueError, match="requires a collective"):

        @T.prim_func
        def wrong_contract():
            with T.Kernel(2, 1, threads=1):
                for pipe in T.tt.foreach_src(point_to_point):
                    T.tt.pipe_dst_range(pipe)

    with pytest.raises(ValueError, match="requires a point-to-point"):

        @T.prim_func
        def other_wrong_contract():
            with T.Kernel(2, 1, threads=1):
                for pipe in T.tt.foreach_src(collective):
                    T.tt.pipe_dst(pipe)
