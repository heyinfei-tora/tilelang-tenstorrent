from __future__ import annotations

import ast
import importlib
from types import SimpleNamespace

import pytest

import tilelang
from tilelang import tvm
from tilelang.tenstorrent import execution_backend
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent.codegen import build_ttl_without_compile
from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody


def _target(arch="wormhole_b0"):
    return tvm.target.Target({"kind": "tenstorrent", "arch": arch, "host": "llvm"})


def _p2p_kernel(dtype=T.bfloat16, shape=(32, 32), block_count=2):
    net = T.tt.PipeNet([T.tt.Pipe(src=(0, 0), dst=(1, 0))])

    @T.prim_func
    def main():
        with T.Kernel(2, 1, threads=1):
            block = T.alloc_shared(
                shape,
                dtype,
                annotations={"tt.dfb_block_count": block_count},
            )
            for pipe in T.tt.foreach_src(net):
                T.copy(block, pipe)
            for pipe in T.tt.foreach_dst(net):
                T.copy(pipe, block)

    return main


def _device_module(func, target=None):
    target = target or _target()
    mod = tvm.IRModule({"main": func.with_attr("global_symbol", "main")})
    with tvm.transform.PassContext(opt_level=3), target:
        lowered = TenstorrentPassPipelineBody(mod, target)
    device_funcs = {
        global_var: base_func
        for global_var, base_func in lowered.functions.items()
        if isinstance(base_func, tvm.tirx.PrimFunc)
        and int(base_func.attrs.get("calling_conv", -1))
        == int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    }
    return tvm.IRModule(device_funcs)


def test_ttl_codegen_emits_deterministic_importable_point_to_point_source():
    target = _target()
    device_mod = _device_module(_p2p_kernel(shape=(64, 32), block_count=3), target)

    first = build_ttl_without_compile(device_mod, target).inspect_source()
    second = build_ttl_without_compile(device_mod, target).inspect_source()

    ast.parse(first)
    assert first == second
    assert '"version": 1' in first
    assert '"entrypoint": \'main_kernel\'' in first
    assert '"target_arch": \'wormhole_b0\'' in first
    assert '"grid": (2, 1)' in first
    assert "@ttl.operation(grid=(2, 1))" in first
    assert "shape=(2, 1), block_count=3" in first
    assert "ttl.Pipe(src=(0, 0), dst=(1, 0))" in first
    assert "with tilelang_dfb.wait() as block" in first
    assert "with tilelang_dfb.reserve() as block" in first
    assert first.count("ttl.copy(") == 2


def test_public_compile_reaches_injected_ttnn_launch(monkeypatch):
    loaded = []

    def load(source, module_name):
        loaded.append((source, module_name))
        return SimpleNamespace(
            __tilelang_ttl_artifact__={
                "version": 1,
                "entrypoint": "main_kernel",
                "target_arch": "wormhole_b0",
                "grid": (2, 1),
            },
            main_kernel=lambda: "launched",
        )

    adapter_module = importlib.import_module("tilelang.jit.adapter.ttnn.adapter")
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(adapter_module, "_load_ttl_module", load)

    kernel = tilelang.compile(
        _p2p_kernel(),
        target={"kind": "tenstorrent", "arch": "wormhole_b0"},
        execution_backend="ttnn",
    )

    assert kernel() == "launched"
    assert len(loaded) == 1
    ast.parse(loaded[0][0])


def test_ttl_codegen_rejects_unsupported_dtype():
    target = _target()
    device_mod = _device_module(_p2p_kernel(dtype=T.float32), target)

    with pytest.raises(NotImplementedError, match="only bfloat16"):
        build_ttl_without_compile(device_mod, target)


def test_ttl_codegen_rejects_runtime_parameters_and_global_access():
    @T.prim_func
    def main(A: T.Tensor((32, 32), T.bfloat16)):
        with T.Kernel(1, 1, threads=1):
            A[0, 0] = 0

    target = _target()
    device_mod = _device_module(main, target)

    with pytest.raises(NotImplementedError, match="runtime parameters or global tensor access"):
        build_ttl_without_compile(device_mod, target)


def test_ttl_codegen_rejects_unmatched_transfer():
    net = T.tt.PipeNet([T.tt.Pipe(src=(0, 0), dst=(1, 0))])

    @T.prim_func
    def main():
        with T.Kernel(2, 1, threads=1):
            block = T.alloc_shared(
                (32, 32),
                T.bfloat16,
                annotations={"tt.dfb_block_count": 2},
            )
            for pipe in T.tt.foreach_src(net):
                T.copy(block, pipe)

    target = _target()
    device_mod = _device_module(main, target)

    with pytest.raises(NotImplementedError, match="exactly one matched pair"):
        build_ttl_without_compile(device_mod, target)


def test_ttl_codegen_rejects_collective_transfer():
    net = T.tt.PipeNet(
        [T.tt.Pipe(src=(0, 0), dst=T.tt.CoreRange(begin=(1, 0), end=(3, 1)))]
    )

    @T.prim_func
    def main():
        with T.Kernel(3, 1, threads=1):
            block = T.alloc_shared(
                (32, 32),
                T.bfloat16,
                annotations={"tt.dfb_block_count": 2},
            )
            for pipe in T.tt.foreach_src(net):
                T.copy(block, pipe)
            for pipe in T.tt.foreach_dst(net):
                T.copy(pipe, block)

    target = _target()
    device_mod = _device_module(main, target)

    with pytest.raises(NotImplementedError, match="collective destination range"):
        build_ttl_without_compile(device_mod, target)
