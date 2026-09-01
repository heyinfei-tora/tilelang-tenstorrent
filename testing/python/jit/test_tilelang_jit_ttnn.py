from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

import pytest

from tilelang import tvm
from tilelang.cache import _dispatch_map
from tilelang.engine.param import CompiledArtifact, KernelParam
from tilelang.jit.adapter.ttnn.adapter import TTNNKernelAdapter, _load_ttl_module
from tilelang.jit.adapter.ttnn.kernel_cache import TTNNKernelCache
from tilelang.jit.kernel import JITKernel


TARGET = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})
SOURCE = "__tilelang_ttl_artifact__ = {}\n"


def _func():
    return tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(0)).with_attr("global_symbol", "main")


def _loader(operation=lambda: "launched", metadata=None):
    metadata = metadata or {
        "version": 1,
        "entrypoint": "main_kernel",
        "target_arch": "wormhole_b0",
        "grid": (2, 1),
    }
    calls = []

    def load(source, module_name):
        calls.append((source, module_name))
        return SimpleNamespace(
            __tilelang_ttl_artifact__=metadata,
            main_kernel=operation,
        )

    return load, calls


def test_ttnn_adapter_injected_loader_builds_loads_and_launches():
    loader, calls = _loader()
    adapter = TTNNKernelAdapter(
        params=[],
        result_idx=None,
        target=TARGET,
        func_or_mod=_func(),
        device_kernel_source=SOURCE,
        module_loader=loader,
    )

    assert len(calls) == 1
    assert calls[0][0] == SOURCE
    assert calls[0][1].startswith("_tilelang_ttl_")
    assert adapter() == "launched"
    assert adapter.get_kernel_source() == SOURCE
    with pytest.raises(TypeError, match="keyword arguments"):
        adapter(unexpected=1)
    with pytest.raises(TypeError, match="expected 0 arguments"):
        adapter(object())


def test_default_ttl_module_loader_registers_importable_source():
    source = "def main_kernel():\n    return 'loaded'\n"

    module = _load_ttl_module(source, "_tilelang_ttl_loader_test")

    assert module.main_kernel() == "loaded"
    assert inspect.getsource(module.main_kernel) == source


@pytest.mark.parametrize(
    "metadata, match",
    [
        (None, "missing mapping metadata"),
        ({"version": 2, "entrypoint": "main_kernel", "target_arch": "wormhole_b0"}, "version"),
        ({"version": 1, "entrypoint": "main_kernel", "target_arch": "blackhole"}, "execution target"),
        ({"version": 1, "entrypoint": "missing", "target_arch": "wormhole_b0"}, "missing or not callable"),
    ],
)
def test_ttnn_adapter_rejects_invalid_artifact_metadata(metadata, match):
    def loader(source, module_name):
        del source, module_name
        namespace = SimpleNamespace(main_kernel=lambda: None)
        if metadata is not None:
            namespace.__tilelang_ttl_artifact__ = metadata
        return namespace

    with pytest.raises(ValueError, match=match):
        TTNNKernelAdapter(
            params=[],
            result_idx=None,
            target=TARGET,
            func_or_mod=_func(),
            device_kernel_source=SOURCE,
            module_loader=loader,
        )


def test_ttnn_adapter_rejects_runtime_params_results_and_missing_sdk():
    loader, _ = _loader()
    param = KernelParam(shape=[tvm.tirx.IntImm("int32", 1)], dtype="float32")
    with pytest.raises(NotImplementedError, match="runtime parameters"):
        TTNNKernelAdapter([param], None, TARGET, _func(), device_kernel_source=SOURCE, module_loader=loader)
    with pytest.raises(NotImplementedError, match="result allocation"):
        TTNNKernelAdapter([], [0], TARGET, _func(), device_kernel_source=SOURCE, module_loader=loader)
    with pytest.raises(ImportError, match="optional 'ttl' and 'ttnn'"):
        TTNNKernelAdapter([], None, TARGET, _func(), device_kernel_source="import ttl\n")


def test_jit_fresh_dispatch_constructs_ttnn_adapter(monkeypatch):
    captured = {}

    class FakeAdapter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    kernel_module = importlib.import_module("tilelang.jit.kernel")
    monkeypatch.setattr(kernel_module, "TTNNKernelAdapter", FakeAdapter)
    kernel = JITKernel.__new__(JITKernel)
    kernel.execution_backend = "ttnn"
    kernel.target = TARGET
    kernel.compile_flags = None
    kernel.verbose = False
    artifact = CompiledArtifact(
        host_mod=tvm.IRModule(),
        device_mod=tvm.IRModule(),
        params=[],
        kernel_source=SOURCE,
        target=TARGET,
    )

    adapter = kernel._create_adapter_from_artifact(
        _func(),
        None,
        artifact,
        {},
        {
            "kernel": "main",
            "target": str(TARGET),
            "target_host": None,
            "backend": "ttnn",
        },
    )

    assert isinstance(adapter, FakeAdapter)
    assert captured["device_kernel_source"] == SOURCE
    assert captured["params"] == []


def test_jit_ttnn_database_dispatch_rejects_persistent_cache():
    kernel = JITKernel.__new__(JITKernel)
    kernel.execution_backend = "ttnn"
    kernel.target = TARGET

    with pytest.raises(NotImplementedError, match="persistent cache is not supported"):
        kernel._create_adapter_from_database(
            params=[],
            result_idx=None,
            target=TARGET,
            func_or_mod=_func(),
            host_kernel_source=None,
            device_kernel_source=SOURCE,
            kernel_lib_path="",
        )


def test_ttnn_cache_is_memory_only(tmp_path):
    cache = _dispatch_map["ttnn"]
    assert isinstance(cache, TTNNKernelCache)

    cache._save_kernel_to_disk("unused", object())

    assert cache._load_kernel_from_disk("unused", None) is None
    assert list(tmp_path.iterdir()) == []
