"""TTNN execution adapter for importable TT-Lang Python artifacts."""

from __future__ import annotations

import hashlib
import linecache
import sys
import types
from collections.abc import Callable, Mapping
from typing import Any

from tvm import IRModule, tirx
from tvm.target import Target

from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.base import BaseKernelAdapter


TTLModuleLoader = Callable[[str, str], object]


def _load_ttl_module(source: str, module_name: str) -> types.ModuleType:
    """Build and load one generated Python module through TT-Lang decorators."""

    filename = f"<{module_name}.py>"
    try:
        code = compile(source, filename, "exec")
    except SyntaxError as err:
        raise RuntimeError(f"Generated TT-Lang source is not valid Python: {err}") from err

    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    module = types.ModuleType(module_name)
    module.__file__ = filename
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except ModuleNotFoundError as err:
        sys.modules.pop(module_name, None)
        if err.name in {"ttl", "ttnn"}:
            raise ImportError(
                "TTNN execution requires the optional 'ttl' and 'ttnn' packages; "
                "install a compatible tt-lang hardware environment"
            ) from err
        raise
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


class TTNNKernelAdapter(BaseKernelAdapter):
    """Load and invoke the operation declared by a source-only TTL artifact."""

    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int] | int | None,
        target: str | dict[str, object] | Target,
        func_or_mod: tirx.PrimFunc | IRModule,
        device_mod: IRModule | None = None,
        device_kernel_source: str | None = None,
        verbose: bool = False,
        module_loader: TTLModuleLoader | None = None,
        **_: Any,
    ) -> None:
        del func_or_mod, device_mod, verbose
        if params:
            raise NotImplementedError("TTNN source artifact v1 does not support runtime parameters")
        if result_idx not in (None, []):
            raise NotImplementedError("TTNN source artifact v1 does not support result allocation")
        if not device_kernel_source:
            raise ValueError("TTNN execution requires non-empty generated TT-Lang source")

        self.params = params
        self.result_idx = []
        self.target = target if isinstance(target, Target) else Target(target)
        self.device_kernel_source = device_kernel_source
        digest = hashlib.sha256(device_kernel_source.encode("utf-8")).hexdigest()[:16]
        loader = module_loader or _load_ttl_module
        self.ttl_module = loader(device_kernel_source, f"_tilelang_ttl_{digest}")
        metadata = getattr(self.ttl_module, "__tilelang_ttl_artifact__", None)
        if not isinstance(metadata, Mapping):
            raise ValueError("TT-Lang module is missing mapping metadata '__tilelang_ttl_artifact__'")
        if metadata.get("version") != 1:
            raise ValueError(f"Unsupported TT-Lang artifact version {metadata.get('version')!r}; expected 1")

        target_arch = str(self.target.attrs.get("arch", ""))
        if metadata.get("target_arch") != target_arch:
            raise ValueError(
                f"TT-Lang artifact targets {metadata.get('target_arch')!r}, but execution target is {target_arch!r}"
            )
        entrypoint = metadata.get("entrypoint")
        if not isinstance(entrypoint, str) or not entrypoint:
            raise ValueError("TT-Lang artifact metadata must contain a non-empty 'entrypoint'")
        operation = getattr(self.ttl_module, entrypoint, None)
        if not callable(operation):
            raise ValueError(f"TT-Lang artifact entrypoint {entrypoint!r} is missing or not callable")
        self.metadata = dict(metadata)
        self.operation = operation
        self.func = self._convert_torch_func()

    def _convert_torch_func(self) -> Callable[..., Any]:
        def launch(*args: Any, **kwargs: Any) -> Any:
            if kwargs:
                names = ", ".join(sorted(kwargs))
                raise TypeError(f"TTNN source artifact v1 does not accept keyword arguments: {names}")
            if len(args) != len(self.params):
                raise TypeError(f"TTNN source artifact v1 expected {len(self.params)} arguments, got {len(args)}")
            return self.operation(*args)

        return launch

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        del kernel_only
        return self.device_kernel_source

    def get_host_source(self) -> str:
        return ""
