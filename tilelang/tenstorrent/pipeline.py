"""Tenstorrent lowering pipeline."""

from __future__ import annotations

from tvm import IRModule, tirx
from tvm.target import Target

import tilelang
from tilelang.backend.pass_pipeline import PassPipeline

from . import transform as tt_transform


def TenstorrentPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    """Lower frontend TIR to the current hardware-independent TT contract."""

    mod = tirx.transform.BindTarget(target)(mod)
    mod = tt_transform.ValidateTenstorrentKernelLaunch()(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tt_transform.LowerTenstorrentFrontendAnnotations()(mod)

    # Allocation planning may move SBlock alloc_buffers.  The frontend pass
    # has already captured their metadata by data-Var identity, which survives
    # these passes and the unchanged shared LowerOpaqueBlock implementation.
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tt_transform.LowerTenstorrentBufferAllocations()(mod)

    mod = tilelang.transform.Simplify()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tirx.transform.RemoveNoOp()(mod)
    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)

    # Keep host/device splitting in the target pipeline even though device
    # code generation remains intentionally hardware-independent here.
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod


TENSTORRENT_PIPELINE = PassPipeline("tenstorrent", TenstorrentPassPipelineBody)
