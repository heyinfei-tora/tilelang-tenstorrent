import pytest

from tilelang import language as T
from tilelang import tvm


_ALLOC_BUFFER_ANNOTATIONS = "tl.alloc_buffer_annotations"


def _make_kernel(annotations, shape=(64, 128)):
    @T.prim_func
    def kernel(A: T.Tensor(shape, T.bfloat16)):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared(shape, T.bfloat16, annotations=annotations)
            A[0, 0] = shared[0, 0]

    return kernel


def _find_metadata_block(func):
    blocks = []

    def collect(node):
        if (
            isinstance(node, tvm.tirx.SBlock)
            and _ALLOC_BUFFER_ANNOTATIONS in node.annotations
        ):
            blocks.append(node)

    tvm.tirx.stmt_functor.post_order_visit(func.body, collect)
    assert len(blocks) == 1
    return blocks[0]


def test_alloc_shared_without_annotations_keeps_existing_ir_shape():
    kernel = _make_kernel(None)
    blocks = []

    def collect(node):
        if isinstance(node, tvm.tirx.SBlock) and node.alloc_buffers:
            blocks.append(node)

    tvm.tirx.stmt_functor.post_order_visit(kernel.body, collect)
    assert len(blocks) == 1
    assert _ALLOC_BUFFER_ANNOTATIONS not in blocks[0].annotations


def test_alloc_shared_records_metadata_by_buffer_identity():
    kernel = _make_kernel(
        {
            "test.block_count": 2,
            "test.tile_shape": (16, 32),
        }
    )

    block = _find_metadata_block(kernel)
    assert len(block.alloc_buffers) == 1
    buffer = block.alloc_buffers[0]
    metadata_by_buffer = block.annotations[_ALLOC_BUFFER_ANNOTATIONS]
    assert buffer.data in metadata_by_buffer

    metadata = metadata_by_buffer[buffer.data]
    assert metadata["test.block_count"] == 2
    assert list(metadata["test.tile_shape"]) == [16, 32]


def test_alloc_shared_keeps_metadata_separate_for_multiple_buffers():
    @T.prim_func
    def kernel(A: T.Tensor((64, 128), T.bfloat16)):
        with T.Kernel(1, threads=1):
            lhs = T.alloc_shared(
                (64, 128),
                T.bfloat16,
                annotations={"test.buffer_id": 1},
            )
            rhs = T.alloc_shared(
                (64, 128),
                T.bfloat16,
                annotations={"test.buffer_id": 3},
            )
            A[0, 0] = lhs[0, 0] + rhs[0, 0]

    block = _find_metadata_block(kernel)
    assert len(block.alloc_buffers) == 2
    metadata_by_buffer = block.annotations[_ALLOC_BUFFER_ANNOTATIONS]
    first, second = block.alloc_buffers
    assert metadata_by_buffer[first.data]["test.buffer_id"] == 1
    assert metadata_by_buffer[second.data]["test.buffer_id"] == 3


def test_alloc_shared_rejects_non_mapping_annotations():
    with pytest.raises(TypeError, match="must be a mapping"):
        _make_kernel([("test.key", 1)])


def test_alloc_shared_rejects_non_string_annotation_keys():
    with pytest.raises(TypeError, match="keys must be strings"):
        _make_kernel({1: "value"})
