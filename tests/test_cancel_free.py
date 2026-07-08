import torch

from cache.block_manager import BlockAllocator
from cache.block_table import SequenceBlockTable
from cache.kv_pool import BLOCK_SIZE


def test_free_blocks_on_cancel():
    allocator = BlockAllocator(16)
    table = SequenceBlockTable()
    table.ensure_blocks_for_length(32, BLOCK_SIZE, allocator)
    blocks = table.all_blocks()
    assert allocator.num_free() == 16 - len(blocks)

    allocator.free(blocks)
    assert allocator.num_free() == 16
