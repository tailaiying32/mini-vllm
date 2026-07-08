import math

import pytest

from cache.block_manager import BlockAllocator, PoolExhausted
from cache.block_table import SequenceBlockTable
from cache.kv_pool import BLOCK_SIZE


def test_allocate_and_free_round_trip():
    allocator = BlockAllocator(4)
    assert allocator.num_free() == 4
    blocks = allocator.allocate(2)
    assert len(blocks) == 2
    assert allocator.num_free() == 2
    allocator.free(blocks)
    assert allocator.num_free() == 4


def test_pool_exhausted():
    allocator = BlockAllocator(2)
    allocator.allocate(2)
    with pytest.raises(PoolExhausted):
        allocator.allocate(1)


def test_block_table_slot_mapping():
    table = SequenceBlockTable([7, 3, 12])
    assert table.slot_for_position(0, BLOCK_SIZE) == 7 * BLOCK_SIZE + 0
    assert table.slot_for_position(16, BLOCK_SIZE) == 3 * BLOCK_SIZE + 0
    assert table.slot_for_position(39, BLOCK_SIZE) == 12 * BLOCK_SIZE + 7


def test_block_table_growth():
    allocator = BlockAllocator(8)
    table = SequenceBlockTable()
    table.ensure_blocks_for_length(40, BLOCK_SIZE, allocator)
    assert table.num_blocks() == math.ceil(40 / BLOCK_SIZE)
