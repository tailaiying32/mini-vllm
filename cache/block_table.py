import math

from cache.block_manager import BlockAllocator


class SequenceBlockTable:
    """Maps logical block indices to physical block IDs for one sequence."""

    def __init__(self, block_ids: list[int] | None = None):
        self._block_ids: list[int] = list(block_ids or [])

    def all_blocks(self) -> list[int]:
        return list(self._block_ids)

    def num_blocks(self) -> int:
        return len(self._block_ids)

    def block_ids(self) -> list[int]:
        return self._block_ids

    def ensure_blocks_for_length(
        self, num_tokens: int, block_size: int, allocator: BlockAllocator
    ) -> None:
        blocks_needed = math.ceil(num_tokens / block_size) if num_tokens > 0 else 0
        while len(self._block_ids) < blocks_needed:
            self._block_ids.extend(allocator.allocate(1))

    def slot_for_position(self, position: int, block_size: int) -> int:
        if position < 0:
            raise ValueError("position must be non-negative")
        logical_block = position // block_size
        if logical_block >= len(self._block_ids):
            raise IndexError(
                f"Position {position} requires block {logical_block}, "
                f"but only {len(self._block_ids)} blocks allocated"
            )
        offset = position % block_size
        physical_block = self._block_ids[logical_block]
        return physical_block * block_size + offset
