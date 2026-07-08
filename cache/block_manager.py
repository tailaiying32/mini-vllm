class PoolExhausted(Exception):
    """Raised when the block pool has no free blocks."""


class BlockAllocator:
    """Free-list allocator for physical KV cache blocks."""

    def __init__(self, num_blocks: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self._num_blocks = num_blocks
        self._free_blocks: list[int] = list(range(num_blocks - 1, -1, -1))

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def num_free(self) -> int:
        return len(self._free_blocks)

    def num_used(self) -> int:
        return self._num_blocks - len(self._free_blocks)

    def allocate(self, n: int = 1) -> list[int]:
        if n <= 0:
            return []
        if len(self._free_blocks) < n:
            raise PoolExhausted(
                f"Cannot allocate {n} blocks; only {len(self._free_blocks)} free"
            )
        return [self._free_blocks.pop() for _ in range(n)]

    def free(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            if block_id < 0 or block_id >= self._num_blocks:
                raise ValueError(f"Invalid block id: {block_id}")
            if block_id in self._free_blocks:
                raise ValueError(f"Block {block_id} is already free")
        self._free_blocks.extend(block_ids)
