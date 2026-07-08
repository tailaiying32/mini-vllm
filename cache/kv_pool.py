import math
import os

import torch

BLOCK_SIZE = int(os.environ.get("MINI_VLLM_BLOCK_SIZE", "16"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("MINI_VLLM_GPU_MEMORY_UTIL", "0.5"))
DEFAULT_NUM_BLOCKS = int(os.environ.get("MINI_VLLM_NUM_BLOCKS", "512"))


def compute_num_blocks(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    block_size: int = BLOCK_SIZE,
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
) -> int:
    bytes_per_block = (
        2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize
    )
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return max(1, int(free_bytes * gpu_memory_utilization // bytes_per_block))
    return DEFAULT_NUM_BLOCKS


class KVCachePool:
    """Pre-allocated paged KV cache storage shared across all sequences."""

    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        block_shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.key_caches = tuple(
            torch.zeros(block_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        )
        self.value_caches = tuple(
            torch.zeros(block_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        )

    def write_slots(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Scatter keys/values into the pool.

        keys/values: [num_tokens, num_kv_heads, head_dim]
        slot_mapping: [num_tokens] flat slot indices
        """
        if slot_mapping.numel() == 0:
            return
        key_cache = self.key_caches[layer_idx]
        value_cache = self.value_caches[layer_idx]
        block_ids = slot_mapping // self.block_size
        offsets = slot_mapping % self.block_size
        for i, (block_id, offset) in enumerate(
            zip(block_ids.tolist(), offsets.tolist())
        ):
            key_cache[block_id, offset] = keys[i]
            value_cache[block_id, offset] = values[i]

    def gather_sequence_kv(
        self,
        layer_idx: int,
        block_ids: list[int],
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather contiguous K/V for one sequence from the pool."""
        if seq_len == 0:
            empty = torch.empty(
                0,
                self.num_kv_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            return empty, empty

        parts_k: list[torch.Tensor] = []
        parts_v: list[torch.Tensor] = []
        num_logical_blocks = math.ceil(seq_len / self.block_size)

        for logical_idx in range(num_logical_blocks):
            physical_block = block_ids[logical_idx]
            offset = min(self.block_size, seq_len - logical_idx * self.block_size)
            parts_k.append(self.key_caches[layer_idx][physical_block, :offset])
            parts_v.append(self.value_caches[layer_idx][physical_block, :offset])

        return torch.cat(parts_k, dim=0), torch.cat(parts_v, dim=0)
