import math

import torch
from transformers.models.qwen2.modeling_qwen2 import (
    eager_attention_forward,
    repeat_kv,
)


def _build_prefill_mask(
    query: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    seq_len = query.shape[2]
    min_value = torch.finfo(query.dtype).min
    attn_mask = torch.triu(
        torch.full((seq_len, seq_len), min_value, device=query.device, dtype=query.dtype),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)

    if attention_mask is not None and not bool(attention_mask.all()):
        pad_q = attention_mask[:, None, :, None].to(dtype=torch.bool)
        pad_k = attention_mask[:, None, None, :].to(dtype=torch.bool)
        attn_mask = attn_mask.masked_fill(~pad_q, min_value)
        attn_mask = attn_mask.masked_fill(~pad_k, min_value)

    return attn_mask


def paged_attention_prefill(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    **_,
) -> torch.Tensor:
    """Causal attention for prefill using HF eager attention."""
    attn_output, _ = eager_attention_forward(
        module,
        query,
        key,
        value,
        _build_prefill_mask(query, attention_mask),
        scaling=scaling,
    )
    return attn_output


def paged_attention_decode(
    module,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scaling: float,
    seq_lens: list[int] | None = None,
) -> torch.Tensor:
    """Attention for decode: one query token per sequence."""
    if seq_lens is None or len(set(seq_lens)) == 1:
        attn_output, _ = eager_attention_forward(
            module,
            query,
            keys,
            values,
            None,
            scaling=scaling,
        )
        return attn_output

    outputs: list[torch.Tensor] = []
    for batch_idx, seq_len in enumerate(seq_lens):
        attn_output, _ = eager_attention_forward(
            module,
            query[batch_idx : batch_idx + 1],
            keys[batch_idx : batch_idx + 1, :, :seq_len, :],
            values[batch_idx : batch_idx + 1, :, :seq_len, :],
            None,
            scaling=scaling,
        )
        outputs.append(attn_output)
    return torch.cat(outputs, dim=0)


def gather_batch_kv(
    kv_pool,
    layer_idx: int,
    block_tables: list[list[int]],
    seq_lens: list[int],
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V for a batch of sequences into padded tensors."""
    batch_size = len(seq_lens)
    max_seq_len = max(seq_lens) if seq_lens else 0
    device = kv_pool.device
    dtype = kv_pool.dtype

    keys = torch.zeros(
        batch_size,
        num_kv_heads,
        max_seq_len,
        head_dim,
        dtype=dtype,
        device=device,
    )
    values = torch.zeros_like(keys)

    for batch_idx, seq_len in enumerate(seq_lens):
        if seq_len == 0:
            continue
        k, v = kv_pool.gather_sequence_kv(
            layer_idx, block_tables[batch_idx], seq_len
        )
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        keys[batch_idx, :, :seq_len, :] = k[0]
        values[batch_idx, :, :seq_len, :] = v[0]

    return keys, values


def compute_prefill_slot_mapping(
    attention_mask: torch.Tensor,
    block_tables: list[list[int]],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build flat slot indices and token indices for prefill scatter."""
    batch_size, seq_len = attention_mask.shape
    slots: list[int] = []
    indices: list[list[int]] = []

    for batch_idx in range(batch_size):
        valid_count = int(attention_mask[batch_idx].sum().item())
        pad_len = seq_len - valid_count
        block_ids = block_tables[batch_idx]
        for col in range(pad_len, seq_len):
            seq_pos = col - pad_len
            logical_block = seq_pos // block_size
            offset = seq_pos % block_size
            physical_block = block_ids[logical_block]
            slots.append(physical_block * block_size + offset)
            indices.append([batch_idx, col])

    if not slots:
        empty = torch.empty(0, dtype=torch.long, device=attention_mask.device)
        return empty, torch.empty(0, 2, dtype=torch.long, device=attention_mask.device)

    return (
        torch.tensor(slots, dtype=torch.long, device=attention_mask.device),
        torch.tensor(indices, dtype=torch.long, device=attention_mask.device),
    )


def compute_decode_slot_mapping(
    block_tables: list[list[int]],
    seq_lens: list[int],
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Flat slot indices for one new token per sequence."""
    slots: list[int] = []
    for block_ids, seq_len in zip(block_tables, seq_lens):
        logical_block = seq_len // block_size
        offset = seq_len % block_size
        physical_block = block_ids[logical_block]
        slots.append(physical_block * block_size + offset)
    return torch.tensor(slots, dtype=torch.long, device=device)


def blocks_needed_for_tokens(num_tokens: int, block_size: int) -> int:
    return math.ceil(num_tokens / block_size) if num_tokens > 0 else 0
