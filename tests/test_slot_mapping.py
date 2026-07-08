import torch

from attention.paged_attention import (
    compute_decode_slot_mapping,
    compute_prefill_slot_mapping,
)
from cache.kv_pool import BLOCK_SIZE


def test_prefill_slot_mapping_left_padded():
    # Two sequences padded to length 4: [pad, a, b, c] and [x, y, z, w]
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    block_tables = [[5], [9]]
    slot_mapping, token_indices = compute_prefill_slot_mapping(
        attention_mask, block_tables, BLOCK_SIZE
    )

    assert slot_mapping.tolist() == [
        5 * BLOCK_SIZE + 0,
        5 * BLOCK_SIZE + 1,
        5 * BLOCK_SIZE + 2,
        9 * BLOCK_SIZE + 0,
        9 * BLOCK_SIZE + 1,
        9 * BLOCK_SIZE + 2,
        9 * BLOCK_SIZE + 3,
    ]
    assert token_indices.tolist() == [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
    ]


def test_decode_slot_mapping():
    block_tables = [[2, 7], [4]]
    seq_lens = [20, 5]
    slots = compute_decode_slot_mapping(
        block_tables, seq_lens, BLOCK_SIZE, torch.device("cpu")
    )
    assert slots.tolist() == [
        7 * BLOCK_SIZE + 4,
        4 * BLOCK_SIZE + 5,
    ]
