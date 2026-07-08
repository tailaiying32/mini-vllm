from dataclasses import dataclass

from cache.block_table import SequenceBlockTable


@dataclass
class PagedKVState:
    """Per-request paged KV metadata."""

    block_table: SequenceBlockTable
    seq_len: int = 0
