import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
)

from attention.paged_attention import (
    compute_decode_slot_mapping,
    compute_prefill_slot_mapping,
    gather_batch_kv,
    paged_attention_decode,
    paged_attention_prefill,
)
from cache.kv_pool import KVCachePool


class PagedModelRunner:
    """Run Qwen2 forward passes with paged KV cache."""

    def __init__(self, model, kv_pool: KVCachePool):
        self.model = model
        self.kv_pool = kv_pool
        config = model.config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.block_size = kv_pool.block_size
        self.scaling = getattr(model.model.layers[0].self_attn, "scaling", self.head_dim**-0.5)
        self.rotary_emb = model.model.rotary_emb

    def forward_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        block_tables: list[list[int]],
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        hidden_states = self.model.model.embed_tokens(input_ids)

        slot_mapping, token_indices = compute_prefill_slot_mapping(
            attention_mask, block_tables, self.block_size
        )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.model.model.layers):
            hidden_states = self._prefill_layer(
                layer,
                layer_idx,
                hidden_states,
                attention_mask,
                position_ids,
                position_embeddings,
                block_tables,
                slot_mapping,
                token_indices,
            )

        hidden_states = self.model.model.norm(hidden_states)
        return self.model.lm_head(hidden_states)

    def forward_decode(
        self,
        input_ids: torch.Tensor,
        block_tables: list[list[int]],
        seq_lens: list[int],
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        device = input_ids.device
        position_ids = torch.tensor(
            seq_lens, dtype=torch.long, device=device
        ).unsqueeze(1)

        hidden_states = self.model.model.embed_tokens(input_ids)
        slot_mapping = compute_decode_slot_mapping(
            block_tables, seq_lens, self.block_size, device
        )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.model.model.layers):
            hidden_states = self._decode_layer(
                layer,
                layer_idx,
                hidden_states,
                block_tables,
                seq_lens,
                position_ids,
                position_embeddings,
                slot_mapping,
            )

        hidden_states = self.model.model.norm(hidden_states)
        return self.model.lm_head(hidden_states)

    def _prefill_layer(
        self,
        layer,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        block_tables: list[list[int]],
        slot_mapping: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        batch_size, seq_len, _ = hidden_states.shape
        query_states = (
            layer.self_attn.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            layer.self_attn.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            layer.self_attn.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if slot_mapping.numel() > 0:
            flat_keys = key_states[
                token_indices[:, 0], :, token_indices[:, 1], :
            ].reshape(-1, self.num_kv_heads, self.head_dim)
            flat_values = value_states[
                token_indices[:, 0], :, token_indices[:, 1], :
            ].reshape(-1, self.num_kv_heads, self.head_dim)
            self.kv_pool.write_slots(
                layer_idx, slot_mapping, flat_keys, flat_values
            )

        key_expanded = repeat_kv(key_states, self.num_kv_groups)
        value_expanded = repeat_kv(value_states, self.num_kv_groups)

        attn_output = paged_attention_prefill(
            layer.self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.scaling,
        )
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = layer.self_attn.o_proj(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        return residual + hidden_states

    def _decode_layer(
        self,
        layer,
        layer_idx: int,
        hidden_states: torch.Tensor,
        block_tables: list[list[int]],
        seq_lens: list[int],
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        batch_size, q_len, _ = hidden_states.shape
        query_states = (
            layer.self_attn.q_proj(hidden_states)
            .view(batch_size, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            layer.self_attn.k_proj(hidden_states)
            .view(batch_size, q_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            layer.self_attn.v_proj(hidden_states)
            .view(batch_size, q_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        flat_keys = key_states[:, :, 0, :]
        flat_values = value_states[:, :, 0, :]
        self.kv_pool.write_slots(layer_idx, slot_mapping, flat_keys, flat_values)

        new_seq_lens = [seq_len + 1 for seq_len in seq_lens]
        cached_keys, cached_values = gather_batch_kv(
            self.kv_pool,
            layer_idx,
            block_tables,
            new_seq_lens,
            self.num_kv_heads,
            self.head_dim,
        )

        attn_output = paged_attention_decode(
            layer.self_attn,
            query_states,
            cached_keys,
            cached_values,
            self.scaling,
            new_seq_lens,
        )
        attn_output = attn_output.reshape(batch_size, q_len, -1)
        attn_output = layer.self_attn.o_proj(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        return residual + hidden_states
