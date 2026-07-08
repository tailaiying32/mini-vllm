import torch
from dataclasses import dataclass

from attention.model_runner import PagedModelRunner
from cache.block_manager import BlockAllocator, PoolExhausted
from cache.block_table import SequenceBlockTable
from cache.kv_pool import BLOCK_SIZE, KVCachePool, compute_num_blocks
from cache.paged_state import PagedKVState
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
model.eval()

_config = model.config
_num_blocks = compute_num_blocks(
    _config.num_hidden_layers,
    _config.num_key_value_heads,
    _config.hidden_size // _config.num_attention_heads,
    model.dtype,
    torch.device(device),
)
block_allocator = BlockAllocator(_num_blocks)
kv_pool = KVCachePool(
    num_blocks=_num_blocks,
    num_layers=_config.num_hidden_layers,
    block_size=BLOCK_SIZE,
    num_kv_heads=_config.num_key_value_heads,
    head_dim=_config.hidden_size // _config.num_attention_heads,
    dtype=model.dtype,
    device=torch.device(device),
)
runner = PagedModelRunner(model, kv_pool)


@dataclass
class StepResult:
    input_ids: torch.Tensor
    token_text: str | None
    is_done: bool
    kv_state: PagedKVState


def free_kv_state(kv_state: PagedKVState | None) -> None:
    if kv_state is not None:
        block_allocator.free(kv_state.block_table.all_blocks())


def _allocate_prefill_state(prompt_len: int) -> PagedKVState:
    block_table = SequenceBlockTable()
    block_table.ensure_blocks_for_length(prompt_len, BLOCK_SIZE, block_allocator)
    return PagedKVState(block_table=block_table, seq_len=0)


def _ensure_decode_capacity(kv_state: PagedKVState) -> None:
    block_table = kv_state.block_table
    needed = kv_state.seq_len + 1
    block_table.ensure_blocks_for_length(needed, BLOCK_SIZE, block_allocator)


def prepare_request(prompt: str) -> torch.Tensor:
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


def batch_prepare_requests(prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad prompts to max length and build an attention mask."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)


def _build_step_result(
    input_ids: torch.Tensor,
    next_token_id: torch.Tensor,
    kv_state: PagedKVState,
    *,
    increment_seq_len: bool = True,
) -> StepResult:
    updated_input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    if increment_seq_len:
        kv_state.seq_len += 1

    if next_token_id.item() == tokenizer.eos_token_id:
        return StepResult(updated_input_ids, None, True, kv_state)

    token_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
    return StepResult(updated_input_ids, token_text, False, kv_state)


def prefill_one_token(input_ids: torch.Tensor) -> StepResult:
    prompt_len = input_ids.shape[1]
    kv_state = _allocate_prefill_state(prompt_len)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        logits = runner.forward_prefill(
            input_ids,
            attention_mask,
            [kv_state.block_table.block_ids()],
        )

    next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    kv_state.seq_len = prompt_len
    return _build_step_result(
        input_ids, next_token_id, kv_state, increment_seq_len=False
    )


def generate_one_token(
    input_ids: torch.Tensor,
    kv_state: PagedKVState | None = None,
) -> StepResult:
    with torch.inference_mode():
        if kv_state is None:
            return prefill_one_token(input_ids)

        _ensure_decode_capacity(kv_state)
        decode_input = input_ids[:, -1:]
        logits = runner.forward_decode(
            decode_input,
            [kv_state.block_table.block_ids()],
            [kv_state.seq_len],
        )

        next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        return _build_step_result(input_ids, next_token_id, kv_state)


def batch_prefill_one_token(requests: list) -> list[StepResult]:
    if not requests:
        return []

    if len(requests) == 1:
        return [prefill_one_token(requests[0].input_ids)]

    prompts = [r.prompt for r in requests]
    batched_input_ids, attention_mask = batch_prepare_requests(prompts)
    kv_states: list[PagedKVState] = []

    for i, request in enumerate(requests):
        prompt_len = int(attention_mask[i].sum().item())
        try:
            kv_state = _allocate_prefill_state(prompt_len)
        except PoolExhausted as exc:
            raise PoolExhausted(f"Prefill allocation failed for request {i}") from exc
        kv_states.append(kv_state)

    block_tables = [state.block_table.block_ids() for state in kv_states]

    with torch.inference_mode():
        logits = runner.forward_prefill(
            batched_input_ids,
            attention_mask,
            block_tables,
        )

    results: list[StepResult] = []
    last_col = batched_input_ids.shape[1] - 1

    for i, request in enumerate(requests):
        next_token_id = torch.argmax(
            logits[i : i + 1, last_col, :], dim=-1, keepdim=True
        )
        kv_states[i].seq_len = int(attention_mask[i].sum().item())
        results.append(
            _build_step_result(
                request.input_ids,
                next_token_id,
                kv_states[i],
                increment_seq_len=False,
            )
        )

    return results


def batch_decode_one_token(requests: list) -> list[StepResult]:
    if not requests:
        return []

    if len(requests) == 1:
        request = requests[0]
        return [generate_one_token(request.input_ids, request.kv_state)]

    for request in requests:
        _ensure_decode_capacity(request.kv_state)

    batched_input = torch.cat([r.input_ids[:, -1:] for r in requests], dim=0)
    block_tables = [r.kv_state.block_table.block_ids() for r in requests]
    seq_lens = [r.kv_state.seq_len for r in requests]

    with torch.inference_mode():
        logits = runner.forward_decode(batched_input, block_tables, seq_lens)

    results: list[StepResult] = []
    for i, request in enumerate(requests):
        next_token_id = torch.argmax(logits[i : i + 1, -1, :], dim=-1, keepdim=True)
        results.append(
            _build_step_result(request.input_ids, next_token_id, request.kv_state)
        )

    return results


def generate_tokens(prompt: str, max_tokens: int):
    input_ids = prepare_request(prompt)
    kv_state = None

    for _ in range(max_tokens):
        result = generate_one_token(input_ids, kv_state)
        input_ids = result.input_ids
        kv_state = result.kv_state
        if result.is_done:
            break
        yield result.token_text

    free_kv_state(kv_state)
