import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
model.eval()


@dataclass
class StepResult:
    input_ids: torch.Tensor
    token_text: str | None
    is_done: bool
    past_key_values: DynamicCache


def _cache_seq_len(past_key_values: DynamicCache) -> int:
    return past_key_values.get_seq_length()


def pad_dynamic_cache(past_key_values: DynamicCache, target_len: int) -> DynamicCache:
    padded_layers = []
    for layer in past_key_values.layers:
        keys, values = layer.keys, layer.values
        pad_len = target_len - keys.shape[-2]
        if pad_len > 0:
            keys = torch.nn.functional.pad(keys, (0, 0, pad_len, 0))
            values = torch.nn.functional.pad(values, (0, 0, pad_len, 0))
        padded_layers.append((keys, values))
    return DynamicCache(padded_layers)


def stack_past_key_values(caches: list[DynamicCache]) -> DynamicCache:
    max_len = max(_cache_seq_len(c) for c in caches)
    padded = [pad_dynamic_cache(c, max_len) for c in caches]
    stacked_layers = []
    for layer_idx in range(len(padded[0].layers)):
        keys = torch.cat([padded[i].layers[layer_idx].keys for i in range(len(padded))], dim=0)
        values = torch.cat([padded[i].layers[layer_idx].values for i in range(len(padded))], dim=0)
        stacked_layers.append((keys, values))
    return DynamicCache(stacked_layers)


def unstack_past_key_values(batched_cache: DynamicCache, batch_size: int) -> list[DynamicCache]:
    per_request = []
    for i in range(batch_size):
        layers = [
            (layer.keys[i : i + 1], layer.values[i : i + 1])
            for layer in batched_cache.layers
        ]
        per_request.append(DynamicCache(layers))
    return per_request


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


def batch_prefill_one_token(requests: list) -> list[StepResult]:
    """Run batched prefill for multiple requests. Each request has .input_ids [1, seq]."""
    if not requests:
        return []

    if len(requests) == 1:
        return [prefill_one_token(requests[0].input_ids)]

    prompts = [r.prompt for r in requests]
    batched_input_ids, attention_mask = batch_prepare_requests(prompts)
    # Left-padded batches are right-aligned; last token is always at the final column.
    last_token_indices = torch.full(
        (len(requests),),
        batched_input_ids.shape[1] - 1,
        device=batched_input_ids.device,
        dtype=torch.long,
    )

    with torch.inference_mode():
        output = model(
            batched_input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    per_request_caches = unstack_past_key_values(output.past_key_values, len(requests))
    results: list[StepResult] = []

    for i, request in enumerate(requests):
        idx = last_token_indices[i].item()
        next_token_id = torch.argmax(
            output.logits[i : i + 1, idx, :], dim=-1, keepdim=True
        )
        results.append(
            _build_step_result(request.input_ids, next_token_id, per_request_caches[i])
        )

    return results


def _build_step_result(
    input_ids: torch.Tensor,
    next_token_id: torch.Tensor,
    past_key_values: DynamicCache,
) -> StepResult:
    updated_input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    if next_token_id.item() == tokenizer.eos_token_id:
        return StepResult(updated_input_ids, None, True, past_key_values)

    token_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
    return StepResult(updated_input_ids, token_text, False, past_key_values)


def generate_one_token(
    input_ids: torch.Tensor,
    past_key_values: DynamicCache | None = None,
) -> StepResult:
    with torch.inference_mode():
        if past_key_values is None:
            model_input = input_ids
        else:
            model_input = input_ids[:, -1:]

        output = model(
            model_input,
            past_key_values=past_key_values,
            use_cache=True,
        )

        next_token_id = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
        return _build_step_result(input_ids, next_token_id, output.past_key_values)


def prefill_one_token(input_ids: torch.Tensor) -> StepResult:
    return generate_one_token(input_ids, past_key_values=None)


def batch_decode_one_token(requests: list) -> list[StepResult]:
    if not requests:
        return []

    if len(requests) == 1:
        request = requests[0]
        return [generate_one_token(request.input_ids, request.past_key_values)]

    results_by_id: dict[int, StepResult] = {}
    groups: dict[int, list] = {}
    for request in requests:
        seq_len = _cache_seq_len(request.past_key_values)
        groups.setdefault(seq_len, []).append(request)

    for group in groups.values():
        if len(group) == 1:
            request = group[0]
            results_by_id[id(request)] = generate_one_token(
                request.input_ids, request.past_key_values
            )
            continue

        batched_input = torch.cat([r.input_ids[:, -1:] for r in group], dim=0)
        batched_past = stack_past_key_values([r.past_key_values for r in group])

        with torch.inference_mode():
            output = model(
                batched_input,
                past_key_values=batched_past,
                use_cache=True,
            )

        per_request_caches = unstack_past_key_values(output.past_key_values, len(group))

        for i, request in enumerate(group):
            next_token_id = torch.argmax(output.logits[i : i + 1, -1, :], dim=-1, keepdim=True)
            results_by_id[id(request)] = _build_step_result(
                request.input_ids, next_token_id, per_request_caches[i]
            )

    return [results_by_id[id(request)] for request in requests]


def generate_tokens(prompt: str, max_tokens: int):
    input_ids = prepare_request(prompt)
    past_key_values = None

    for _ in range(max_tokens):
        result = generate_one_token(input_ids, past_key_values)
        input_ids = result.input_ids
        past_key_values = result.past_key_values
        if result.is_done:
            break
        yield result.token_text
