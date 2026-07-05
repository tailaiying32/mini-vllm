import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
model.eval()


def prepare_request(prompt: str) -> torch.Tensor:
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


def generate_one_token(
    input_ids: torch.Tensor,
    past_key_values: tuple | None = None,
) -> tuple[torch.Tensor, str | None, bool, tuple]:
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

        next_token_logits = output.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        updated_input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        updated_cache = output.past_key_values

        if next_token_id.item() == tokenizer.eos_token_id:
            return updated_input_ids, None, True, updated_cache

        token_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
        return updated_input_ids, token_text, False, updated_cache


def generate_tokens(prompt: str, max_tokens: int):
    input_ids = prepare_request(prompt)
    past_key_values = None

    for _ in range(max_tokens):
        input_ids, token_text, is_done, past_key_values = generate_one_token(
            input_ids, past_key_values
        )
        if is_done:
            break
        yield token_text
