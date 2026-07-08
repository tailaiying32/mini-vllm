"""Verify engine and server output matches naive greedy generation."""

import argparse
import asyncio
import json
import sys

import httpx
import torch

from engine import (
    MODEL_ID,
    batch_decode_one_token,
    batch_prefill_one_token,
    generate_one_token,
    generate_tokens,
    model,
    prepare_request,
    prefill_one_token,
    tokenizer,
)


class _PrefillRequest:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.input_ids = prepare_request(prompt)


class _DecodeRequest:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.input_ids = prepare_request(prompt)
        result = prefill_one_token(self.input_ids)
        self.input_ids = result.input_ids
        self.kv_state = result.kv_state


def naive_greedy(prompt: str, max_tokens: int) -> str:
    input_ids = prepare_request(prompt)

    with torch.inference_mode():
        for _ in range(max_tokens):
            output = model(input_ids)
            next_token_id = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def verify_engine_parity(prompt: str, max_tokens: int) -> None:
    naive_text = naive_greedy(prompt, max_tokens)
    engine_text = prompt + "".join(generate_tokens(prompt, max_tokens))

    if naive_text != engine_text:
        print(f"FAIL engine parity")
        print(f"  prompt:   {prompt!r}")
        print(f"  naive:    {naive_text!r}")
        print(f"  engine:   {engine_text!r}")
        sys.exit(1)

    print(f"OK engine parity: {prompt!r}")


def verify_batched_prefill(prompts: list[str]) -> None:
    requests = [_PrefillRequest(p) for p in prompts]
    sequential = [prefill_one_token(r.input_ids) for r in requests]
    batched = batch_prefill_one_token(requests)

    for prompt, seq, bat in zip(prompts, sequential, batched):
        if seq.input_ids[0].tolist() != bat.input_ids[0].tolist():
            print(f"FAIL batched prefill: {prompt!r}")
            sys.exit(1)

    print(f"OK batched prefill parity ({len(prompts)} prompts)")


def verify_heterogeneous_decode() -> None:
    """Different sequence lengths in one decode batch must match sequential decode."""
    prompts = ["Hi", "The capital of France is", "Once upon a time in a land far away"]

    sequential_ids = []
    for prompt in prompts:
        request = _DecodeRequest(prompt)
        result = generate_one_token(request.input_ids, request.kv_state)
        sequential_ids.append(result.input_ids[0].tolist())

    batched_requests = [_DecodeRequest(p) for p in prompts]
    batched = batch_decode_one_token(batched_requests)
    for i, (prompt, result) in enumerate(zip(prompts, batched)):
        if result.input_ids[0].tolist() != sequential_ids[i]:
            print(f"FAIL heterogeneous decode on prompt {prompt!r}")
            sys.exit(1)

    print(f"OK heterogeneous decode parity ({len(prompts)} prompts)")


async def verify_http_parity(prompt: str, max_tokens: int, base_url: str) -> None:
    naive_text = naive_greedy(prompt, max_tokens)
    streamed_text = prompt

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{base_url}/v1/completions",
            json={"prompt": prompt, "max_tokens": max_tokens, "stream": True},
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                streamed_text += chunk["choices"][0].get("text", "")

    if naive_text != streamed_text:
        print(f"FAIL http parity")
        print(f"  prompt:   {prompt!r}")
        print(f"  naive:    {naive_text!r}")
        print(f"  http:     {streamed_text!r}")
        sys.exit(1)

    print(f"OK http parity: {prompt!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify mini-vLLM output parity")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--http", action="store_true", help="Also test HTTP endpoint")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    print(f"Model: {MODEL_ID}")
    verify_engine_parity(args.prompt, args.max_tokens)
    verify_batched_prefill([
        "The capital of France is",
        "Once upon a time",
        "Hello",
    ])
    verify_heterogeneous_decode()

    if args.http:
        asyncio.run(verify_http_parity(args.prompt, args.max_tokens, args.base_url))

    print("All parity checks passed.")


if __name__ == "__main__":
    main()
