"""Benchmark sequential and concurrent inference throughput."""

import argparse
import asyncio
import json
import time

import httpx

from engine import generate_tokens, prepare_request, prefill_one_token, batch_prefill_one_token


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "The meaning of life is",
    "In the year 2050",
    "Python is a programming language that",
]


class _PrefillRequest:
    """Minimal stand-in for InferenceRequest during engine benchmarks."""

    def __init__(self, prompt: str):
        self.prompt = prompt
        self.input_ids = prepare_request(prompt)


def benchmark_engine(num_prompts: int, max_tokens: int) -> None:
    prompts = (DEFAULT_PROMPTS * ((num_prompts // len(DEFAULT_PROMPTS)) + 1))[:num_prompts]

    start = time.perf_counter()
    total_tokens = 0

    for prompt in prompts:
        for token in generate_tokens(prompt, max_tokens):
            if token is not None:
                total_tokens += 1

    elapsed = time.perf_counter() - start
    print(f"Mode:        engine (sequential)")
    print(f"Prompts:     {num_prompts}")
    print(f"Max tokens:  {max_tokens}")
    print(f"Total tokens:{total_tokens}")
    print(f"Elapsed:     {elapsed:.2f}s")
    print(f"Throughput:  {total_tokens / elapsed:.1f} tokens/sec")


def benchmark_batched_prefill(num_prompts: int) -> None:
    """Compare batched vs sequential prefill output."""
    prompts = (DEFAULT_PROMPTS * ((num_prompts // len(DEFAULT_PROMPTS)) + 1))[:num_prompts]
    requests = [_PrefillRequest(p) for p in prompts]

    sequential = [prefill_one_token(r.input_ids) for r in requests]
    batched = batch_prefill_one_token(requests)

    for i, (seq, bat) in enumerate(zip(sequential, batched)):
        seq_ids = seq.input_ids[0].tolist()
        bat_ids = bat.input_ids[0].tolist()
        assert seq_ids == bat_ids, f"Prefill mismatch on prompt {i}: {prompts[i]!r}"

    print(f"Batched prefill parity: OK ({num_prompts} prompts)")


async def benchmark_http(num_prompts: int, max_tokens: int, base_url: str) -> None:
    prompts = (DEFAULT_PROMPTS * ((num_prompts // len(DEFAULT_PROMPTS)) + 1))[:num_prompts]

    async def run_one(client: httpx.AsyncClient, prompt: str) -> tuple[int, float]:
        ttft = None
        tokens = 0
        t0 = time.perf_counter()

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
                text = chunk["choices"][0].get("text", "")
                if text and ttft is None:
                    ttft = time.perf_counter() - t0
                if text:
                    tokens += 1

        elapsed = time.perf_counter() - t0
        return tokens, ttft or elapsed, elapsed

    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[run_one(client, p) for p in prompts])

    total_tokens = sum(r[0] for r in results)
    ttfts = [r[1] for r in results]
    elapsed = time.perf_counter() - start

    print(f"Mode:        http (concurrent)")
    print(f"Prompts:     {num_prompts}")
    print(f"Max tokens:  {max_tokens}")
    print(f"Total tokens:{total_tokens}")
    print(f"Elapsed:     {elapsed:.2f}s")
    print(f"Throughput:  {total_tokens / elapsed:.1f} tokens/sec")
    print(f"Avg TTFT:    {sum(ttfts) / len(ttfts):.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark mini-vLLM inference")
    parser.add_argument(
        "--mode",
        choices=["engine", "http", "prefill"],
        default="engine",
        help="Benchmark mode",
    )
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    if args.mode == "engine":
        benchmark_engine(args.num_prompts, args.max_tokens)
    elif args.mode == "prefill":
        benchmark_batched_prefill(args.num_prompts)
    else:
        asyncio.run(benchmark_http(args.num_prompts, args.max_tokens, args.base_url))


if __name__ == "__main__":
    main()
