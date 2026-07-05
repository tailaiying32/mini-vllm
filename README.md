# mini-vLLM

A minimal LLM inference server inspired by [vLLM](https://github.com/vllm-project/vllm). It demonstrates continuous batching, KV-cache reuse, batched prefill, client disconnect cancellation, and OpenAI-compatible streaming completions.

**Model:** [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (downloaded automatically from HuggingFace on first run).

## Architecture

```
Client
  │  POST /v1/completions (SSE)
  ▼
FastAPI (main.py)
  │  enqueue InferenceRequest
  ▼
REQUEST_QUEUE ──► engine_loop (async scheduler)
                      │
                      ├─► batch_prefill_one_token  (new prompts, one forward pass)
                      └─► batch_decode_one_token   (active requests, 1 token each)
                              │
                              ▼
                         engine.py (PyTorch model + DynamicCache)
                              │
                              ▼
                    per-request output_queue ──► StreamingResponse
```

Each request moves through two phases:

1. **Prefill** — one forward pass over the full prompt; builds the initial KV cache and emits the first generated token.
2. **Decode** — repeated single-token forwards using cached K/V; emits one token per scheduler cycle.

The scheduler runs decode for all active requests first, then batches all pending prefills, enabling concurrent requests to share GPU work.

## Quick Start

### Setup

```bash
# Create and activate a conda environment (optional)
conda create -n mini-vllm python=3.11 -y
conda activate mini-vllm

# Install dependencies
pip install -r requirements.txt
```

On first run, HuggingFace will download ~1 GB of model weights. GPU (CUDA) is used when available; CPU works but is slower.

### Run the server

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Test with curl

```bash
curl -N http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 50, "stream": true}'
```

The response is OpenAI-style Server-Sent Events (SSE):

```
data: {"choices":[{"text":" Paris"}]}

data: {"choices":[{"text":"."}]}

data: {"choices":[{"text":"","finish_reason":"stop"}]}

data: [DONE]

```

Press Ctrl+C mid-stream to test client disconnect cancellation.

Non-streaming mode returns a single JSON object:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 50, "stream": false}'
```

## API

### `POST /v1/completions`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | required | Input text |
| `max_tokens` | int | 100 | Maximum tokens to generate |
| `stream` | bool | true | Stream SSE chunks when true; return JSON when false |
| `model` | string | optional | Ignored; model is fixed |

## How It Works

### Prefill vs decode

- **Prefill** processes the entire prompt in one forward pass. For batched prefill, prompts are left-padded to the same length with an attention mask so shorter prompts ignore pad tokens.
- **Decode** feeds only the last generated token plus the KV cache from prior steps. Multiple requests with equal cache lengths are batched into one forward pass.

### KV cache

Each request stores a HuggingFace `DynamicCache` holding key/value tensors from all prior tokens. This avoids recomputing attention over the full sequence on every decode step.

### Continuous batching

The `engine_loop` scheduler:

1. Drains new requests from the queue
2. Runs **batched decode** for all active decode-phase requests
3. Runs **batched prefill** for all new prefill-phase requests
4. Removes finished or cancelled requests

When a client disconnects, the request is marked cancelled and skipped on the next cycle, freeing its cache.

### Inference threading

PyTorch forwards run in a single-worker thread pool (`asyncio.to_thread`) so GPU work does not block FastAPI's event loop.

## Development / Testing

### Ground-truth baseline

`naive_generate.py` runs greedy generation without a KV cache (full sequence recompute each step). Output should match the server for the same prompt.

```bash
python naive_generate.py
```

### Parity verification

```bash
python verify_parity.py
```

### Benchmarks

Sequential engine-only benchmark:

```bash
python benchmark.py --mode engine --num-prompts 5 --max-tokens 50
```

Concurrent HTTP benchmark (server must be running):

```bash
python benchmark.py --mode http --num-prompts 5 --max-tokens 50
```

- [x] Benchmarks
- [ ] Paged KV cache (stretch goal)
