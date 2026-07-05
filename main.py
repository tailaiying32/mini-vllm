import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers.cache_utils import DynamicCache

from engine import (
    StepResult,
    prepare_request,
    batch_decode_one_token,
    batch_prefill_one_token,
)

INFERENCE_EXECUTOR = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan for the application.
    This function starts the engine loop when the application starts and stops it when it stops.
    """
    engine_task = asyncio.create_task(engine_loop())
    try:
        yield
    finally:
        engine_task.cancel()
        try:
            await engine_task
        except asyncio.CancelledError:
            pass
        INFERENCE_EXECUTOR.shutdown(wait=False)


app = FastAPI(title="Mini-vLLM", lifespan=lifespan)
REQUEST_QUEUE = asyncio.Queue()


class RequestPhase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class CompletionRequest(BaseModel):
    """
    Request for a completion from the model.
    """
    prompt: str
    max_tokens: int = 100
    stream: bool = True
    model: str | None = None


class CompletionChoice(BaseModel):
    text: str = ""
    finish_reason: str | None = None


class CompletionChunk(BaseModel):
    choices: list[CompletionChoice]


@dataclass
class InferenceRequest:
    """
    Request for an inference from the model.
    """

    request_id: str
    prompt: str
    max_tokens: int
    output_queue: asyncio.Queue
    input_ids: torch.Tensor
    past_key_values: DynamicCache | None = None
    phase: RequestPhase = RequestPhase.PREFILL
    generated_count: int = 0
    finished: bool = False
    cancelled: bool = False


def _sse_chunk(text: str = "", finish_reason: str | None = None) -> str:
    chunk = CompletionChunk(choices=[CompletionChoice(text=text, finish_reason=finish_reason)])
    return f"data: {chunk.model_dump_json()}\n\n"


async def stream_tokens(output_queue: asyncio.Queue, raw_request: Request, inference_request: InferenceRequest):
    """
    Stream tokens from the output queue, polling for client disconnect.
    """
    while True:
        if await raw_request.is_disconnected():
            inference_request.cancelled = True
            break

        try:
            token = await asyncio.wait_for(output_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue

        if token is None:
            break
        yield _sse_chunk(text=token)

    if not inference_request.cancelled:
        finish_reason = "stop" if inference_request.generated_count < inference_request.max_tokens else "length"
        yield _sse_chunk(finish_reason=finish_reason)
    yield "data: [DONE]\n\n"


async def apply_step_result(request: InferenceRequest, result: StepResult):
    request.input_ids = result.input_ids
    request.past_key_values = result.past_key_values
    request.phase = RequestPhase.DECODE

    if result.token_text is not None:
        await request.output_queue.put(result.token_text)
        request.generated_count += 1

    if result.is_done or request.generated_count >= request.max_tokens:
        request.finished = True
        await request.output_queue.put(None)


async def run_inference(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(INFERENCE_EXECUTOR, fn, *args)


@app.post("/v1/completions")
async def completions(request: CompletionRequest, raw_request: Request):
    """
    Handle a completion request.
    """
    if not request.stream:
        output_queue = asyncio.Queue()
        inference_request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            output_queue=output_queue,
            input_ids=prepare_request(request.prompt),
        )
        await REQUEST_QUEUE.put(inference_request)

        tokens: list[str] = []
        while True:
            if await raw_request.is_disconnected():
                inference_request.cancelled = True
                break
            try:
                token = await asyncio.wait_for(output_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            if token is None:
                break
            tokens.append(token)
        return {
            "choices": [{"text": "".join(tokens)}],
            "model": request.model or "Qwen/Qwen2.5-0.5B-Instruct",
        }

    output_queue = asyncio.Queue()
    inference_request = InferenceRequest(
        request_id=str(uuid.uuid4()),
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        output_queue=output_queue,
        input_ids=prepare_request(request.prompt),
    )

    await REQUEST_QUEUE.put(inference_request)
    return StreamingResponse(
        stream_tokens(output_queue, raw_request, inference_request),
        media_type="text/event-stream",
    )


async def engine_loop():
    """
    Continuous batching scheduler: batched decode + batched prefill per cycle.
    """
    active_requests: list[InferenceRequest] = []

    while True:
        while not REQUEST_QUEUE.empty():
            active_requests.append(await REQUEST_QUEUE.get())

        if not active_requests:
            await asyncio.sleep(0.001)
            continue

        prefill_requests = [
            r for r in active_requests
            if not r.finished and not r.cancelled and r.phase == RequestPhase.PREFILL
        ]
        decode_requests = [
            r for r in active_requests
            if not r.finished and not r.cancelled and r.phase == RequestPhase.DECODE
        ]

        if decode_requests:
            results = await run_inference(batch_decode_one_token, decode_requests)
            for request, result in zip(decode_requests, results):
                if not request.cancelled:
                    await apply_step_result(request, result)

        if prefill_requests:
            results = await run_inference(batch_prefill_one_token, prefill_requests)
            for request, result in zip(prefill_requests, results):
                if not request.cancelled:
                    await apply_step_result(request, result)

        active_requests = [
            r for r in active_requests if not r.finished and not r.cancelled
        ]
        await asyncio.sleep(0)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
