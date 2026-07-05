import asyncio
import uuid
from dataclasses import dataclass
from enum import Enum
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch

from transformers.cache_utils import DynamicCache

from engine import StepResult, prepare_request, prefill_one_token, batch_decode_one_token

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


async def stream_tokens(output_queue: asyncio.Queue):
    """
    Stream tokens from the output queue.
    """
    while True:
        token = await output_queue.get()
        if token is None:
            break
        yield token


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


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    Handle a completion request.
    """
    output_queue = asyncio.Queue()

    inference_request = InferenceRequest(
        request_id=str(uuid.uuid4()),
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        output_queue=output_queue,
        input_ids=prepare_request(request.prompt),
    )

    await REQUEST_QUEUE.put(inference_request)
    return StreamingResponse(stream_tokens(output_queue), media_type="text/plain")


async def engine_loop():
    """
    Continuous batching scheduler: batched decode + one prefill per cycle.
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
            if not r.finished and r.phase == RequestPhase.PREFILL
        ]
        decode_requests = [
            r for r in active_requests
            if not r.finished and r.phase == RequestPhase.DECODE
        ]

        if decode_requests:
            results = batch_decode_one_token(decode_requests)
            for request, result in zip(decode_requests, results):
                await apply_step_result(request, result)

        if prefill_requests:
            request = prefill_requests[0]
            result = prefill_one_token(request.input_ids)
            await apply_step_result(request, result)

        active_requests = [r for r in active_requests if not r.finished]
        await asyncio.sleep(0)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
