import asyncio
import uuid
from dataclasses import dataclass
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch

from engine import prepare_request, generate_one_token

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

# define the FastAPI app with the REQUEST_QUEUE as a global variable
app = FastAPI(title="Mini-vLLM", lifespan=lifespan)
REQUEST_QUEUE = asyncio.Queue()

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
    past_key_values: tuple | None = None
    generated_count: int = 0
    finished: bool = False


async def stream_tokens(output_queue: asyncio.Queue):
    """
    Stream tokens from the output queue.

    @param output_queue: The queue to stream tokens from.
    @return: A generator that yields tokens.
    """

    while True:
        token = await output_queue.get()
        if token is None:
            break
        yield token


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    Handle a completion request. 
    This function creates a new inference request and adds it to the request queue.
    It then streams the tokens from the output queue to the client.

    @param request: The completion request.
    @return: A streaming response of tokens.
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
    Round-robin scheduler: advance each active request by one token per cycle.
    """
    active_requests: list[InferenceRequest] = []

    while True:
        while not REQUEST_QUEUE.empty():
            active_requests.append(await REQUEST_QUEUE.get())

        if not active_requests:
            await asyncio.sleep(0.001)
            continue

        still_active = []

        for request in active_requests:
            if request.finished:
                continue

            if request.generated_count >= request.max_tokens:
                request.finished = True
                await request.output_queue.put(None)
                continue

            request.input_ids, token_text, is_done, request.past_key_values = generate_one_token(
                request.input_ids,
                request.past_key_values,
            )

            if token_text is not None:
                await request.output_queue.put(token_text)
                request.generated_count += 1

            if is_done or request.generated_count >= request.max_tokens:
                request.finished = True
                await request.output_queue.put(None)
            else:
                still_active.append(request)

            await asyncio.sleep(0)

        active_requests = still_active


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
