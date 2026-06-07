import asyncio
import uuid
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI(title="Mini-vLLM")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading model {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)

print(f"Model loaded on {device}")

REQUEST_QUEUE = asyncio.Queue()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 50



