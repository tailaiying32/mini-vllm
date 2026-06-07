import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# load tokenizer and model weights into memory, set model to evaluation mode
print(f"Loading model from {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f"Tokenizer loaded")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
print(f"Model loaded with device {device}")
model.eval()
print(f"Model in evaluation mode")


# prompt
prompt = "The capital of France is"
max_tokens = 50

# encode prompt to tokens (tensor of shape [1, num_tokens]), move tokens to device
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

with torch.inference_mode(): # disable gradient computation, faster inference
    for i in range(max_tokens):
        output = model(input_ids) # predict next possible tokens
        next_token_logits = output.logits[:, -1, :] # model's scores for possible next tokens
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True) # most likely next token
        input_ids = torch.cat([input_ids, next_token_id], dim=-1) # add new token to input

        if next_token_id.item() == tokenizer.eos_token_id: # stop if EOS token is generated
            break

generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True) # decode tokens to text
print(generated_text)

