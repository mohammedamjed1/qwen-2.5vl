from fastapi import FastAPI, Request
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

app = FastAPI(title="EVO AI - Qwen2.5-VL")

# تحميل النموذج من Hugging Face
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-VL",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    inputs = processor(text=user_message, images=None, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    reply = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return {"reply": reply}
