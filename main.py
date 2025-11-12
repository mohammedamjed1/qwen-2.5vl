from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, uvicorn

# إعداد FastAPI
app = FastAPI(title="EVO AI - Qwen2.5VL", version="1.0")

# تحميل النموذج
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

print("🔄 Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",        # يوزّع تلقائيًا على الـ GPU
    trust_remote_code=True,   # ضروري لـ Qwen
)

print("✅ Model loaded successfully!")

# نقطة اختبار سريعة (GET)
@app.get("/")
async def root():
    return {"status": "ready", "model": model_name}

# استقبال الأسئلة (POST)
@app.post("/run")
async def run(request: Request):
    data = await request.json()
    prompt = data.get("prompt") or data.get("message") or ""

    if not prompt:
        return {"error": "No prompt provided."}

    # ترميز النص
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # التوليد
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True
    )

    # فك الترميز
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"output": result}

# تشغيل التطبيق محلياً داخل RunPod
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
