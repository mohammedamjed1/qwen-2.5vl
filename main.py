from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForConditionalGeneration
import torch, uvicorn

# اسم النموذج من Hugging Face
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# إنشاء تطبيق FastAPI
app = FastAPI(title="EVO AI - Qwen2.5VL", version="1.0")

print("🔄 Loading model and tokenizer...")

# تحميل الـ tokenizer والموديل
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

print("✅ Model loaded successfully!")

# نقطة اختبار بسيطة
@app.get("/")
async def root():
    return {"status": "ready", "model": MODEL_NAME}

# استقبال الأسئلة (POST)
@app.post("/run")
async def run(request: Request):
    data = await request.json()
    prompt = data.get("prompt") or data.get("message") or ""

    if not prompt:
        return {"error": "No prompt provided."}

    # ترميز النص
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # توليد الإجابة
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True
    )

    # فك الترميز
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"output": result}

# تشغيل التطبيق داخل RunPod
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
