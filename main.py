import os
import json
import re
import uuid
import asyncio
from datetime import datetime
from typing import Optional, List, Dict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

# ------------------------------
# استخدام الحزمة الجديدة من Google
# ------------------------------
from google import genai
from google.genai import types

load_dotenv()

# ------------------------------
# التأكد من وجود ملفات الإعدادات بالقيم الافتراضية
# ------------------------------
def ensure_config_files():
    # config.json
    if not os.path.exists("config.json"):
        default_config = {
            "version": "1.0",
            "model": {"name": "gemini-1.5-flash"},
            "storage": {"conversations_dir": "conversations"},
            "rag": {"enabled": True, "top_k": 3},
            "safety": {"block_prompt_injection": True}
        }
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)

    # prompts.json
    if not os.path.exists("prompts.json"):
        default_prompts = {
            "system_prompt_default": "أنت مرشد مهني خبير في العالم العربي. أجب باختصار وبأسلوب مفيد.",
            "system_prompt_with_rag": "استخدم المعلومات التالية للمساعدة:\n{context}\n\n{instructions}",
            "prompt_injection_blocked": "عذرًا، لا يمكنني تجاوز تعليماتي الأمنية.",
            "fallback": "عذرًا، لم أستطع فهم سؤالك. حاول مرة أخرى."
        }
        with open("prompts.json", "w", encoding="utf-8") as f:
            json.dump(default_prompts, f, ensure_ascii=False, indent=2)

    # knowledge_base.json
    if not os.path.exists("knowledge_base.json"):
        default_kb = {
            "documents": [
                {"content": "الهندسة: رواتب عالية، طلب كبير على المبرمجين والمطورين."},
                {"content": "الطب: سنوات دراسة طويلة لكن مكانة اجتماعية مرتفعة."},
                {"content": "التسويق الرقمي: فرص عمل عن بعد، يناسب المبدعين."},
                {"content": "جامعة القاهرة: حكومية، تكلفة منخفضة، سمعة قوية."},
                {"content": "كورس Python: لغة سهلة، تستخدم في الذكاء الاصطناعي."}
            ]
        }
        with open("knowledge_base.json", "w", encoding="utf-8") as f:
            json.dump(default_kb, f, ensure_ascii=False, indent=2)

ensure_config_files()

# ------------------------------
# تحميل الإعدادات
# ------------------------------
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

with open("prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

with open("knowledge_base.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# ------------------------------
# إعداد عميل Gemini الجديد
# ------------------------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("الرجاء وضع مفتاح GEMINI_API_KEY في ملف .env")

client = genai.Client(api_key=api_key)
model_name = config["model"]["name"]

# ------------------------------
# FastAPI App
# ------------------------------
app = FastAPI(title="PathFinder AI", version=config["version"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# التخزين
# ------------------------------
def ensure_conversation_dir():
    os.makedirs(config["storage"]["conversations_dir"], exist_ok=True)

def save_conversation(session_id: str, messages: List[Dict]):
    ensure_conversation_dir()
    file_path = os.path.join(config["storage"]["conversations_dir"], f"{session_id}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump({
            "session_id": session_id,
            "messages": messages,
            "updated_at": datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)

def load_conversation(session_id: str) -> List[Dict]:
    file_path = os.path.join(config["storage"]["conversations_dir"], f"{session_id}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f).get("messages", [])
    return []

# ------------------------------
# الأمان
# ------------------------------
def sanitize_pii(text: str) -> str:
    text = re.sub(r'\b01[0-9]{9}\b', '[رقم هاتف]', text)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[بريد إلكتروني]', text)
    return text

def detect_prompt_injection(text: str) -> bool:
    attacks = [
        "ignore previous instructions", "system prompt", "you are now",
        "forget your role", "تجاهل التعليمات", "أنت الآن"
    ]
    return any(x in text.lower() for x in attacks)

# ------------------------------
# RAG
# ------------------------------
def retrieve_context(query: str, top_k: int = 3) -> str:
    if not config.get("rag", {}).get("enabled", True):
        return ""
    words = set(query.lower().split())
    scored = []
    for doc in knowledge_base.get("documents", []):
        content = doc.get("content", "").lower()
        score = sum(1 for w in words if w in content)
        if score:
            scored.append((score, doc["content"]))
    scored.sort(reverse=True, key=lambda x: x[0])
    return "\n".join([c for _, c in scored[:top_k]])

# ------------------------------
# Models
# ------------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class CompareRequest(BaseModel):
    item1: str
    item2: str
    type: str

# ------------------------------
# دوال الذكاء الاصطناعي باستخدام google.genai
# ------------------------------
async def generate_reply(session_id: str, user_message: str, history: List[Dict]) -> str:
    if config.get("safety", {}).get("block_prompt_injection", True) and detect_prompt_injection(user_message):
        return prompts.get("prompt_injection_blocked", "عذرًا، لا يمكنني تجاوز التعليمات.")

    clean = sanitize_pii(user_message)
    context = retrieve_context(clean)

    if context:
        system = prompts.get("system_prompt_with_rag", "").format(
            context=context,
            instructions="كن دقيقاً."
        )
    else:
        system = prompts.get("system_prompt_default", "أنت مرشد مهني.")

    # بناء تاريخ المحادثة (بالنظام الجديد)
    chat_history = []
    for msg in history[-10:]:
        role = "user" if msg["role"] == "user" else "model"
        chat_history.append({"role": role, "parts": [msg["content"]]})

    # بدء جلسة دردشة جديدة
    chat = client.chats.create(model=model_name, history=chat_history)
    full_prompt = f"{system}\n\nالمستخدم: {clean}"

    try:
        response = await asyncio.to_thread(chat.send_message, full_prompt)
        return response.text.strip() if response.text else prompts.get("fallback", "لم أستطع الرد.")
    except Exception as e:
        return f"❌ خطأ تقني: {str(e)}"

async def generate_comparison(item1: str, item2: str, comp_type: str) -> str:
    templates = {
        "career": "قارن بين: {item1} و {item2} من حيث العمل والراتب والمستقبل.",
        "university": "قارن بين: {item1} و {item2} من حيث الجودة والتكلفة.",
        "course": "قارن بين: {item1} و {item2} من حيث المحتوى والسعر.",
        "job": "قارن بين: {item1} و {item2} من حيث المهام والراتب."
    }
    prompt = templates.get(comp_type, templates["career"]).format(item1=item1, item2=item2)

    try:
        response = await asyncio.to_thread(client.models.generate_content, model=model_name, contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ خطأ في المقارنة: {str(e)}"

# ------------------------------
# Endpoints (كما هي ولكن مع دفق الإجابات)
# ------------------------------
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    history = load_conversation(sid)

    history.append({
        "role": "user",
        "content": req.message,
        "timestamp": datetime.now().isoformat()
    })
    reply = await generate_reply(sid, req.message, history[:-1])

    history.append({
        "role": "assistant",
        "content": reply,
        "timestamp": datetime.now().isoformat()
    })
    save_conversation(sid, history)
    return {"session_id": sid, "reply": reply}

@app.post("/compare")
async def compare_endpoint(req: CompareRequest):
    result = await generate_comparison(req.item1, req.item2, req.type)
    return {"result": result}

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    return {"messages": load_conversation(session_id)}

@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    path = os.path.join(config["storage"]["conversations_dir"], f"{session_id}.json")
    if os.path.exists(path):
        os.remove(path)
    return {"status": "deleted"}

@app.get("/", response_class=HTMLResponse)
async def home():
    if not os.path.exists("index.html"):
        return HTMLResponse("<h2>الملف index.html غير موجود</h2>", status_code=404)
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ------------------------------
# تشغيل السيرفر
# ------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)