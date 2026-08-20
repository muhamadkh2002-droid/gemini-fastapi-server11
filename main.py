import os
import time
import logging
import asyncio
import psutil
from typing import Dict, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai.errors import APIError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("GeminiGlobalServer")

app = FastAPI(title="Gemini High-Concurrency API Server")

# جلب مفتاح الـ API من متغيرات البيئة (Environment Variables)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

# إدارة الجلسات والأقفال
user_sessions: Dict[str, Tuple[genai.chats.AsyncChat, asyncio.Lock]] = {}
global_session_lock = asyncio.Lock()


async def get_or_create_user_session(user_id: str, model_name: str) -> Tuple[genai.chats.AsyncChat, asyncio.Lock]:
    async with global_session_lock:
        if user_id not in user_sessions:
            logger.info(f"Creating Async Chat Session for User [{user_id}]")
            chat = client.aio.chats.create(model=model_name)
            user_lock = asyncio.Lock()
            user_sessions[user_id] = (chat, user_lock)
        return user_sessions[user_id]


class ServerMetrics:
    def __init__(self):
        self.active_requests: int = 0
        self.user_last_active: Dict[str, float] = {}

    def record_activity(self, user_id: str):
        self.user_last_active[user_id] = time.time()

    def get_active_users_count(self, window_seconds: int = 300) -> int:
        now = time.time()
        return sum(1 for last_time in self.user_last_active.values() if now - last_time <= window_seconds)


metrics = ServerMetrics()


class UserChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    reset_session: bool = False


@app.post("/api/chat")
async def chat_endpoint(request: UserChatRequest):
    if not client:
        raise HTTPException(
            status_code=500,
            detail="Server API Key is not configured in Environment Variables."
        )

    metrics.active_requests += 1
    metrics.record_activity(request.user_id)

    try:
        if request.reset_session:
            async with global_session_lock:
                if request.user_id in user_sessions:
                    del user_sessions[request.user_id]

        # قائمة النماذج حسب الأولوية لتفادي خطأ 429
        fallback_models = ["gemini-2.5-flash", "gemini-1.5-flash"]
        response_text = None
        last_error = None

        for model_name in fallback_models:
            try:
                chat_session, user_lock = await get_or_create_user_session(request.user_id, model_name)

                async with user_lock:
                    # إعادة المحاولة في حال وجود ضغط مؤقت (Rate Limit)
                    for attempt in range(3):
                        try:
                            response = await chat_session.send_message(request.message)
                            response_text = response.text
                            break
                        except APIError as e:
                            if "429" in str(e) and attempt < 2:
                                logger.warning(
                                    f"Rate limited (429) for user {request.user_id}. Retrying in {attempt + 1}s..."
                                )
                                await asyncio.sleep(1.5 * (attempt + 1))
                            else:
                                raise e

                if response_text:
                    break

            except Exception as e:
                logger.warning(f"Model {model_name} failed: {str(e)}")
                last_error = e

        if not response_text:
            raise last_error or Exception("تعذر معالجة الطلب حالياً بسبب الضغط العالي.")

        return {
            "status": "success",
            "user_id": request.user_id,
            "response": response_text
        }

    except Exception as e:
        logger.error(f"Error serving User [{request.user_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطأ في السيرفر: {str(e)}")

    finally:
        metrics.active_requests -= 1


@app.get("/api/status")
async def server_status():
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    is_busy = metrics.active_requests > 20 or cpu > 80.0 or ram > 85.0

    return {
        "status": "مضغوط" if is_busy else "طبيعي",
        "is_busy": is_busy,
        "concurrent_requests_now": metrics.active_requests,
        "active_users_5min": metrics.get_active_users_count(300),
        "resources": {"cpu_usage": f"{cpu}%", "ram_usage": f"{ram}%"}
    }


if __name__ == "__main__":
    import uvicorn

    # قراءة المنافذ المخصصة من الاستضافة آلياً أو الاعتماد على 8000 محلياً
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)