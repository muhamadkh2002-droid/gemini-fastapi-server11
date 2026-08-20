import os
import psutil
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from google import genai
from google.genai.errors import APIError

app = FastAPI(title="Gemini Proxy Server")

# 1. التحقق من مفتاح البيئة
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("مفتاح GEMINI_API_KEY غير معرف في متغيّرات البيئة.")

client = genai.Client(api_key=api_key)

# 2. النموذج الأساسي المطلوب
MODEL_NAME = "gemini-3.6-flash"

# نماذج البيانات المدخلة والمخرجة
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    reset_session: bool = False

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

# تخزين جلسات المحادثة في الذاكرة
user_sessions = {}

@app.get("/api/status")
def get_status():
    return {
        "status": "طبيعي",
        "is_busy": False,
        "concurrent_requests_now": 0,
        "active_users_5min": len(user_sessions),
        "resources": {
            "cpu_usage": f"{psutil.cpu_percent()}%",
            "ram_usage": f"{psutil.virtual_memory().percent}%"
        }
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        # إنشاء جلسة جديدة إذا طلب المستخدم أو إذا لم تكن موجودة
        if request.reset_session or request.user_id not in user_sessions:
            user_sessions[request.user_id] = client.chats.create(model=MODEL_NAME)
        
        chat = user_sessions[request.user_id]
        
        # إرسال الرسالة إلى Gemini
        response = chat.send_message(request.message)
        
        return ChatResponse(response=response.text)

    except APIError as e:
        # إذا لم يكن النموذج متوفراً أو حدث خطأ من API، محاولة الاتصال بالنموذج البديل
        try:
            fallback_chat = client.chats.create(model="gemini-2.0-flash")
            response = fallback_chat.send_message(request.message)
            user_sessions[request.user_id] = fallback_chat
            return ChatResponse(response=response.text)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"خطأ في الاتصال بـ Gemini API: {str(e)}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"خطأ داخلي في السيرفر: {str(e)}"
        )
