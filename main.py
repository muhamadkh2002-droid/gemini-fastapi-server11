import base64
import os
import psutil
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import APIError

app = FastAPI(title="Gemini Proxy Server")

# 1. إضافة middleware لـ CORS للسماح بالطلبات من واجهة التطبيق
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. التحقق من مفتاح البيئة
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("مفتاح GEMINI_API_KEY غير معرف في متغيّرات البيئة.")

client = genai.Client(api_key=api_key)

# 3. النموذج الأساسي لدعم النصوص والصور
MODEL_NAME = "gemini-2.5-flash"

# نماذج البيانات المدخلة والمخرجة
class ChatRequest(BaseModel):
    message: str
    image_base64: Optional[str] = None  # نص الصورة المشفر بـ Base64 (اختياري)
    mime_type: Optional[str] = "image/jpeg"  # نوع الصورة (image/png, image/jpeg, ...)
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
        
        # تجهيز قائمة محتويات الرسالة
        contents = []

        # في حال تم إرسال صورة مع الطلب
        if request.image_base64:
            # فك تشفير الصورة وتحويلها إلى Part مناسب للـ SDK الجديد
            image_bytes = base64.b64decode(request.image_base64)
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=request.mime_type
            )
            contents.append(image_part)

        # إضافة نص الرسالة/طلب التعديل
        contents.append(request.message)
        
        # إرسال الرسالة والصورة إلى Gemini
        response = chat.send_message(contents)
        
        return ChatResponse(response=response.text)

    except APIError as e:
        # محاولة الاتصال بالنموذج البديل في حال تعثر النموذج الرئيسي
        try:
            fallback_chat = client.chats.create(model="gemini-2.5-flash")
            
            contents = []
            if request.image_base64:
                image_bytes = base64.b64decode(request.image_base64)
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type=request.mime_type))
            contents.append(request.message)

            response = fallback_chat.send_message(contents)
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
