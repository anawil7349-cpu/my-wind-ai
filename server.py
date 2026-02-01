import os
import json
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

# =====================================================
# 1. ตั้งค่าและเตรียมระบบ
# =====================================================
app = Flask(__name__)
CORS(app)

print("🔄 กำลังเริ่มระบบ AI Data Scientist Server (Web Mode)...")

# --- ส่วนแก้ไขกุญแจ (แบบ Auto-Fix) ---
raw_key = "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDAdQ71VIcSz6tq\n3CKL7T6PjedlbPcoJnk2XjCe5+uPFK83G7B94xi0h2EAtG0AOFmeVICCLWm9gkfB\nhmCJBFBHHPcafswYMVAHETD2B5vjd/dZkiCqgLPT8BThJdDw7DDHw83Rv2bhRXSN\n79+TdjHOfCL4A0hnQ41HWgNHj4KWJoCnbP3IVPYB5dkLYqkz4Uw3dE0cOX8/Nd4k\nz3SbXoj95JRY974oiBLNoohjHtzvqdZG0HZ/0tq34VK5zD9vV9FlhlVxX4BrP3m0\nFi9YiDprrQVeKSjPCb75V8pYF4/zUkzQZ83l3EIKYZ3DtfvpCkcavfaDSg9RoEj0\nveOnCcZlAgMBAAECggEAAhLAwH/SnK9EB3irnppFrEI5FeyglPwlHiLRn0ScUwRE\nBvHzasfBgmBa+Sj4a6IvxPbgE4bttq7qmvkZnSBAxSNYvh5TkIcnd4wF3QCj+0VV\nks9yLqQIS+YwM2S25YGF3QEM/I91SkP3R3goDmydiL3pmoZeh05A/V3I30J6g6eN\nvvUccK4V3yoY0lN5kDRKsRkfwfmB4qg5ULi1F7tv1OoCvlJqXFq7fCVExr+A/4yM\nanTgUpovIWdAcGx1HD+muI5Rn4XJuKXGosv7++EjIAgOxgysZV6w5YMPpoRotuip\n5kVeI8G0D+zi7vnmmgSWloEXeJP8mpt+RoTYjlZtgQKBgQDe1dYIksIgij1w7dpX\nESQ5TpCGEsTu4yNPNaHYjb0wrE1DC1OUO+89QZtf+SZrscI/Wnkf6OSI2nHFr1ha\ncOQCY0TgeG99DvQgduVvDi92AyGDH9p3wOVB9qLljZFslERTEDUuSTzSnA89rv1E\n4u3D2medYV3oU3pnV5/UbFplNQKBgQDdGczXjSI50Zh1BArmMgVFK60TgtURY0j1\nkle9QO2mg7AZV1+/Ce8xVKN4LmbEQmuXSLMoXXSUu+/4fW+2Uwas2URK97NelFFo\n/GvveoeKsVzNoGlc6jaFo7gZKMevlavHX0j0x3edQQO9ruOn0upA3F/I71quZF4g\ns+rno3vycQKBgQDZ3GL32tQlEELlyAYyHbY2uRMfofYcQMHizWLA4ELZ9XtMUySR\nxs8uKpiICoV/wTlSy1ek1QOqsTeOuNI/CiRCGV/bvqPxts8Ddnr2Sv4n+QOouVnU\nvyjlhwbYO8K0T3lFZJE6AayPlLhp7E3+LYecdknbWrh/Ti5cHxVKj+0JCQKBgQDQ\nGvmgJPoC+9GIyj5L/ubQ/VQRmkJb9Fx2r8CfpF5LLYXxxDidgoc9olGey+X0ciP8\np/PhWV1ipSYweDhOnwUYagOKoGyW5/lcXMJnDKhJFbmo3YRubRDWZovgOm8BSFn/\n9SKhKqHeRJR11Af5LV9Jn2MUqJ1sqZGjLFU8o7cFMQKBgEPE9mND9HvYx5lxpbnx\nx3MFUhqz4LiA34+7qVE9N5Lx7j5lpynKBwbHlAdddUdC9Zcmzv0QOpCIDR6BO8Io\noeQDMbmeUzw0En+3Qo6tIRkNzSD92TQvqt0nJ1yKMPged0hoMrU0i8ffdsfwzyFw\nN3wQcAfw8RUN3Eeo5+252gL2\n-----END PRIVATE KEY-----\n"

# ซ่อมตัวอักษรขึ้นบรรทัดใหม่
fixed_key = raw_key.replace('\\n', '\n')

key_dict = {
  "type": "service_account",
  "project_id": "win-assistant-462002",
  "private_key_id": "8bd1899625d7cb5da8d9af585f2fa919999df02c",
  "private_key": fixed_key,
  "client_email": "firebase-adminsdk-fbsvc@win-assistant-462002.iam.gserviceaccount.com",
  "client_id": "115508101362044082902",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40win-assistant-462002.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

try:
    cred = credentials.Certificate(key_dict)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://win-assistant-462002-default-rtdb.asia-southeast1.firebasedatabase.app'
    })
    print("✅ Firebase Connected! (Auto-Fixed Key)")
except Exception as e:
    print(f"❌ Firebase Error: {e}")

# ตั้งค่า Gemini
GEMINI_API_KEY = "AIzaSyD0xILMuDcMuQBpYUO2G5odNUp_xTDY4u0"
genai.configure(api_key=GEMINI_API_KEY)

# หา Model
valid_model_name = "models/gemini-1.5-flash"
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name:
            valid_model_name = m.name
            break
except: pass
print(f"✅ ใช้โมเดล: {valid_model_name}")

# =====================================================
# 2. โหลดข้อมูล
# =====================================================

df = None 

def refresh_data():
    global df
    print("📥 กำลังซิงค์ข้อมูลจาก Firebase...")
    try:
        ref = db.reference('History')
        data = ref.get()
        if not data: return "Database Empty"

        records = []
        for key, val in data.items():
            if isinstance(val, dict) and 'ts' in val:
                dt = datetime.fromtimestamp(val['ts'] / 1000)
                wind_p = float(val.get('wind', {}).get('p', 0))
                batt_p = float(val.get('batt', {}).get('p', 0))
                wind_v = float(val.get('wind', {}).get('v', 0))
                batt_v = float(val.get('batt', {}).get('v', 0))
                
                records.append({
                    "datetime": dt,
                    "date": dt.strftime("%Y-%m-%d"),
                    "hour": dt.hour,
                    "minute": dt.minute,
                    "wind_p": wind_p,
                    "batt_p": batt_p,
                    "wind_wh": wind_p / 60,
                    "batt_wh": batt_p / 60,
                    "wind_v": wind_v,
                    "batt_v": batt_v
                })
        
        df = pd.DataFrame(records)
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"✅ ข้อมูลพร้อมวิเคราะห์: {len(df)} แถว")
        return f"อัปเดตข้อมูลสำเร็จ มีทั้งหมด {len(df)} รายการ"
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return f"Error loading data: {e}"

refresh_data()

# ฟังก์ชันดึงค่าสด
def get_realtime_string():
    try:
        ref = db.reference('History')
        snapshot = ref.order_by_key().limit_to_last(1).get()
        if not snapshot: return "No Data"
        val = list(snapshot.values())[0]
        w_v = val.get('wind', {}).get('v', 0)
        b_v = val.get('batt', {}).get('v', 0)
        pct = max(0, min(100, ((b_v - 3.2) / (4.2 - 3.2)) * 100))
        return f"Wind: {w_v}V, Batt: {b_v}V ({int(pct)}%)"
    except: return "Error"

# =====================================================
# 3. Code Executor & Memory
# =====================================================

def execute_python_analysis(code_string):
    global df
    print(f"\n[AI Thinking] 🧠 กำลังรันโค้ดวิเคราะห์ข้อมูล...")
    if "import os" in code_string or "import sys" in code_string: return "Security Alert"
    local_vars = {"df": df, "pd": pd, "result": None}
    try:
        exec(code_string, {}, local_vars)
        return str(local_vars.get('result'))
    except Exception as e: return f"Error: {str(e)}"

# แก้ไขส่วนนี้ให้ถูกต้อง (Indent properly)
MEMORY_FILE = "ai_memory.json"
ai_memory = {}
if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "r") as f:
            ai_memory = json.load(f)
    except:
        pass

def remember_info(topic, info):
    ai_memory[topic] = info
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(ai_memory, f)
    except:
        pass
    return f"จำแล้ว: {topic}"

tools_list = [execute_python_analysis, remember_info, refresh_data]

# =====================================================
# 4. Start AI & Server
# =====================================================

model = genai.GenerativeModel(
    model_name=valid_model_name,
    tools=tools_list,
    system_instruction="คุณคือ Data Scientist AI ตอบเป็นภาษาไทย"
)
chat = model.start_chat(enable_automatic_function_calling=True)

@app.route('/ask', methods=['POST'])
def ask_ai():
    try:
        data = request.json
        user_input = data.get('question')
        live_status = get_realtime_string()
        prompt = f"[Realtime: {live_status}] User: {user_input}"
        print(f"User asking: {user_input}")
        response = chat.send_message(prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
