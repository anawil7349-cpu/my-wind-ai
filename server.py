import os
import json
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime, timedelta  # เพิ่ม timedelta เพื่อจัดการเวลาไทย
import pandas as pd
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

# =====================================================
# 1. ตั้งค่าและเตรียมระบบ (Secure Cloud Mode)
# =====================================================
app = Flask(__name__)
CORS(app) 

print("🔄 กำลังเริ่มระบบ AI Data Scientist Server (Secure Mode)...")

# 🔐 จุดแก้ไขที่ 1: ดึง Key จากระบบแทนการเขียนลงในโค้ด (ป้องกันการโดนแบน)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FIREBASE_CONFIG_JSON = os.environ.get("FIREBASE_SERVICE_ACCOUNT")

# ตั้งค่า Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ตั้งค่า Firebase (รองรับทั้งไฟล์ในเครื่อง และ JSON String บน Cloud)
try:
    if not firebase_admin._apps:
        if FIREBASE_CONFIG_JSON:
            # กรณีรันบน Render/Cloud
            service_account_info = json.loads(FIREBASE_CONFIG_JSON)
            cred = credentials.Certificate(service_account_info)
        elif os.path.exists("serviceAccountKey.json"):
            # กรณีรันในเครื่อง (Local)
            cred = credentials.Certificate("serviceAccountKey.json")
        else:
            cred = None
            print("⚠️ ไม่พบข้อมูลเชื่อมต่อ Firebase")

        if cred:
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://win-assistant-462002-default-rtdb.asia-southeast1.firebasedatabase.app'
            })
            print("✅ Firebase Connected!")
except Exception as e:
    print(f"❌ Firebase Error: {e}")

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
# 2. โหลดข้อมูลเข้า RAM (พร้อมแก้เรื่องเวลา UTC+7)
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
                # 🕒 จุดแก้ไขที่ 2: บังคับเป็นเวลาไทย (UTC+7) เสมอ
                dt = datetime.utcfromtimestamp(val['ts'] / 1000) + timedelta(hours=7)
                
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
        print(f"✅ ข้อมูลพร้อมวิเคราะห์ (เวลาไทย): {len(df)} แถว")
        return f"อัปเดตข้อมูลสำเร็จ มีทั้งหมด {len(df)} รายการ"
    except Exception as e:
        return f"Error loading data: {e}"

if firebase_admin._apps:
    refresh_data()

# =====================================================
# 3. เครื่องมือ Python Code Executor (เหมือนเดิมเป๊ะ)
# =====================================================
def execute_python_analysis(code_string):
    global df
    print(f"\n[AI Thinking] 🧠 กำลังรันโค้ดวิเคราะห์ข้อมูล...")
    if any(forbidden in code_string for forbidden in ["import os", "import sys", "open(", "eval("]):
        return "Security Alert: โค้ดมีความเสี่ยง"
    local_vars = {"df": df, "pd": pd, "result": None}
    try:
        exec(code_string, {}, local_vars)
        output = local_vars.get('result')
        return str(output) if output is not None else "ไม่ได้กำหนดค่าใส่ตัวแปร 'result'"
    except Exception as e:
        return f"Error: {str(e)}"

def get_realtime_string():
    try:
        ref = db.reference('History')
        snapshot = ref.order_by_key().limit_to_last(1).get()
        val = list(snapshot.values())[0]
        w_v, b_v = val.get('wind', {}).get('v', 0), val.get('batt', {}).get('v', 0)
        pct = max(0, min(100, ((b_v - 3.2) / (4.2 - 3.2)) * 100))
        return f"Wind: {w_v}V, Batt: {b_v}V ({int(pct)}%)"
    except: return "Error"

tools_list = [execute_python_analysis, refresh_data]

# =====================================================
# 4. เริ่มต้นสมอง AI (เหมือนเดิมเป๊ะ)
# =====================================================
model = genai.GenerativeModel(
    model_name=valid_model_name,
    tools=tools_list,
    system_instruction="""คุณคือ Data Scientist AI วิเคราะห์พลังงานลม
    1. มีตัวแปร Global ชื่อ `df` (Pandas) เก็บประวัติข้อมูล (เวลาไทย UTC+7)
    2. คำนวณซับซ้อนให้เขียน Python ผ่าน `execute_python_analysis` เก็บผลที่ `result`
    ตอบเป็นภาษาไทยอย่างมั่นใจ"""
)
chat = model.start_chat(enable_automatic_function_calling=True)

# =====================================================
# 5. API Route (รองรับ GitHub/Render)
# =====================================================
@app.route('/ask', methods=['POST'])
def ask_ai():
    try:
        data = request.json
        user_input = data.get('question')
        
        # 🕒 จุดแก้ไขที่ 3: ส่งเวลาไทยปัจจุบันให้ AI รู้เรื่องวันนี้/เมื่อวาน
        now_thai = (datetime.utcnow() + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S")
        live_status = get_realtime_string()
        
        prompt = f"[Current Time: {now_thai}] [Realtime Status: {live_status}] Question: {user_input}"
        response = chat.send_message(prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

@app.route('/')
def home():
    return "Wind AI Server is Ready!"

if __name__ == '__main__':
    # 🕒 จุดแก้ไขที่ 4: รับพอร์ตอัตโนมัติจาก Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
