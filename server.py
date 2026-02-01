import os
import json
import pandas as pd
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime, timedelta # <--- เพิ่ม timedelta เพื่อบวกเวลาไทย
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

# =====================================================
# 1. เริ่มต้นระบบ & โหลดความลับจาก Environment
# =====================================================
app = Flask(__name__)
CORS(app)

print("🔄 กำลังเริ่มระบบ AI Data Scientist Server (Cloud Mode + Timezone Fix)...")

# ดึง API Key จาก Environment Variable (Render)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FIREBASE_CONFIG_JSON = os.environ.get("FIREBASE_SERVICE_ACCOUNT")

if not GEMINI_API_KEY:
    print("❌ Error: ไม่พบ GEMINI_API_KEY ใน Environment")
    # ไม่ exit() เพื่อให้ Server ยังรันได้และแจ้งเตือนผ่านหน้าเว็บแทน

if not FIREBASE_CONFIG_JSON:
    print("❌ Error: ไม่พบ FIREBASE_SERVICE_ACCOUNT ใน Environment")
    # ไม่ exit() เพื่อให้ Server ยังรันได้

# ตั้งค่า Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ตั้งค่า Firebase จาก JSON String
try:
    if not firebase_admin._apps and FIREBASE_CONFIG_JSON:
        service_account_info = json.loads(FIREBASE_CONFIG_JSON)
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://win-assistant-462002-default-rtdb.asia-southeast1.firebasedatabase.app'
        })
        print("✅ Firebase Connected!")
except Exception as e:
    print(f"❌ Firebase Error: {e}")

# เลือก Model
valid_model_name = "models/gemini-1.5-flash" 

# =====================================================
# 2. การจัดการข้อมูล (Pandas)
# =====================================================
df = pd.DataFrame() 

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
                # --------------------------------------------------------
                # 🕒 จุดแก้ไขที่ 1: บังคับแปลงเป็นเวลาไทย (UTC+7)
                # --------------------------------------------------------
                # ใช้ utcfromtimestamp เพื่อให้เป็นกลาง แล้วบวก 7 ชม.
                dt = datetime.utcfromtimestamp(val['ts'] / 1000) + timedelta(hours=7)
                
                wind_p = float(val.get('wind', {}).get('p', 0))
                batt_p = float(val.get('batt', {}).get('p', 0))
                wind_v = float(val.get('wind', {}).get('v', 0))
                batt_v = float(val.get('batt', {}).get('v', 0))
                
                records.append({
                    "datetime": dt,
                    "date": dt.strftime("%Y-%m-%d"),
                    "hour": dt.hour,
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
        return f"อัปเดตสำเร็จ มีทั้งหมด {len(df)} รายการ"
    except Exception as e:
        return f"Error: {e}"

# เรียกโหลดข้อมูลครั้งแรก
if firebase_admin._apps:
    refresh_data()

# =====================================================
# 3. AI Tools & Functions
# =====================================================

def execute_python_analysis(code_string):
    global df
    print(f"\n[AI Thinking] 🧠 รันโค้ดวิเคราะห์...")
    if any(forbidden in code_string for forbidden in ["import os", "import sys", "open(", "eval("]):
        return "Security Alert: โค้ดมีความเสี่ยง"

    local_vars = {"df": df, "pd": pd, "result": None}
    try:
        exec(code_string, {}, local_vars)
        return str(local_vars.get('result', "ไม่ได้กำหนดค่า 'result'"))
    except Exception as e:
        return f"Code Error: {e}"

def get_realtime_string():
    try:
        ref = db.reference('History')
        snapshot = ref.order_by_key().limit_to_last(1).get()
        if not snapshot: return "No Data"
        val = list(snapshot.values())[0]
        w_v, b_v = val.get('wind', {}).get('v', 0), val.get('batt', {}).get('v', 0)
        pct = max(0, min(100, ((b_v - 3.2) / (4.2 - 3.2)) * 100))
        return f"Wind: {w_v}V, Batt: {b_v}V ({int(pct)}%)"
    except: return "Error"

# =====================================================
# 4. เริ่มต้นสมอง AI
# =====================================================
tools_list = [execute_python_analysis, refresh_data]
chat = None

if GEMINI_API_KEY:
    try:
        model = genai.GenerativeModel(
            model_name=valid_model_name,
            tools=tools_list,
            system_instruction="""คุณคือ Data Scientist AI วิเคราะห์พลังงานลม 
            - ใช้ execute_python_analysis เมื่อต้องคำนวณจาก df (Pandas)
            - ต้องเก็บผลลัพธ์ในตัวแปร result เสมอ
            - ตอบเป็นภาษาไทยอย่างเป็นกันเองและมั่นใจ"""
        )
        chat = model.start_chat(enable_automatic_function_calling=True)
    except Exception as e:
        print(f"❌ Model Init Error: {e}")

# =====================================================
# 5. API Routes
# =====================================================

@app.route('/')
def home():
    return "Wind AI Server is Running (Timezone Fixed)!"

@app.route('/ask', methods=['POST'])
def ask_ai():
    global chat
    try:
        if not chat:
             # ลองต่อใหม่ถ้าหลุด
             if GEMINI_API_KEY:
                 model = genai.GenerativeModel(model_name=valid_model_name, tools=tools_list)
                 chat = model.start_chat(enable_automatic_function_calling=True)
             else:
                 return jsonify({"answer": "Error: AI not initialized (Check API Key)"})

        user_input = request.json.get('question')
        live_status = get_realtime_string()
        
        # --------------------------------------------------------
        # 🕒 จุดแก้ไขที่ 2: ส่งเวลาไทยปัจจุบันให้ AI รู้เรื่อง
        # --------------------------------------------------------
        current_time_thai = (datetime.utcnow() + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S")
        
        prompt = f"[Time: {current_time_thai}] [Status: {live_status}] Question: {user_input}"
        
        response = chat.send_message(prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == '__main__':
    # สำหรับรันบน Cloud พอร์ตต้องดึงจาก Environment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
