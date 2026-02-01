import os
import json
import pandas as pd
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

# =====================================================
# 1. เริ่มต้นระบบ & โหลดความลับ
# =====================================================
app = Flask(__name__)
CORS(app)

print("🔄 กำลังเริ่มระบบ AI Data Scientist Server (Master Version)...")

# ดึง Key จาก Environment (Render)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FIREBASE_CONFIG_JSON = os.environ.get("FIREBASE_SERVICE_ACCOUNT")

# ตั้งค่า Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ตั้งค่า Firebase (รองรับทั้งไฟล์ในเครื่อง และ Environment บน Cloud)
try:
    if not firebase_admin._apps:
        if FIREBASE_CONFIG_JSON:
            # กรณีรันบน Cloud
            cred = credentials.Certificate(json.loads(FIREBASE_CONFIG_JSON))
            print("✅ โหลด Firebase จาก Environment สำเร็จ")
        elif os.path.exists("serviceAccountKey.json"):
            # กรณีรันในเครื่อง
            cred = credentials.Certificate("serviceAccountKey.json")
            print("✅ โหลด Firebase จากไฟล์ JSON สำเร็จ")
        else:
            cred = None
            
        if cred:
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://win-assistant-462002-default-rtdb.asia-southeast1.firebasedatabase.app'
            })
            print("✅ Firebase Connected!")
except Exception as e:
    print(f"❌ Firebase Error: {e}")

# =====================================================
# ⚡️ FIX: ระบบเลือก Model อัตโนมัติ (กัน Error 404)
# =====================================================
def get_smart_model():
    candidates = [
        "gemini-1.5-flash", 
        "models/gemini-1.5-flash",
        "gemini-2.0-flash-exp",
        "models/gemini-2.5-flash"
    ]
    
    print("🔍 กำลังค้นหาโมเดล...")
    for m_name in candidates:
        try:
            model = genai.GenerativeModel(model_name=m_name)
            model.generate_content("test") # ลองยิงเทส
            print(f"✅ ใช้โมเดลสำเร็จ: {m_name}")
            return model
        except: continue
            
    return genai.GenerativeModel("gemini-1.5-flash") # ตัวกันตาย

model = None
if GEMINI_API_KEY:
    try:
        model = get_smart_model()
    except Exception as e:
        print(f"❌ Model Init Error: {e}")

# =====================================================
# 2. จัดการข้อมูล + แก้เวลาไทย (UTC+7)
# =====================================================
df = pd.DataFrame() 

def refresh_data():
    global df
    print("📥 Syncing Data...")
    try:
        ref = db.reference('History')
        data = ref.get()
        if not data: return "Database Empty"

        records = []
        for key, val in data.items():
            if isinstance(val, dict) and 'ts' in val:
                # 🕒 FIX 1: แปลง Timestamp เป็นเวลาไทยทันที (UTC+7)
                dt = datetime.utcfromtimestamp(val['ts'] / 1000) + timedelta(hours=7)
                
                wind_p = float(val.get('wind', {}).get('p', 0))
                batt_p = float(val.get('batt', {}).get('p', 0))
                
                records.append({
                    "datetime": dt,           # เวลาไทย
                    "date": dt.strftime("%Y-%m-%d"), # วันที่ไทย (เอาไว้ Group)
                    "wind_wh": wind_p / 60,   # พลังงานผลิต (Wh)
                    "batt_wh": batt_p / 60    # พลังงานใช้ (Wh)
                })
        
        df = pd.DataFrame(records)
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"✅ Data Ready: {len(df)} rows (Thai Timezone)")
        return f"Updated {len(df)} records."
    except Exception as e:
        return f"Error: {e}"

# โหลดข้อมูลครั้งแรก
if firebase_admin._apps:
    refresh_data()

# =====================================================
# 3. AI Tools
# =====================================================
def execute_python_analysis(code_string):
    global df
    print(f"\n[AI Thinking] 🧠 Running Code...")
    if any(f in code_string for f in ["import os", "import sys", "open(", "eval("]):
        return "Security Alert"
    
    local_vars = {"df": df, "pd": pd, "result": None}
    try:
        exec(code_string, {}, local_vars)
        res = local_vars.get('result')
        return str(res) if res is not None else "Code ran, but no 'result' variable set."
    except Exception as e:
        return f"Code Error: {e}"

def get_realtime_string():
    try:
        ref = db.reference('History')
        snapshot = ref.order_by_key().limit_to_last(1).get()
        val = list(snapshot.values())[0]
        w_v = val.get('wind', {}).get('v', 0)
        b_v = val.get('batt', {}).get('v', 0)
        return f"Wind: {w_v}V, Batt: {b_v}V"
    except: return "No Data"

tools_list = [execute_python_analysis, refresh_data]
chat = None

# =====================================================
# 4. API Routes
# =====================================================
@app.route('/')
def home():
    return "Wind AI Server is Running 24/7 (Timezone Fixed)!"

@app.route('/ask', methods=['POST'])
def ask_ai():
    global chat
    try:
        # Re-connect ถ้า Chat หลุด
        if not chat and model:
             chat = model.start_chat(enable_automatic_function_calling=True)
        
        if not chat:
            return jsonify({"answer": "ระบบ AI ไม่พร้อมใช้งาน (เช็ค API Key)"})

        user_input = request.json.get('question')
        
        # 🕒 FIX 2: ส่งเวลาไทยปัจจุบันให้ AI (เพื่อให้ตรงกับเวลาในตาราง)
        now_thai = (datetime.utcnow() + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S")
        live_status = get_realtime_string()
        
        # 🔥 Prompt บังคับโครงสร้างข้อมูลและเวลา
        system_prompt = f"""
        Current Thai Time: {now_thai}
        Role: Python Data Scientist.
        
        DATASET (`df`):
        - `datetime`: Thai Time (UTC+7)
        - `wind_wh`: Power Production (Wh) -> ใช้สำหรับ "ไฟเข้า", "ผลิตไฟ"
        - `batt_wh`: Power Consumption (Wh) -> ใช้สำหรับ "ใช้ไฟ", "กินไฟ"
        
        RULES:
        1. For past data/stats, usage of `execute_python_analysis` is MANDATORY.
        2. Assign final answer to `result` variable.
        3. Do NOT output python code text. Execute it.
        4. Answer in Thai.
        
        Question: {user_input}
        """
        
        response = chat.send_message(system_prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == '__main__':
    # รับ Port จาก Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
