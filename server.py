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
# 1. เริ่มต้นระบบ & โหลดความลับจาก Environment
# =====================================================
app = Flask(__name__)
CORS(app)

print("🔄 กำลังเริ่มระบบ AI Data Scientist Server (Full Fixed)...")

# 🔐 ดึง Key จาก Environment (Render)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FIREBASE_CONFIG_JSON = os.environ.get("FIREBASE_SERVICE_ACCOUNT")

# ตั้งค่า Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ตั้งค่า Firebase
try:
    if not firebase_admin._apps:
        if FIREBASE_CONFIG_JSON:
            service_account_info = json.loads(FIREBASE_CONFIG_JSON)
            cred = credentials.Certificate(service_account_info)
        elif os.path.exists("serviceAccountKey.json"):
            cred = credentials.Certificate("serviceAccountKey.json")
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
# ⚡️ FIX: ระบบเลือก Model แบบที่คุณต้องการ
# =====================================================
def get_smart_model():
    candidates = [
        "gemini-1.5-flash", 
        "models/gemini-1.5-flash",
        "gemini-2.0-flash-exp",
        "models/gemini-2.5-flash"
    ]
    
    print("🔍 กำลังค้นหาโมเดลที่ใช้งานได้...")
    for m_name in candidates:
        try:
            print(f"   ...ทดสอบ: {m_name}")
            model = genai.GenerativeModel(model_name=m_name)
            # ลองยิงคำถามสั้นๆ เพื่อเช็คว่า 404 ไหม
            model.generate_content("test") 
            print(f"✅ ใช้โมเดลสำเร็จ: {m_name}")
            return model
        except Exception:
            continue
            
    print("⚠️ ไม่พบโมเดลในรายการ พยายามใช้ 'gemini-1.5-flash' เป็นค่าสุดท้าย")
    return genai.GenerativeModel("gemini-1.5-flash")

# สร้างโมเดลเตรียมไว้
model = None
if GEMINI_API_KEY:
    try:
        model = get_smart_model()
    except Exception as e:
        print(f"❌ Model Init Error: {e}")

# =====================================================
# 2. การจัดการข้อมูล (Pandas + Timezone Fix)
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
                # 🕒 FIX TIMEZONE: แปลง UTC -> ไทย (UTC+7)
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
        print(f"✅ ข้อมูลพร้อมวิเคราะห์: {len(df)} แถว (เวลาไทย UTC+7)")
        return f"อัปเดตสำเร็จ มีทั้งหมด {len(df)} รายการ"
    except Exception as e:
        return f"Error: {e}"

if firebase_admin._apps:
    refresh_data()

# =====================================================
# 3. AI Tools & Functions
# =====================================================
def execute_python_analysis(code_string):
    global df
    print(f"\n[AI Thinking] 🧠 รันโค้ดวิเคราะห์...")
    if any(f in code_string for f in ["import os", "import sys", "open(", "eval("]):
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
        val = list(snapshot.values())[0]
        w_v = val.get('wind', {}).get('v', 0)
        b_v = val.get('batt', {}).get('v', 0)
        return f"Wind: {w_v}V, Batt: {b_v}V"
    except: return "No Realtime Data"

tools_list = [execute_python_analysis, refresh_data]

# สร้าง Chat Session
chat = None
if model:
    try:
        # Instruction สำคัญ: ย้ำให้ใช้ Python ถ้าถามเรื่องประวัติ
        chat = model.start_chat(enable_automatic_function_calling=True)
    except: pass

# =====================================================
# 4. API Routes
# =====================================================
@app.route('/')
def home():
    return "Wind AI Server is Running (Final Fixed)!"

@app.route('/ask', methods=['POST'])
def ask_ai():
    global chat
    try:
        # Re-connect ถ้า Chat ยังไม่มี หรือหลุด
        if not chat and model:
             chat = model.start_chat(enable_automatic_function_calling=True)
        
        if not chat:
            return jsonify({"answer": "AI ยังไม่พร้อมใช้งาน (กรุณารอสักครู่ หรือเช็ค API Key)"})

        user_input = request.json.get('question')
        
        # 🕒 FIX TIMEZONE 2: ส่งเวลาไทยปัจจุบันให้ AI รู้เรื่อง
        now_thai = (datetime.utcnow() + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S")
        live_status = get_realtime_string()
        
        # Prompt ที่บังคับให้ AI เข้าใจบริบท
        system_prompt = """
        คุณคือ Data Scientist AI
        - ข้อมูลประวัติอยู่ในตัวแปร `df` (Pandas)
        - ถ้าผู้ใช้ถามยอดรวม, ประวัติ, สถิติ -> ต้องใช้ `execute_python_analysis` เขียนโค้ดเสมอ
        - เวลาปัจจุบัน (ใน Prompt) คือเวลาไทย ใช้สำหรับอ้างอิง "วันนี้/เมื่อวาน"
        """
        
        prompt = f"{system_prompt}\n[Current Thai Time: {now_thai}] [Realtime Status: {live_status}] Question: {user_input}"
        
        response = chat.send_message(prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
