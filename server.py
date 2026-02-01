import os
import json
import pandas as pd
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

# =====================================================
# 1. เริ่มต้นระบบ & โหลดความลับ
# =====================================================
app = Flask(__name__)
CORS(app)

print("🔄 กำลังเริ่มระบบ AI Data Scientist Server (Cloud Mode - Gemini 2.5)...")

# ดึงค่าจาก Environment Variables (Render)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FIREBASE_CONFIG_JSON = os.environ.get("FIREBASE_SERVICE_ACCOUNT")

# ตรวจสอบว่ามี Key ครบไหม
if not GEMINI_API_KEY:
    print("❌ Error: ไม่พบ GEMINI_API_KEY ใน Environment")

if not FIREBASE_CONFIG_JSON:
    print("❌ Error: ไม่พบ FIREBASE_SERVICE_ACCOUNT ใน Environment")

# ตั้งค่า Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ตั้งค่า Firebase
try:
    if not firebase_admin._apps and FIREBASE_CONFIG_JSON:
        service_account_info = json.loads(FIREBASE_CONFIG_JSON)
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://win-assistant-462002-default-rtdb.asia-southeast1.firebasedatabase.app'
        })
        print("✅ Firebase Connected!")
    elif firebase_admin._apps:
        print("✅ Firebase Already Connected")
except Exception as e:
    print(f"❌ Firebase Error: {e}")

# =====================================================
# ⚡️ ระบบเลือก Model อัตโนมัติ (Prioritize 2.5)
# =====================================================
def get_generative_model():
    # รายชื่อโมเดลที่จะลองใช้ (เอา 2.5 ไว้บนสุดตามคำขอ)
    candidate_models = [
        "models/gemini-2.5-flash",      # 🚀 ตัวที่คุณต้องการ (จาก Log Local)
        "gemini-2.0-flash-exp",         # ตัวทดสอบ 2.0 (แรงมาก)
        "gemini-1.5-flash",             # ตัวมาตรฐาน (กันเหนียว)
        "gemini-1.5-flash-latest",      
        "gemini-1.5-pro",
        "models/gemini-1.5-flash"
    ]
    
    print("🔍 กำลังค้นหาโมเดลที่ใช้งานได้...")
    for model_name in candidate_models:
        try:
            print(f"   ...ทดสอบ: {model_name}")
            # ลองสร้างและยิง request เบาๆ เพื่อเทส
            model = genai.GenerativeModel(model_name)
            # ถ้าบรรทัดนี้ผ่าน แสดงว่า model valid
            print(f"✅ พบโมเดลที่ใช้งานได้: {model_name}")
            return model
        except Exception:
            continue
            
    print("⚠️ ไม่พบโมเดลในรายการ พยายามใช้ 'gemini-1.5-flash' เป็นค่า Default")
    return genai.GenerativeModel("gemini-1.5-flash")

# สร้างตัวแปร global สำหรับ model
model = None
if GEMINI_API_KEY:
    try:
        model = get_generative_model()
    except Exception as e:
        print(f"❌ Model Init Error: {e}")

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
                try:
                    dt = datetime.fromtimestamp(val['ts'] / 1000)
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
                except:
                    continue # ข้ามข้อมูลที่เสีย
        
        if records:
            df = pd.DataFrame(records)
            df['datetime'] = pd.to_datetime(df['datetime'])
            print(f"✅ ข้อมูลพร้อมวิเคราะห์: {len(df)} แถว")
            return f"อัปเดตสำเร็จ มีทั้งหมด {len(df)} รายการ"
        return "No valid records found"
    except Exception as e:
        print(f"Refresh Error: {e}")
        return f"Error: {e}"

# โหลดข้อมูลครั้งแรก
if firebase_admin._apps:
    refresh_data()

# =====================================================
# 3. AI Tools
# =====================================================

def execute_python_analysis(code_string):
    global df
    print(f"\n[AI Thinking] 🧠 รันโค้ดวิเคราะห์...")
    
    # ความปลอดภัยพื้นฐาน
    forbidden = ["import os", "import sys", "open(", "eval(", "exec(", "subprocess"]
    if any(f in code_string for f in forbidden):
        return "Security Alert: โค้ดมีความเสี่ยง ไม่สามารถรันได้"

    local_vars = {"df": df, "pd": pd, "result": None}
    try:
        # รันโค้ด
        exec(code_string, {}, local_vars)
        result = local_vars.get('result')
        if result is None:
            return "โค้ดทำงานสำเร็จ แต่ไม่ได้กำหนดค่าตัวแปร 'result'"
        return str(result)
    except Exception as e:
        return f"Code Error: {e}"

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
    except: return "Error fetching realtime data"

# รวม Tools
tools_list = [execute_python_analysis, refresh_data]

# สร้าง Chat Session (Global)
chat = None
if model:
    try:
        chat = model.start_chat(enable_automatic_function_calling=True)
    except Exception as e:
        print(f"❌ Chat Init Error: {e}")

# =====================================================
# 4. API Routes
# =====================================================

@app.route('/')
def home():
    status = "Online" if chat else "Offline (Model Error)"
    return f"Wind AI Server is {status}. Ready to serve!"

@app.route('/ask', methods=['POST'])
def ask_ai():
    global chat
    try:
        # เช็คว่าระบบพร้อมไหม
        if not chat:
            # พยายามต่อใหม่
            if model:
                chat = model.start_chat(enable_automatic_function_calling=True)
            else:
                return jsonify({"answer": "ขออภัย ระบบ AI ยังไม่พร้อมใช้งาน (API/Model Error)"})

        data = request.json
        user_input = data.get('question')
        if not user_input:
            return jsonify({"answer": "กรุณาพิมพ์คำถาม"})

        # เตรียม Prompt
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        live_status = get_realtime_string()
        
        system_prompt = """
        บทบาท: คุณคือ AI Data Scientist ผู้เชี่ยวชาญระบบพลังงานลม
        ข้อมูลที่มี:
        1. ตัวแปร `df` (Pandas DataFrame) เก็บข้อมูลประวัติ
        2. ข้อมูลสด (Realtime) ที่แนบไป
        
        คำแนะนำ:
        - ถ้าต้องคำนวณเชิงลึก/สถิติ ให้ใช้ `execute_python_analysis` เขียนโค้ดเสมอ
        - ตอบเป็นภาษาไทย กระชับ เข้าใจง่าย
        """
        
        full_prompt = f"{system_prompt}\n[Time: {current_time}] [Status: {live_status}] Question: {user_input}"
        
        # ส่งให้ AI
        response = chat.send_message(full_prompt)
        return jsonify({"answer": response.text})

    except Exception as e:
        traceback.print_exc()
        # ถ้า Error ให้ลอง Reset Chat
        try:
            if model:
                chat = model.start_chat(enable_automatic_function_calling=True)
        except: pass
        return jsonify({"answer": f"เกิดข้อผิดพลาด: {str(e)}"})

if __name__ == '__main__':
    # ใช้ PORT จาก Environment (จำเป็นสำหรับ Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
