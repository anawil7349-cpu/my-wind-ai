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

print("🔄 กำลังเริ่มระบบ AI Data Scientist Server (Timezone & Model Fixed)...")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FIREBASE_CONFIG_JSON = os.environ.get("FIREBASE_SERVICE_ACCOUNT")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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

# =====================================================
# ⚡️ FIX: ระบบเลือก Model แบบ Fallback (ป้องกัน 404)
# =====================================================
def get_smart_model():
    # รายชื่อโมเดลที่ต้องการลอง (รวม 2.5 ตามที่คุณเคยใช้ได้)
    candidates = [
        "gemini-1.5-flash", 
        "models/gemini-1.5-flash",
        "gemini-2.0-flash-exp",
        "models/gemini-2.5-flash" # ตัวที่คุณแจ้งว่าใช้ได้ในเครื่อง
    ]
    
    for m_name in candidates:
        try:
            print(f"🔍 กำลังลองใช้โมเดล: {m_name}")
            model = genai.GenerativeModel(model_name=m_name)
            # ทดสอบยิงคำถามสั้นๆ เพื่อเช็คว่า 404 ไหม
            model.generate_content("test") 
            print(f"✅ ใช้โมเดลสำเร็จ: {m_name}")
            return model
        except Exception:
            continue
    return genai.GenerativeModel("gemini-1.5-flash") # สุดท้ายถ้าไม่ได้จริงๆ ให้ใช้ตัวมาตรฐาน

model = get_smart_model()

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
                # 🕒 จุดแก้ไข: บังคับเวลาไทย UTC+7 ให้ตรงกันทั้ง VS Code และ Render
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
        print(f"✅ ข้อมูลพร้อมวิเคราะห์: {len(df)} แถว (เวลาไทย UTC+7)")
        return f"อัปเดตสำเร็จ มีทั้งหมด {len(df)} รายการ"
    except Exception as e:
        return f"Error: {e}"

if firebase_admin._apps:
    refresh_data()

# =====================================================
# 3. AI Tools & API Logic
# =====================================================
def execute_python_analysis(code_string):
    global df
    local_vars = {"df": df, "pd": pd, "result": None}
    try:
        exec(code_string, {}, local_vars)
        return str(local_vars.get('result', "No result"))
    except Exception as e:
        return f"Error: {e}"

def get_realtime_string():
    try:
        ref = db.reference('History')
        snapshot = ref.order_by_key().limit_to_last(1).get()
        val = list(snapshot.values())[0]
        w_v, b_v = val.get('wind', {}).get('v', 0), val.get('batt', {}).get('v', 0)
        return f"Wind: {w_v}V, Batt: {b_v}V"
    except: return "No Realtime Data"

tools_list = [execute_python_analysis, refresh_data]
chat = model.start_chat(enable_automatic_function_calling=True)

@app.route('/')
def home():
    return "Wind AI Server is Running (Fixed)!"

@app.route('/ask', methods=['POST'])
def ask_ai():
    try:
        user_input = request.json.get('question')
        live_status = get_realtime_string()
        # 🕒 ส่งเวลาปัจจุบันที่เป็นเวลาไทยให้ AI รู้เรื่อง "วันนี้/เมื่อวาน"
        now_thai = (datetime.utcnow() + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S")
        
        prompt = f"[Current Thai Time: {now_thai}] [Status: {live_status}] Question: {user_input}"
        
        response = chat.send_message(prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
