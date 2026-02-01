import os
import json
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import pandas as pd
import sys
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

# =====================================================
# 1. ตั้งค่าและเตรียมระบบ (เพิ่ม Flask เข้ามา)
# =====================================================
app = Flask(__name__)
CORS(app) # อนุญาตให้หน้าเว็บคุยกับ Server ได้

print("🔄 กำลังเริ่มระบบ AI Data Scientist Server (Secure Cloud Mode)...")

# -----------------------------------------------------
# 🔐 จุดแก้ไขความปลอดภัย (Security Fix)
# -----------------------------------------------------
# ดึง Key จากระบบ Cloud แทนการเขียนลงไปตรงๆ
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FIREBASE_CONFIG = os.environ.get("FIREBASE_SERVICE_ACCOUNT")

# ตรวจสอบความถูกต้อง
if not GEMINI_API_KEY:
    print("❌ Error: ไม่พบ GEMINI_API_KEY ใน Environment Variables")
    # ไม่ exit() เพื่อให้ Server ยังรันได้และแจ้งเตือนผ่านหน้าเว็บแทน

# ตั้งค่า Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ตั้งค่า Firebase (รองรับทั้งแบบไฟล์และแบบ Env Var)
try:
    if not firebase_admin._apps:
        if FIREBASE_CONFIG:
            # กรณีรันบน Render (แนะนำ)
            cred = credentials.Certificate(json.loads(FIREBASE_CONFIG))
            print("✅ โหลด Firebase จาก Environment Variable สำเร็จ")
        elif os.path.exists("serviceAccountKey.json"):
            # กรณีรันในเครื่อง (Fallback)
            cred = credentials.Certificate("serviceAccountKey.json")
            print("✅ โหลด Firebase จากไฟล์ JSON สำเร็จ")
        else:
            print("⚠️ ไม่พบข้อมูลเชื่อมต่อ Firebase")
            cred = None

        if cred:
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://win-assistant-462002-default-rtdb.asia-southeast1.firebasedatabase.app'
            })
            print("✅ Firebase Connected!")
except Exception as e:
    print(f"❌ Firebase Connection Error: {e}")

# หา Model (ปรับปรุงให้รองรับหลายเวอร์ชัน)
valid_model_name = "models/gemini-1.5-flash"
try:
    # พยายามใช้ 2.0 ก่อน ถ้าไม่ได้จะถอยไป 1.5
    preferred_models = ["gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-1.5-flash"]
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    
    for want in preferred_models:
        for av in available_models:
            if want in av:
                valid_model_name = av
                break
        if valid_model_name != "models/gemini-1.5-flash":
            break
except: pass
print(f"✅ ใช้โมเดล: {valid_model_name}")

# =====================================================
# 2. โหลดข้อมูลเข้า RAM (DataFrame)
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
                        "minute": dt.minute,
                        "wind_p": wind_p,
                        "batt_p": batt_p,
                        "wind_wh": wind_p / 60,
                        "batt_wh": batt_p / 60,
                        "wind_v": wind_v,
                        "batt_v": batt_v
                    })
                except: continue
        
        df = pd.DataFrame(records)
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"✅ ข้อมูลพร้อมวิเคราะห์: {len(df)} แถว")
        return f"อัปเดตข้อมูลสำเร็จ มีทั้งหมด {len(df)} รายการ"
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return f"Error loading data: {e}"

# เรียกใช้ครั้งแรกถ้า Firebase พร้อม
if firebase_admin._apps:
    refresh_data()

# ฟังก์ชันช่วยดึงค่าสด
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
# 3. เครื่องมือ Python Code Executor
# =====================================================

def execute_python_analysis(code_string):
    global df
    print(f"\n[AI Thinking] 🧠 กำลังรันโค้ดวิเคราะห์ข้อมูล...")
    
    if any(x in code_string for x in ["import os", "import sys", "open(", "eval("]):
        return "Security Alert: ไม่สามารถรันโค้ดที่มีความเสี่ยงได้"

    local_vars = {"df": df, "pd": pd, "result": None}
    
    try:
        exec(code_string, {}, local_vars)
        output = local_vars.get('result')
        if output is None:
            return "โค้ดรันสำเร็จ แต่ไม่ได้กำหนดค่าใส่ตัวแปร 'result'"
        return str(output)

    except Exception as e:
        return f"เกิดข้อผิดพลาดในการรันโค้ด: {str(e)}"

# =====================================================
# 4. ความจำระยะยาว & AI Setup
# =====================================================
MEMORY_FILE = "ai_memory.json"
ai_memory = {}
# บน Cloud อาจจะไม่มีไฟล์นี้ หรือใช้ไม่ได้ (เพราะระบบรีเซ็ตตัวเอง) แต่ใส่กัน Error ไว้
if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f: ai_memory = json.load(f)
    except: pass

def remember_info(topic, info):
    ai_memory[topic] = info
    return f"จำแล้ว: {topic} = {info}"

def get_realtime_status():
    return get_realtime_string()

tools_list = [execute_python_analysis, remember_info, refresh_data, get_realtime_status]

print("🧠 เชื่อมต่อสมอง...")
chat = None
if GEMINI_API_KEY:
    try:
        model = genai.GenerativeModel(
            model_name=valid_model_name,
            tools=tools_list,
            system_instruction="""
            คุณคือ Data Scientist AI อัจฉริยะที่มีข้อมูลพลังงานลมทั้งหมดในมือ
            1. คุณมีตัวแปร Global ชื่อ `df` (Pandas DataFrame) เก็บข้อมูลประวัติ
            2. เมื่อผู้ใช้ถามคำถามที่ซับซ้อน หรือต้องคำนวณ -> **จงเขียนโค้ด Python** ใส่ตัวแปร `result`
            3. ข้อมูลสถานะปัจจุบัน (Real-time) จะถูกแนบไปใน Prompt
            ตอบเป็นภาษาไทยอย่างมั่นใจ
            """
        )
        chat = model.start_chat(enable_automatic_function_calling=True)
    except Exception as e:
        print(f"❌ Model Init Error: {e}")

# =====================================================
# 6. Server API Route
# =====================================================

@app.route('/ask', methods=['POST'])
def ask_ai():
    global chat
    try:
        # Re-initialize chat if needed
        if not chat and GEMINI_API_KEY:
             # (ใส่ Logic เดียวกับข้างบนเพื่อ Start Chat ใหม่ถ้าหลุด)
             pass 

        if not chat:
            return jsonify({"answer": "ระบบ AI ไม่พร้อมใช้งาน (ตรวจสอบ API Key)"})

        data = request.json
        user_input = data.get('question')
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        live_status = get_realtime_string()
        
        prompt = f"""
        [Time: {current_time}]
        [Realtime Status: {live_status}]
        User Question: {user_input}
        """
        
        print(f"User asking: {user_input}")
        response = chat.send_message(prompt)
        return jsonify({"answer": response.text})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"answer": f"Error: {str(e)}"})

# รัน Server (รองรับ Port ของ Render)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
