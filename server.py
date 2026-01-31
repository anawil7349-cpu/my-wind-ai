import os
import json
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import pandas as pd
import io
import sys
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

# =====================================================
# 1. ตั้งค่าและเตรียมระบบ (เพิ่ม Flask เข้ามา)
# =====================================================
app = Flask(__name__)
CORS(app) # อนุญาตให้หน้าเว็บคุยกับ Server ได้

print("🔄 กำลังเริ่มระบบ AI Data Scientist Server (Web Mode)...")

if not os.path.exists("serviceAccountKey.json"):
    print("❌ Error: ไม่พบไฟล์ serviceAccountKey.json")
    exit()

try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://win-assistant-462002-default-rtdb.asia-southeast1.firebasedatabase.app'
    })
    print("✅ Firebase Connected!")
except ValueError:
    pass

GEMINI_API_KEY = "AIzaSyD0xILMuDcMuQBpYUO2G5odNUp_xTDY4u0"
genai.configure(api_key=GEMINI_API_KEY)

# หา Model (Logic เดิมของคุณ)
valid_model_name = "models/gemini-1.5-flash"
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name:
            valid_model_name = m.name
            break
except: pass
print(f"✅ ใช้โมเดล: {valid_model_name}")

# =====================================================
# 2. โหลดข้อมูลเข้า RAM (DataFrame) - (เหมือนเดิมเป๊ะ)
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

# ฟังก์ชันช่วยดึงค่าสด (Internal Helper)
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
# 3. เครื่องมือ Python Code Executor (เหมือนเดิมเป๊ะ)
# =====================================================

def execute_python_analysis(code_string):
    global df
    print(f"\n[AI Thinking] 🧠 กำลังรันโค้ดวิเคราะห์ข้อมูล...")
    
    if "import os" in code_string or "import sys" in code_string or "open(" in code_string:
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
# 4. ความจำระยะยาว (เหมือนเดิมเป๊ะ)
# =====================================================
MEMORY_FILE = "ai_memory.json"
ai_memory = {}
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f: ai_memory = json.load(f)

def remember_info(topic, info):
    ai_memory[topic] = info
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(ai_memory, f, ensure_ascii=False, indent=4)
    return f"จำแล้ว: {topic} = {info}"

def get_realtime_status():
    try:
        ref = db.reference('History')
        snapshot = ref.order_by_key().limit_to_last(1).get()
        val = list(snapshot.values())[0]
        return f"สถานะล่าสุด: ลม {val['wind']['v']}V, แบต {val['batt']['v']}V"
    except: return "ไม่สามารถดึงข้อมูลสดได้"

tools_list = [execute_python_analysis, remember_info, refresh_data, get_realtime_status]

# =====================================================
# 5. เริ่มต้นสมอง AI (เหมือนเดิมเป๊ะ)
# =====================================================

print("🧠 เชื่อมต่อสมอง...")
model = genai.GenerativeModel(
    model_name=valid_model_name,
    tools=tools_list,
    system_instruction=f"""
    คุณคือ Data Scientist AI อัจฉริยะที่มีข้อมูลพลังงานลมทั้งหมดในมือ
    
    1. คุณมีตัวแปร Global ชื่อ `df` (Pandas DataFrame) เก็บข้อมูลประวัติทั้งหมดไว้
       - คอลัมน์ใน df: [datetime, date (str), hour (int), wind_wh (พลังงานที่ผลิต), batt_wh (พลังงานที่ใช้), wind_v, batt_v]
    
    2. เมื่อผู้ใช้ถามคำถามที่ซับซ้อน หรือต้องคำนวณ -> **จงเขียนโค้ด Python** เพื่อหาคำตอบ
       - ให้ใช้เครื่องมือ `execute_python_analysis`
       - เขียนโค้ด Pandas เพื่อคำนวณ และ **ต้องเก็บผลลัพธ์สุดท้ายไว้ในตัวแปรชื่อ `result` เสมอ**
       
       ตัวอย่างการคิด:
       - ถาม: "วันไหนผลิตไฟเยอะสุด"
       - โค้ดที่ AI ควรสร้าง: 
         daily_grp = df.groupby('date')['wind_wh'].sum()
         best_day = daily_grp.idxmax()
         max_val = daily_grp.max()
         result = f"วันที่ {{best_day}} ผลิตได้ {{max_val:.2f}} Wh"
    
    3. ข้อมูลสถานะปัจจุบัน (Real-time) จะถูกแนบไปใน Prompt ให้แล้ว (ดูที่ [Realtime Status])
    4. ถ้าถามความรู้ทั่วไปที่คุณจำไว้ -> ตอบจาก Memory: {json.dumps(ai_memory, ensure_ascii=False)}
    5. ถ้าข้อมูลดูเก่าไป -> เรียก `refresh_data` ได้
    
    ตอบเป็นภาษาไทยอย่างมั่นใจ
    """
)

chat = model.start_chat(enable_automatic_function_calling=True)

# =====================================================
# 6. เปลี่ยนลูป input() เป็น Server API Route
# =====================================================

@app.route('/ask', methods=['POST'])
def ask_ai():
    try:
        # รับคำถามจากหน้าเว็บ
        data = request.json
        user_input = data.get('question')
        
        # เตรียมข้อมูลบริบท (Context)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        live_status = get_realtime_string() # ดึงค่าแบตสดๆ
        
        # สร้าง Prompt แบบยัดข้อมูลใส่ปาก (แก้ปัญหา AI บอกไม่รู้เรื่องแบต)
        prompt = f"""
        [Time: {current_time}]
        [Realtime Status: {live_status}]
        User Question: {user_input}
        """
        
        print(f"User asking: {user_input}")
        
        # ส่งให้ AI คิด
        response = chat.send_message(prompt)
        
        return jsonify({"answer": response.text})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"answer": f"Error: {str(e)}"})

# รัน Server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)