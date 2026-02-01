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
CORS(app)  # อนุญาตให้หน้าเว็บคุยกับ Server ได้

print("🔄 กำลังเริ่มระบบ AI Data Scientist Server (Web Mode)...")

# --- ส่วนแก้ไขกุญแจ (ฝังลงโค้ดโดยตรง) ---
# นี่คือกุญแจ Firebase ของคุณที่ฝังลงไปเลย เพื่อแก้ปัญหาไฟล์อ่านไม่ได้
key_dict = {
  "type": "service_account",
  "project_id": "win-assistant-462002",
  "private_key_id": "874e7636b5dc4ad0bd835fc972c2b2ba760533ba",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDF0w8/eq4za/l2\nozelSwXbSB8V6ThxnNwjqhXnaNVXiXqtWwjVkhiiuwAfdmn/pHNvhSBrBXBv807T\ngGpU3wW1kzHHMHIkuCghmRbQQ2LdcCyyddQANYxVoJp6kvWESdK389fpU0+E/Mpz\nSBrNOaaf1+KDPdt2zptKYLztAKmq6Plbi7N/8/TP4wEljKq2pQ1p93ayl3AXkoEm\nGk/U7cuOZuGLoGJ6DSm5PYM156jbIyxMP8Q3VgK/BNSBlCAlKEDy0dWhRRcuZQaN\nozd1J3YNqREdF2i6+nETUIUpLqtl/pU6Hc4rieSfue+m7C4q/KBNmGqOYlT3sXPJ\n/YviDkk5AgMBAAECggEAC6acVzdA+RR4hmdsCuvFb1DPb+BHYoa62l+5+8+v3O5Q\nqxwQX1jWUQfKAEc53s1zviWa+GKesdgSOOnvK4rivkQhGSzHL5+6Y6C/wUncbUFC\nrTjgl5drMs1CKZPHc84GrRwdUOMZSFbhD4XJ1byMIA28QtGBwA8uSy3opya0Dkfq\nQLDE/d2k+YTHdaL5V21+3aITUVKMt2amwyVxBT1ynyJBVp1eMc4dNTocz7R23071\ngg+AfqpiLwcmj92KB9VrTaNdiZMwsak6enuLfaucSIfcgFQqeRJe+k45nlzhguEC\nJz/4DfqKMnY2qCkeFqyqgj2xBwKIH6lK/7ZNdCF/7wKBgQDlBFHjOUYpui/iooMV\nDypajIypXzV8kZbwZPSflutXhUm6N8lyWv5YPmUek21HczlnogeddJ1g1DqPQd5l\niV6At0PggTxOSesDlZQt2CZLfIyfCTohrYPwynV7DWHz9LxOzvt2eziFMZotlyUC\nLqN9igYcultmLyVSEz5J9FoA1wKBgQDdIedjVoCn0n7mM+4waTc6DhR0exPRY2S9\nwBsNGqHQFGghjTh9q8FV78E/dlIN9xxCJn9UHtmRbVQchjiE5M94LlcGK3vki66h\nlJzI/6CzInxapI9fhSu4oU9luwu9A/MJ0ERN7GqLrtlRXqYJFFdGqU//ywbTC/ji\nbOI/RT30bwKBgHXNTg3ylRu7sQwECidYALJJH8WzusCT5y9KzuYUbIQ9hJosPgv7\nsF9V4Q/kR877/yhGmWIt7RI0uNadzDcwfRL6sgiWkZ23uhLC61DVoYUs0Oyxg1x1\nc6v1iI6+aIdjeUWUhJcCdSVWSXdwCtJfiSt3RwOZ/I/IaosYaO8DqRRLAoGBANiG\n1jejAB9UMgXfW5/zpqwmFUlpKqKhHIfgj5xpM4C4Oq5/xYzonUs0lJk7lmUuTnFH\nmO4Ztxh7YRz9IGKgWbZoSbY05f+H9tso1czK4eQGJJXtBKaXk5QZ/9CxMnFGaLh3\nQiq7ECjucMUIVLQXQs5iA3+IoYoN8wpja7ZgaqXpAoGAA3B89PGl+EcBN79xcUCJ\nvDSAbjMYxlCZvpRl5i54kcUZbaG/mwTR+61zcohbmtSWTbt+tzVzsnCrIvDVT7MC\n6RAW8hHo6qDOebh3LEGjCzPS3C0D0aV880Hv4PUpNnFeO6Z4wKGOFEa/LxWxcRyD\nsj+UOptYCr2jak4a8S0y1i8=\n-----END PRIVATE KEY-----\n",
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
    print("✅ Firebase Connected! (ใช้กุญแจแบบฝังโค้ด)")
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
# 4. ความจำระยะยาว
# =====================================================
MEMORY_FILE = "ai_memory.json"
ai_memory = {}
if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f: ai_memory = json.load(f)
    except: pass

def remember_info(topic, info):
    ai_memory[topic] = info
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(ai_memory, f, ensure_ascii=False, indent=4)
    except: pass
    return f"จำแล้ว: {topic} = {info}"

def get_realtime_status():
    return get_realtime_string()

tools_list = [execute_python_analysis, remember_info, refresh_data, get_realtime_status]

# =====================================================
# 5. เริ่มต้นสมอง AI
# =====================================================

print("🧠 เชื่อมต่อสมอง...")
model = genai.GenerativeModel(
    model_name=valid_model_name,
    tools=tools_list,
    system_instruction=f"""
    คุณคือ Data Scientist AI อัจฉริยะที่มีข้อมูลพลังงานลมทั้งหมดในมือ
    
    1. คุณมีตัวแปร Global ชื่อ `df` (Pandas DataFrame) เก็บข้อมูลประวัติทั้งหมดไว้
       - คอลัมน์: [datetime, date (str), hour (int), wind_wh, batt_wh, wind_v, batt_v]
    
    2. ถ้าผู้ใช้ถามคำถามที่ต้องคำนวณ -> **จงเขียนโค้ด Python**
       - ใช้เครื่องมือ `execute_python_analysis`
       - ต้องเก็บผลลัพธ์ใส่ตัวแปร `result` เสมอ
    
    3. ข้อมูลสถานะปัจจุบัน (Real-time) จะแนบไปใน Prompt คำว่า [Realtime Status]
    4. ตอบเป็นภาษาไทยอย่างมั่นใจ สั้นกระชับ และเป็นธรรมชาติ
    """
)

chat = model.start_chat(enable_automatic_function_calling=True)

# =====================================================
# 6. Server API Route
# =====================================================

@app.route('/ask', methods=['POST'])
def ask_ai():
    try:
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
