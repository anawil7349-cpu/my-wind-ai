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
# 1. เริ่มต้นระบบ
# =====================================================
app = Flask(__name__)
CORS(app)

print("🔄 กำลังเริ่มระบบ AI Data Scientist Server (Stricter Mode)...")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FIREBASE_CONFIG_JSON = os.environ.get("FIREBASE_SERVICE_ACCOUNT")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

try:
    if not firebase_admin._apps:
        if FIREBASE_CONFIG_JSON:
            cred = credentials.Certificate(json.loads(FIREBASE_CONFIG_JSON))
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

# เลือกโมเดล (ใช้ Fallback)
def get_smart_model():
    candidates = ["gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-2.0-flash-exp"]
    for m in candidates:
        try:
            model = genai.GenerativeModel(model_name=m)
            model.generate_content("test")
            print(f"✅ ใช้โมเดล: {m}")
            return model
        except: continue
    return genai.GenerativeModel("gemini-1.5-flash")

model = None
if GEMINI_API_KEY:
    try:
        model = get_smart_model()
    except Exception as e:
        print(f"❌ Model Error: {e}")

# =====================================================
# 2. จัดการข้อมูล + เวลาไทย (UTC+7)
# =====================================================
df = pd.DataFrame() 

def refresh_data():
    global df
    print("📥 Syncing data...")
    try:
        ref = db.reference('History')
        data = ref.get()
        if not data: return "Database Empty"

        records = []
        for key, val in data.items():
            if isinstance(val, dict) and 'ts' in val:
                # 🕒 เวลาไทย UTC+7
                dt = datetime.utcfromtimestamp(val['ts'] / 1000) + timedelta(hours=7)
                wind_p = float(val.get('wind', {}).get('p', 0))
                batt_p = float(val.get('batt', {}).get('p', 0))
                
                records.append({
                    "datetime": dt,
                    "date": dt.strftime("%Y-%m-%d"), # เอาไว้ Group by
                    "wind_wh": wind_p / 60, # สมมติเก็บเป็นรายนาที /60 ให้เป็น Wh
                    "batt_wh": batt_p / 60
                })
        
        df = pd.DataFrame(records)
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"✅ Data Loaded: {len(df)} rows (Thai Time)")
        return f"Updated {len(df)} records."
    except Exception as e:
        return f"Error: {e}"

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
        # สั่งให้ AI ต้องเก็บผลลัพธ์ลงตัวแปร result
        exec(code_string, {}, local_vars)
        res = local_vars.get('result')
        if res is None: return "Code ran but 'result' variable was None."
        return str(res)
    except Exception as e:
        return f"Code Error: {e}"

def get_realtime_string():
    try:
        ref = db.reference('History')
        snapshot = ref.order_by_key().limit_to_last(1).get()
        val = list(snapshot.values())[0]
        return f"Wind: {val.get('wind',{}).get('v',0)}V"
    except: return "No Data"

tools_list = [execute_python_analysis, refresh_data]
chat = None

# =====================================================
# 4. API Routes
# =====================================================
@app.route('/ask', methods=['POST'])
def ask_ai():
    global chat
    try:
        if not chat and model:
            # 🔥 จุดแก้สำคัญ: Instruction ต้องชัดเจนมาก
            chat = model.start_chat(
                enable_automatic_function_calling=True
            )

        user_input = request.json.get('question')
        now_thai = (datetime.utcnow() + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S")
        
        # 🔥 Prompt แบบบังคับ (CoT - Chain of Thought)
        system_prompt = f"""
        Current Thai Time: {now_thai}
        Role: You are a Python Data Scientist.
        
        RULES:
        1. When asked about past data (sum, average, count, "how much"), you MUST usage `execute_python_analysis`.
        2. DO NOT output python code blocks (```python ...```) in the final response. Run it instead!
        3. The dataframe `df` is already loaded.
        4. Always assign the final answer to the variable `result` in your python code.
        5. Respond in Thai Language only.
        
        Question: {user_input}
        """
        
        response = chat.send_message(system_prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
