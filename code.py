import streamlit as st
import streamlit_authenticator as stauth
import os
import datetime
import json
import re
import requests
import io
import base64
from groq import Groq
from huggingface_hub import InferenceClient, HfApi
from deep_translator import GoogleTranslator
from streamlit_lottie import st_lottie
from gtts import gTTS

# --- PAGE CONFIG ---
st.set_page_config(page_title="Indian-AI Dietician", page_icon="🥗", layout="wide")

# --- CUSTOM CSS FOR MODERN DRIBBBLE LOOK ---
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { justify-content: center; }
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# --- 1. LOTTIE ANIMATION LOADER ---
def load_lottieurl(url: str):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200: return None
        return r.json()
    except Exception as e:
        return None

# Guaranteed Working Public Lottie JSON URLs
lottie_login_hero = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json") 
lottie_register_hero = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json") 
lottie_health = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_l0u0rf4r.json") 

# --- 2. SETUP API CLIENTS ---
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
hf_client = InferenceClient(token=hf_token) if hf_token else None

if not groq_client and not hf_client:
    st.error("🚨 Error: No API Keys found. Please add GROQ_API_KEY or HF_TOKEN to Secrets.")
    st.stop()

# --- 3. FEEDBACK SYSTEM ---
def save_feedback(feedback_text, rating):
    repo_id = "missrufi/diet-app-feedback" 
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    feedback_data = {"timestamp": timestamp, "rating": rating, "comment": feedback_text}
    file_name = f"feedback_{timestamp}.json"
    try:
        if hf_token:
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=json.dumps(feedback_data, indent=4).encode(),
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="dataset"
            )
            return True
    except:
        return False

# --- 4. MASTER KNOWLEDGE BASE (ALL CONDITIONS & AGES) ---
CORE_CONTEXT = """
OFFICIAL DIETARY GUIDELINES (ICMR/NIN) - STRICT ADHERENCE REQUIRED:

**SECTION A: UNIVERSAL CULTURAL & ECONOMIC RULES**
1. INGREDIENTS: NEVER use Western items (Quinoa, Avocado, Turkey, Cod, Asparagus, Kale, Zucchini, Olive Oil unless specified). 
2. SUBSTITUTES: ALWAYS use Indian staples (Dalia, Bajra, Jowar, Ragi, Poha, Upma, Lauki, Tori, Palak, Methi, Rohu, Catla, Mustard/Groundnut Oil).
3. PRICING: NEVER estimate prices. Use easily accessible, budget-friendly items.
4. PORTIONS: Use standard Indian measures (1 Katori/Bowl, 1 medium Roti, 1 Glass). NEVER guess total meal calories.

**SECTION B: CONDITION-SPECIFIC MEDICAL GUIDELINES**
1. PCOS:
   * RULE: Low GI, Anti-inflammatory, Seed Cycling.
   * AVOID: Maida, Sugar, Fried Foods, Dairy (if acne-prone).
   * EAT: Ragi, Jowar, Moong, Seeds (Flax, Pumpkin). Meals: Methi Thepla, Sprouts.

2. DIABETES (Type 2):
   * RULE: Complex carbs, high fiber. NO simple sugars.
   * AVOID: Fruit juices, white rice eaten alone, Maida, potatoes/tubers.
   * EAT: Bajra, Oats, Karela, Jamun, Methi. Meals: Jowar Roti + Karela Sabzi.

3. CARDIAC (Heart Health):
   * RULE: Strict Low Salt (<1500mg sodium/day), High Fiber.
   * LIMIT: Oil to 3 tsp/day (Groundnut, Gingelly, or Mustard oil).
   * AVOID: Red Meat, Samosa/Papad/Pickles, Ghee, Palm Oil.
   * EAT: Millets, Local Fish, Garlic, Walnuts.

4. HYPERTENSION (High BP):
   * RULE: DASH Diet principles. High Potassium, Strict Low Sodium (<1500mg).
   * AVOID: Pickles, Papad, Processed Cheese, Salted Nuts.
   * EAT: Bananas, Spinach, Coconut Water, Curd.

5. WEIGHT LOSS:
   * RULE: Portion control, High Protein, High Fiber.
   * AVOID: Deep fried foods, sweets, sugary beverages.
   * EAT: Dal, Paneer, Soya, Chicken breast, Fish, Dalia, Buttermilk (Chaas).

6. WEIGHT GAIN:
   * RULE: Calorie surplus with nutrient-dense whole foods.
   * AVOID: Empty junk food calories.
   * EAT: Ghee, Nuts/Dry Fruits, Banana, Sweet Potato, Full-fat Milk, Paneer, Eggs. Meals: Banana milkshakes, Ghee Roti, Dal Fry.

7. PREGNANCY & LACTATION:
   * RULE: High Iron, Calcium, Folic Acid, and Protein.
   * AVOID: Raw/undercooked meat, raw eggs, papaya, excessive caffeine.
   * EAT: Drumstick leaves (Moringa), Spinach, Jaggery, Dates, Milk, Well-cooked meats, Eggs.

8. INFANTS & TODDLERS (6 months - 2 years):
   * RULE: Soft, easily digestible, weaning foods. NO added salt or sugar for infants under 1 year.
   * AVOID: Honey, whole nuts (choking hazard), cow's milk (under 1 year).
   * EAT: Mashed banana, Ragi sari (porridge), soft Dal-Khichdi, mashed potatoes.

9. SENIORS / ELDERLY:
   * RULE: Soft texture, easily digestible, high calcium, moderate protein.
   * AVOID: Excessively spicy, oily, or hard-to-chew foods.
   * EAT: Soft cooked rice, Dal soup, Steamed veggies, Papaya, Curd.

**SECTION C: NUTRITIONAL REFERENCE (IFCT 2017)**
   * Roti (Whole Wheat): ~85 kcal, 3g Protein.
   * Brown/White Rice (1 cup cooked): ~110 kcal.
   * Dal (1 cup cooked): ~110 kcal, 6g Protein.
   * Idli (1 medium): ~40 kcal.
   * Paneer (50g): ~130 kcal, 9g Protein.
   * Chicken/Fish (100g grilled): ~150 kcal, 25g Protein.
"""

# --- TEXT TO SPEECH CLEANER ---
def clean_for_speech(text):
    return text.replace("*", "").replace("#", "")

def generate_tts_audio(text, user_prompt):
    lang_code = 'en'
    if any('\u0900' <= c <= '\u097F' for c in user_prompt): lang_code = 'hi'
    elif any('\u0C80' <= c <= '\u0CFF' for c in user_prompt): lang_code = 'kn'
    try:
        clean_text = clean_for_speech(text)
        tts = gTTS(text=clean_text, lang=lang_code, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception:
        return None

# --- 5. OUTPUT CLEANER ---
def clean_output(text):
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    garbage = ["Enough thinking", "Here is the response", "Explanation:", "Note: This is a general", "I'm sorry", "As an AI", "I cannot"]
    for g in garbage: text = text.replace(g, "")
    replacements = {"Samosa": "Steamed Idli", "Pizza": "Vegetable Wrap", "Burger": "Sandwich", "Mutton": "Grilled Chicken", "Beef": "Fish", "Pork": "Lean Protein", "Maida": "Whole Wheat"}
    for bad, good in replacements.items(): text = text.replace(bad, good)
    return text.strip()

# --- 6. SMART FOLLOW-UP GENERATOR ---
def get_follow_up(query_lower, text):
    if "Next Step:" in text: return text
    greetings = ["hi", "hello", "hey", "namaste", "start", "greetings"]
    if any(x in query_lower for x in greetings) and len(query_lower.split()) < 3:
        return text + f"\n\n**Next Step:** How can I help you today? (e.g., Diet for PCOS, Baby Food Plan, Weight Loss)"
        
    if "pcos" in query_lower: q = "Would you like a list of fat-burning seeds for PCOS?"
    elif "diabet" in query_lower or "sugar" in query_lower: q = "Shall I suggest some low-sugar fruits?"
    elif "heart" in query_lower or "cardiac" in query_lower: q = "Would you like to know which cooking oils are safe for the heart?"
    elif "bp" in query_lower or "hypertension" in query_lower or "blood pressure" in query_lower: q = "Would you like a list of high-potassium foods to lower BP?"
    elif "baby" in query_lower or "infant" in query_lower or "toddler" in query_lower: q = "Would you like to know how to properly introduce solids?"
    elif "pregnan" in query_lower or "lactat" in query_lower: q = "Would you like a list of iron-rich foods?"
    elif "plan" in query_lower or "days" in query_lower: q = "Would you like the recipe for any of the dishes mentioned above?"
    else: q = "Would you like a full 7-day meal plan for this?"
    return text + f"\n\n**Next Step:** {q}"

# --- 7. GENERATE RESPONSE ---
def generate_response(system_msg, chat_history):
    if groq_client:
        try:
            messages = [{"role": "system", "content": system_msg}] + chat_history
            chat_completion = groq_client.chat.completions.create(
                messages=messages, 
                model="llama-3.3-70b-versatile", 
                temperature=0.35, # Bumped up slightly to allow for human warmth and empathy
                max_tokens=1500,
            ) 
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Groq Error: {e}")
    if hf_client:
        try:
            full_prompt = system_msg + "\n\n"
            for m in chat_history[-3:]: full_prompt += f"{m['role'].capitalize()}: {m['content']}\n"
            full_prompt += "Assistant:"
            response = hf_client.text_generation(full_prompt, model="mistralai/Mistral-7B-Instruct-v0.3", max_new_tokens=1000)
            return response
        except: return "Error: Unable to generate response."
    return "Error: No API connected."

# --- 8. DATABASE SETUP FOR ACCOUNTS ---
os.makedirs("user_data", exist_ok=True)
CRED_FILE = "user_data/credentials.json"

if not os.path.exists(CRED_FILE):
    default_credentials = {
        "credentials": {
            "usernames": {
                "rufi": {"email": "rufi@swasthya.com", "name": "Rufi", "password": "$2b$12$4mU2Yy0Gk0Xq1V8k3M6E8u1N6W5P7S9X8U9T8V8W9X0Y1Z2A3B4C5"},
                "khansa": {"email": "khansa@swasthya.com", "name": "Bi Bi Khansa", "password": "$2b$12$4mU2Yy0Gk0Xq1V8k3M6E8u1N6W5P7S9X8U9T8V8W9X0Y1Z2A3B4C5"}
            }
        }
    }
    with open(CRED_FILE, "w") as f: json.dump(default_credentials, f, indent=4)

with open(CRED_FILE, "r") as f: config = json.load(f)
authenticator = stauth.Authenticate(config['credentials'], "dietician_cookie", "auth_key", cookie_expiry_days=30)

# --- 9. UI & APP ROUTING ---
if not st.session_state.get("authentication_status"):
    st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🥗 Indian-AI Dietician</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Your Personal Indian Dietician</p>", unsafe_allow_html=True)
    st.write("") 
    
    tab1, tab2 = st.tabs(["🔒 Login", "📝 Create New Account"])
    with tab1:
        col_anim, col_form = st.columns([1.2, 1])
        with col_anim:
            if lottie_login_hero: st_lottie(lottie_login_hero, height=400, key="login_hero")
            else: st.markdown("<h1 style='text-align: center; font-size: 100px;'>👩‍⚕️</h1>", unsafe_allow_html=True)
        with col_form:
            st.write("\n\n")
            try: authenticator.login()
            except Exception as e: st.error(f"Login widget error: {e}")
            if st.session_state.get("authentication_status") is False: st.error("Username or password is incorrect")
                
    with tab2:
        col_anim2, col_form2 = st.columns([1.2, 1])
        with col_anim2:
            if lottie_register_hero: st_lottie(lottie_register_hero, height=400, key="reg_hero")
            else: st.markdown("<h1 style='text-align: center; font-size: 100px;'>📝</h1>", unsafe_allow_html=True)
        with col_form2:
            st.subheader("Join Indian-AI Dietician 🚀")
            new_name = st.text_input("Full Name")
            new_username = st.text_input("Choose a Username").lower()
            new_password = st.text_input("Choose a Password", type="password")
            
            if st.button("Create My Account", use_container_width=True, type="primary"):
                if new_username in config['credentials']['usernames']: st.error("Username already exists!")
                elif new_username and new_password and new_name:
                    try: hashed_pass = stauth.Hasher([new_password]).generate()[0]
                    except:
                        import bcrypt
                        hashed_pass = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                    config['credentials']['usernames'][new_username] = {"email": f"{new_username}@user.com", "name": new_name, "password": hashed_pass}
                    with open(CRED_FILE, "w") as f: json.dump(config, f, indent=4)
                    st.success("✅ Account created successfully! Please click the 'Login' tab to enter.")
                else: st.warning("Please fill out all fields.")

if st.session_state.get("authentication_status"):
    username = st.session_state["username"]
    name = st.session_state["name"]
    history_file = f"user_data/{username}_history.json"

    if "all_chats" not in st.session_state:
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        default_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                        st.session_state.all_chats = {default_id: {"name": "Previous Chat", "messages": data}}
                    else: st.session_state.all_chats = data
            except:
                st.session_state.all_chats = {datetime.datetime.now().strftime("%Y%m%d%H%M%S"): {"name": "New Chat", "messages": []}}
        else:
            st.session_state.all_chats = {datetime.datetime.now().strftime("%Y%m%d%H%M%S"): {"name": "New Chat", "messages": []}}
            
        st.session_state.current_chat_id = list(st.session_state.all_chats.keys())[-1]

    # --- SIDEBAR UI ---
    with st.sidebar:
        if lottie_health: st_lottie(lottie_health, height=180, key="health_anim")
        st.write(f"Welcome back, **{name}**! 👋")
        authenticator.logout(location="sidebar")
        st.divider()
        
        st.header("💬 Your Chats")
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            new_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            st.session_state.all_chats[new_id] = {"name": "New Chat", "messages": []}
            st.session_state.current_chat_id = new_id
            with open(history_file, "w") as f: json.dump(st.session_state.all_chats, f, indent=4)
            st.rerun()

        chat_options = list(st.session_state.all_chats.keys())
        chat_names = [st.session_state.all_chats[cid]["name"] for cid in chat_options]
        current_index = chat_options.index(st.session_state.current_chat_id)
        
        selected_name = st.radio("History:", chat_names, index=current_index, label_visibility="collapsed")
        selected_id = chat_options[chat_names.index(selected_name)]
        if selected_id != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected_id
            st.rerun()

        st.write("") 
        if st.button("🗑️ Delete Chat", use_container_width=True):
            if len(st.session_state.all_chats) > 1:
                del st.session_state.all_chats[st.session_state.current_chat_id]
                st.session_state.current_chat_id = list(st.session_state.all_chats.keys())[-1]
            else:
                st.session_state.all_chats[st.session_state.current_chat_id] = {"name": "New Chat", "messages": []}
            with open(history_file, "w") as f: json.dump(st.session_state.all_chats, f, indent=4)
            st.rerun()

    st.title("🥗 Indian AI Dietician")
    st.caption(f"Personalized Dashboard for {name} | Powered by ICMR/IFCT Data")

    # --- CURRENT CHAT LOGIC ---
    curr_id = st.session_state.current_chat_id
    current_messages = st.session_state.all_chats[curr_id]["messages"]

    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio_base64"):
                try:
                    audio_bytes = base64.b64decode(msg["audio_base64"])
                    st.audio(audio_bytes, format="audio/mp3")
                except: pass

    # --- INPUT SECTION ---
    st.markdown("---")
    
    col_mic, col_lang = st.columns([2, 1])
    with col_lang:
        voice_lang = st.selectbox("🗣️ Voice Language", ["Auto-Detect", "Kannada", "Hindi", "English"], index=0)
    
    lang_code_map = {"Kannada": "kn", "Hindi": "hi", "English": "en"}
    
    audio_val = None
    with col_mic:
        if hasattr(st, "audio_input"):
            dynamic_mic_key = f"mic_{curr_id}_{len(current_messages)}"
            audio_val = st.audio_input("🎤 Tap to speak your query:", key=dynamic_mic_key)
            
    text_val = st.chat_input("Ask about a diet (e.g., 'Diet for PCOS', 'Baby food plan')...")

    prompt = None
    if audio_val:
        with st.spinner("Transcribing your voice..."):
            try:
                audio_bytes = audio_val.getvalue()
                if len(audio_bytes) > 500: 
                    file_tuple = ("audio.wav", audio_bytes)
                    kwargs = {
                        "file": file_tuple, "model": "whisper-large-v3",
                        "prompt": "ನಮಸ್ಕಾರ, ಒಂದು ದಿನದ ಡಯಟ್ ಪ್ಲಾನ್ ಕೊಡಿ. Diet plan kodi." 
                    }
                    if voice_lang != "Auto-Detect": kwargs["language"] = lang_code_map[voice_lang]
                    transcription = groq_client.audio.transcriptions.create(**kwargs)
                    prompt = transcription.text
                else: st.warning("Audio was too short or mic is blocked.")
            except Exception as e: st.error(f"Groq API Error: {str(e)}")
    elif text_val: prompt = text_val

    if prompt:
        if st.session_state.all_chats[curr_id]["name"] == "New Chat":
            st.session_state.all_chats[curr_id]["name"] = prompt[:20] + "..."

        current_messages.append({"role": "user", "content": prompt})
        with open(history_file, "w") as f: json.dump(st.session_state.all_chats, f, indent=4)
            
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing Database..."):
                try: query_en = GoogleTranslator(source='auto', target='en').translate(prompt)
                except: query_en = prompt
                query_lower = query_en.lower()

                # --- UNIVERSAL SYSTEM PROMPT ENFORCEMENT ---
                if "week" in query_lower or "7 days" in query_lower or "10 days" in query_lower or ("plan" in query_lower and "yes" in query_lower):
                    sys_msg = f"""
                    You are a warm, empathetic, and highly professional Indian Dietician. 
                    CONTEXT: {CORE_CONTEXT} 
                    TASK: Create a Diet Plan for the requested demographic/condition. 
                    CRITICAL RULES: 
                    1. ALWAYS strictly follow the Medical Guidelines for the specific condition/age.
                    2. DO NOT invent or guess calorie counts. Use Indian portion sizes (e.g., '1 katori', '2 pieces'). 
                    3. You MUST start your response with a friendly, supportive human greeting.
                    FORMAT: 
                    [Warm, empathetic greeting acknowledging their goal/condition]
                    
                    **Day 1** * Breakfast: [Dish] (Portion) 
                    * Lunch: [Dish] (Portion) 
                    * Dinner: [Dish] (Portion) 
                    (Repeat for requested days)
                    
                    [Brief encouraging lifestyle tip]
                    """
                elif any(k in query_lower for k in ['chart', 'plan', 'routine', 'schedule']):
                    sys_msg = f"""
                    You are a warm, empathetic, and highly professional Indian Dietician. 
                    CONTEXT: {CORE_CONTEXT} 
                    TASK: Create a 1-Day Diet Chart based on the demographic/condition. 
                    CRITICAL RULES: 
                    1. ALWAYS strictly follow the Medical Guidelines for the specific condition/age.
                    2. DO NOT guess calorie counts. Use Indian portion sizes.
                    3. You MUST start your response with a friendly, supportive human greeting.
                    FORMAT: 
                    [Warm, empathetic greeting acknowledging their goal/condition]
                    
                    * Breakfast: [Dish] (Portion) 
                    * Lunch: [Dish] (Portion) 
                    * Dinner: [Dish] (Portion)
                    
                    [Brief encouraging lifestyle tip]
                    """
                elif any(k in query_lower for k in ['calorie', 'protein', 'nutri', 'fat', 'carb']):
                    sys_msg = f"You are a friendly Expert Nutritionist. CONTEXT: {CORE_CONTEXT} TASK: Provide specific nutritional values requested in a warm tone. RULE: Use ONLY the 'NUTRITIONAL REFERENCE' section."
                else:
                    sys_msg = f"You are a warm, empathetic Indian Dietician. CONTEXT: {CORE_CONTEXT} TASK: Answer helpfully based on the specific condition/demographic. RULE: NEVER hallucinate exact calorie numbers. Use Indian portion sizes. Speak warmly."

                chat_history_for_api = [{"role": m["role"], "content": m["content"]} for m in current_messages[-6:]]
                english_ans = generate_response(sys_msg, chat_history_for_api)
                english_ans = clean_output(english_ans)
                english_ans = get_follow_up(query_lower, english_ans)

                # --- LANGUAGE CONSISTENCY & TRANSLATION ---
                target_lang = 'en'
                if voice_lang == "Kannada": target_lang = 'kn'
                elif voice_lang == "Hindi": target_lang = 'hi'
                elif voice_lang == "English": target_lang = 'en'
                else:
                    if any('\u0900' <= c <= '\u097F' for c in prompt): target_lang = 'hi'
                    elif any('\u0C80' <= c <= '\u0CFF' for c in prompt): target_lang = 'kn'
                    elif any('\u0C00' <= c <= '\u0C7F' for c in prompt): target_lang = 'te'

                if target_lang != 'en': final_ans = GoogleTranslator(source='en', target=target_lang).translate(english_ans)
                else: final_ans = english_ans

                st.markdown(final_ans)

                audio_file = generate_tts_audio(final_ans, prompt)
                audio_b64 = None
                
                if audio_file:
                    audio_bytes_tts = audio_file.getvalue()
                    st.audio(audio_bytes_tts, format="audio/mp3")
                    audio_b64 = base64.b64encode(audio_bytes_tts).decode('utf-8')

                current_messages.append({"role": "assistant", "content": final_ans, "audio_base64": audio_b64})
                with open(history_file, "w") as f: json.dump(st.session_state.all_chats, f, indent=4)