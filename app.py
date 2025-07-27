import time
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import google.generativeai as genai
import mediapipe as mp

# ------------------ Streamlit Page Config ------------------
st.set_page_config(page_title="Math Gesture Solver", layout="wide", page_icon="✋")

# ------------------ Header ------------------
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image("MathGestures.png", use_column_width=True)
with col_title:
    st.markdown("<h1 style='color:#4CAF50;'>✋ Math Gesture Solver</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:gray;'>Draw math problems with your hand and get instant solutions!</p>", unsafe_allow_html=True)

st.markdown("---")

# ------------------ Sidebar Controls ------------------
st.sidebar.header("⚙️ Controls")
run = st.sidebar.checkbox("Run Camera", value=True)
st.sidebar.info("✊ **Thumb Up** to clear\n\n🖐 **All Fingers Up** to solve problem")

# ------------------ UI Layout ------------------
col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("🎥 Live Camera Feed")
    FRAME_WINDOW = st.image([])

with col2:
    st.subheader("🧠 AI Answer")
    output_text_area = st.empty()

# ------------------ Cached Resources ------------------
@st.cache_resource(show_spinner=False)
def init_gemini():
    api_key = st.secrets.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

@st.cache_resource(show_spinner=False)
def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return mp_hands, hands, mp.solutions.drawing_utils

@st.cache_resource(show_spinner=False)
def init_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    return cap

model = init_gemini()
mp_hands, hands, mp_drawing = init_mediapipe()
cap = init_camera()

# ------------------ Session State ------------------
if "prev_pos" not in st.session_state:
    st.session_state.prev_pos = None
if "canvas" not in st.session_state:
    st.session_state.canvas = None
if "output_text" not in st.session_state:
    st.session_state.output_text = ""
if "last_call_ts" not in st.session_state:
    st.session_state.last_call_ts = 0.0

# ------------------ Helper Functions ------------------
def get_hand_info(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None
    hand_landmarks = results.multi_hand_landmarks[0]
    h, w, _ = img.shape
    lmList = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
    fingers = [1 if lmList[4][0] < lmList[3][0] else 0]  # Thumb
    fingers += [1 if lmList[i][1] < lmList[i-2][1] else 0 for i in [8, 12, 16, 20]]
    return fingers, lmList

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Draw
        current_pos = lmList[8]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, prev_pos, current_pos, (0, 180, 180), 10)
    elif fingers == [1, 0, 0, 0, 0]:  # Clear
        canvas.fill(0)
    return current_pos, canvas

def send_to_ai(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 1]:
        now = time.time()
        if now - st.session_state.last_call_ts < 3:
            return None
        try:
            pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            response = model.generate_content(["Solve this math problem from the image", pil_image])
            st.session_state.last_call_ts = now
            return response.text
        except Exception as e:
            return f"Error: {e}"
    return None

# ------------------ Main Loop ------------------
try:
    while run:
        success, img = cap.read()
        if not success:
            st.error("❌ Failed to capture video from webcam.")
            break
        img = cv2.flip(img, 1)
        if st.session_state.canvas is None:
            st.session_state.canvas = np.zeros_like(img)
        info = get_hand_info(img)
        if info:
            fingers, lmList = info
            current_pos, st.session_state.canvas = draw(info, st.session_state.prev_pos, st.session_state.canvas)
            st.session_state.prev_pos = current_pos if current_pos else None
            ai_response = send_to_ai(model, st.session_state.canvas, fingers)
            if ai_response:
                st.session_state.output_text = ai_response
        image_combined = cv2.addWeighted(img, 0.7, st.session_state.canvas, 0.3, 0)
        FRAME_WINDOW.image(image_combined, channels="BGR")
        output_text_area.markdown(
            f"""
            <div style='background:#ffffff; color:#000000; 
            padding:12px; border-radius:10px; 
            min-height:200px; overflow:auto; font-size:16px;'>
                {st.session_state.output_text}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.warning("🛑 Application stopped.")

finally:
    cap.release()
    cv2.destroyAllWindows()
