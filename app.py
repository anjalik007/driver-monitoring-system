import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import base64
from collections import deque

# --- Audio Injection Helper ---
def get_audio_html(file_path):
    """Converts an audio file into an HTML audio tag that plays automatically in the browser."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            return f"""
                <audio autoplay="true" loop="true">
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                </audio>
                """
    except FileNotFoundError:
        st.error(f"Audio file not found: {file_path}")
        return ""

# --- Streamlit Page Config ---
st.set_page_config(page_title="Driver Monitoring System", layout="wide")
st.title("🚗 Real-Time Driver Monitoring System")
st.markdown("This dashboard tracks eye closure (PERCLOS), yawning (MAR), and head pitch to detect drowsiness.")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ System Settings")
PERCLOS_THRESHOLD = st.sidebar.slider("PERCLOS Threshold", 0.1, 1.0, 0.5, 0.05)
MAR_THRESHOLD = st.sidebar.slider("MAR (Yawn) Threshold", 0.4, 1.2, 0.75, 0.05)
HEAD_TILT_THRESHOLD = st.sidebar.slider("Head Pitch Threshold (°)", 15, 45, 25, 1)
EMA_ALPHA = st.sidebar.slider("Smoothing Factor (Lower = Smoother)", 0.1, 1.0, 0.4, 0.1)

st.sidebar.markdown("---")
st.sidebar.info("Adjust these sliders in real-time to tune the sensitivity of the drowsiness detection.")

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Camera Feed")
    frame_placeholder = st.empty()
    alert_placeholder = st.empty()
    audio_placeholder = st.empty() 

with col2:
    st.subheader("Live Metrics")
    metric_perclos = st.empty()
    metric_mar = st.empty()
    metric_pitch = st.empty()
    metric_closed = st.empty()
    metric_score = st.empty()
    
    # --- Fatigue Graph Section ---
    st.markdown("---")
    st.subheader("Fatigue Trend")
    chart_placeholder = st.empty()

# --- DMS Functions ---
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 81, 311, 291, 78, 308]
CHIN = 152
NOSE = 1
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

def EAR(eye, landmarks, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h1 = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h1)

def MAR(landmarks, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in MOUTH])
    vertical = np.linalg.norm(pts[1] - pts[5])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    return vertical / horizontal

# --- Main App Logic ---
run_app = st.checkbox("▶️ Start Monitoring System", value=False)

if run_app:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)
    cap = cv2.VideoCapture(0)

    # State Variables
    PERCLOS_WINDOW_SECONDS = 10.0
    perclos_buffer = deque()
    score_history = deque([0]*50, maxlen=50) # Buffer for the live graph
    
    eye_closed = False
    eye_closed_start = None
    eye_closed_duration = 0
    alert_counter = 0
    alarm_playing = False
    smoothed_ear = 0.0
    smoothed_mar = 0.0
    smoothed_pitch = 0.0

    CALIBRATION_FRAMES = 50
    frame_count = 0
    baseline_ear_sum = 0
    calibrated_ear_threshold = 0.25

    font = cv2.FONT_HERSHEY_SIMPLEX
    color_black = (0, 0, 0)
    color_red = (0, 0, 255)
    line_type = cv2.LINE_AA

    while run_app:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam. (Note: standard cv2 webcam access fails on cloud servers).")
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        current_time = time.time()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark  

            leftEAR = EAR(LEFT_EYE, landmarks, w, h)
            rightEAR = EAR(RIGHT_EYE, landmarks, w, h)
            raw_ear = (leftEAR + rightEAR) / 2.0
            raw_mar = MAR(landmarks, w, h)

            # Head Tilt
            image_points = np.array([
                (landmarks[NOSE].x*w, landmarks[NOSE].y*h),
                (landmarks[LEFT_EYE[0]].x*w, landmarks[LEFT_EYE[0]].y*h),
                (landmarks[RIGHT_EYE[0]].x*w, landmarks[RIGHT_EYE[0]].y*h),
                (landmarks[CHIN].x*w, landmarks[CHIN].y*h),
                (landmarks[LEFT_MOUTH].x*w, landmarks[LEFT_MOUTH].y*h),
                (landmarks[RIGHT_MOUTH].x*w, landmarks[RIGHT_MOUTH].y*h)
            ], dtype=np.float64)

            model_points = np.array([
                (0.0, 0.0, 0.0), (-30.0, 30.0, -30.0), (30.0, 30.0, -30.0),
                (0.0, -70.0, -50.0), (-50.0, -30.0, -25.0), (50.0, -30.0, -25.0)
            ], dtype=np.float64)

            camera_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
            success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, np.zeros((4,1)))
            rmat, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            raw_pitch = abs(angles[0])

            # Calibration
            if frame_count < CALIBRATION_FRAMES:
                baseline_ear_sum += raw_ear
                frame_count += 1
                cv2.putText(rgb, f"CALIBRATING... {int((frame_count/CALIBRATION_FRAMES)*100)}%", 
                            (30, 160), font, 1, (0, 255, 255), 2, line_type)
                frame_placeholder.image(rgb, channels="RGB")
                continue
            elif frame_count == CALIBRATION_FRAMES:
                calibrated_ear_threshold = (baseline_ear_sum / CALIBRATION_FRAMES) * 0.75
                frame_count += 1

            # Smoothing
            if smoothed_ear == 0.0:
                smoothed_ear, smoothed_mar, smoothed_pitch = raw_ear, raw_mar, raw_pitch
            else:
                smoothed_ear = (EMA_ALPHA * raw_ear) + ((1 - EMA_ALPHA) * smoothed_ear)
                smoothed_mar = (EMA_ALPHA * raw_mar) + ((1 - EMA_ALPHA) * smoothed_mar)
                smoothed_pitch = (EMA_ALPHA * raw_pitch) + ((1 - EMA_ALPHA) * smoothed_pitch)

            # PERCLOS
            is_closed = 1 if smoothed_ear < calibrated_ear_threshold else 0
            perclos_buffer.append((current_time, is_closed))

            while perclos_buffer and (current_time - perclos_buffer[0][0]) > PERCLOS_WINDOW_SECONDS:
                perclos_buffer.popleft()

            perclos = sum(state for _, state in perclos_buffer) / len(perclos_buffer) if len(perclos_buffer) > 0 else 0

            # Duration
            if is_closed:
                if not eye_closed:
                    eye_closed = True
                    eye_closed_start = current_time
                eye_closed_duration = current_time - eye_closed_start
            else:
                eye_closed = False
                eye_closed_start = None
                eye_closed_duration = 0

            # Scoring
            drowsy_score = 0
            if perclos > PERCLOS_THRESHOLD: drowsy_score += 1
            if smoothed_mar > MAR_THRESHOLD: drowsy_score += 1
            if smoothed_pitch > HEAD_TILT_THRESHOLD: drowsy_score += 1
            if eye_closed_duration > 1.5: drowsy_score += 2 

            # Update graph history
            score_history.append(drowsy_score)

            if drowsy_score >= 2:
                alert_counter += 1
            else:
                alert_counter = max(0, alert_counter - 1) 

            # Alerts
            if alert_counter > 10:
                alert_placeholder.error("🚨 **DROWSINESS ALERT!** Please pull over and rest.")
                cv2.putText(rgb, "!!! DROWSINESS ALERT !!!", (30, h-50), font, 1.5, color_black, 5, line_type)
                cv2.putText(rgb, "!!! DROWSINESS ALERT !!!", (30, h-50), font, 1.5, color_red, 2, line_type)

                if not alarm_playing:
                    audio_html = get_audio_html("alarm.wav")
                    audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
                    alarm_playing = True
            else:
                alert_placeholder.empty()
                if alarm_playing:
                    audio_placeholder.empty() 
                    alarm_playing = False

            # Draw
            for i in LEFT_EYE + RIGHT_EYE + MOUTH:
                x, y = int(landmarks[i].x*w), int(landmarks[i].y*h)
                cv2.circle(rgb, (x, y), 2, (0, 255, 0), -1)

            # Update Metrics
            metric_perclos.metric("PERCLOS (10s Window)", f"{perclos:.2f}", 
                                  delta="High!" if perclos > PERCLOS_THRESHOLD else "Normal", 
                                  delta_color="inverse")
            metric_mar.metric("MAR (Yawn Metric)", f"{smoothed_mar:.2f}")
            metric_pitch.metric("Head Pitch", f"{smoothed_pitch:.1f}°")
            metric_closed.metric("Eye Closed Duration", f"{eye_closed_duration:.1f}s")
            metric_score.metric("Total Drowsiness Score", f"{drowsy_score}/5")
            
            # Update Live Graph
            chart_placeholder.line_chart(list(score_history), color="#FF4B4B")

        frame_placeholder.image(rgb, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()
    if alarm_playing:
        audio_placeholder.empty()
else:
    st.info("Check the box above to start the webcam and begin monitoring.")
