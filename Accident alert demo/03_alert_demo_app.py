# ============================================================================
# AIRI PITB Internship Task 1 - CCTV Accident Detection (YOLOv8)
# STEP SCRIPT 3 of 3: Optional Phase 11 Demo
# Accident Detection -> Incident Record -> Simulated Dispatch Interface
#
# This is a PROTOTYPE, exactly as scoped:
#   - Detects accidents in an uploaded/selected image using best.pt
#   - Generates an incident record: detection result, confidence, timestamp,
#     camera ID
#   - Looks up the camera's location from a mock camera-location database
#     (camera_id -> lat/lon), simulating what a real deployment would have
#   - Displays the generated alert on a simple "dispatch interface" screen
#   - Does NOT contact any real ambulance/hospital/emergency API. A real
#     deployment would forward this payload to an authorized emergency
#     services API - that integration is explicitly future work.
#
# HOW TO RUN (locally or in Colab with a tunnel):
#   pip install streamlit ultralytics
#   streamlit run 03_alert_demo_app.py -- --weights /path/to/best.pt
# ============================================================================

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="RescueBot CCTV Accident Alert - Prototype", layout="wide")

# ---------------------------------------------------------------------------
# Mock camera-location database
# In a real deployment this would be a proper database keyed by camera_id,
# populated when each CCTV unit is installed. Here it's hardcoded for the demo.
# ---------------------------------------------------------------------------
CAMERA_LOCATION_DB = {
    "CAM_001": {"name": "Mall Road & 2nd Ave Intersection", "lat": 31.5204, "lon": 74.3587},
    "CAM_002": {"name": "Highway Exit 12", "lat": 31.5497, "lon": 74.3436},
    "CAM_003": {"name": "University Road Junction", "lat": 31.4805, "lon": 74.3376},
    "CAM_004": {"name": "Canal Bank Crossing", "lat": 31.5100, "lon": 74.3200},
    "CAM_005": {"name": "Ring Road Overpass", "lat": 31.4700, "lon": 74.4100},
}

WEIGHTS_PATH = st.sidebar.text_input(
    "Path to best.pt", value="best.pt"
)
CONF_THRESHOLD = st.sidebar.slider("Detection confidence threshold", 0.1, 0.9, 0.35, 0.05)

st.title("CCTV Accident Detection - Alert & Dispatch Prototype")
st.caption(
    "Prototype only. Detected incidents are displayed here for demonstration; "
    "nothing is sent to a real ambulance, hospital, or emergency service."
)

@st.cache_resource
def load_model(weights_path):
    return YOLO(weights_path)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Camera Feed Input")
    camera_id = st.selectbox("Simulated camera ID", list(CAMERA_LOCATION_DB.keys()))
    uploaded_file = st.file_uploader("Upload a CCTV frame (jpg/png)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Frame from {camera_id}", use_container_width=True)

with col2:
    st.subheader("2. Detection Result")

    if uploaded_file:
        try:
            model = load_model(WEIGHTS_PATH)
        except Exception as e:
            st.error(f"Could not load model weights from '{WEIGHTS_PATH}'. "
                     f"Update the path in the sidebar. ({e})")
            st.stop()

        results = model.predict(image, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]

        names = result.names
        detections = []
        accident_detected = False
        best_conf = 0.0

        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = names[cls_id]
            detections.append({"label": label, "confidence": round(conf, 3)})
            if label == "accident":
                accident_detected = True
                best_conf = max(best_conf, conf)

        st.image(result.plot(), caption="Detections", use_container_width=True)
        st.json(detections)

        st.subheader("3. Incident Record")
        if accident_detected:
            location = CAMERA_LOCATION_DB[camera_id]
            incident = {
                "incident_id": f"INC-{random.randint(10000, 99999)}",
                "detection_result": "accident",
                "confidence_score": round(best_conf, 3),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "camera_id": camera_id,
                "location_name": location["name"],
                "latitude": location["lat"],
                "longitude": location["lon"],
            }

            st.error("ACCIDENT DETECTED - Incident generated")
            st.json(incident)

            st.subheader("4. Emergency Dispatch Interface (Simulated)")
            st.warning(
                f"ALERT: Accident detected at **{incident['location_name']}** "
                f"(Camera {incident['camera_id']}) — confidence "
                f"{incident['confidence_score']*100:.1f}%. "
                f"Time: {incident['timestamp']}."
            )
            st.info(
                "In this prototype, no real dispatch occurs. In a future version, "
                "this incident payload would be forwarded via an authorized API to "
                "the relevant ambulance/hospital/emergency-service system."
            )

            st.download_button(
                "Download incident record (JSON)",
                data=json.dumps(incident, indent=2),
                file_name=f"{incident['incident_id']}.json",
                mime="application/json",
            )
        else:
            st.success("No accident detected in this frame.")
    else:
        st.info("Upload a CCTV frame to run detection.")

st.divider()
st.caption(
    "Camera-location lookup uses a fixed mock database for this demo. "
    "In a real deployment, each installed CCTV unit's ID would be registered "
    "once against its physical location, so no GPS extraction from the image "
    "itself is required."
)
