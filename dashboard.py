"""
dashboard.py — Streamlit dashboard for the Thermal Comfort Detector.

Run with:
    streamlit run dashboard.py

Requires:
    pip install streamlit plotly ultralytics opencv-python numpy pandas
"""

from __future__ import annotations

import os
import tempfile
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Thermal Comfort Detector",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        border: 1px solid #2d3249;
    }
    .metric-card h1 { margin: 0; font-size: 2.8rem; }
    .metric-card p  { margin: 0; color: #888; font-size: 0.85rem; letter-spacing: 0.05em; text-transform: uppercase; }
    .room-badge {
        font-size: 1.6rem;
        font-weight: 700;
        padding: 0.4rem 1.4rem;
        border-radius: 999px;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .hot-badge   { background: #ff4b4b22; color: #ff4b4b; border: 1.5px solid #ff4b4b; }
    .cold-badge  { background: #4b9fff22; color: #4b9fff; border: 1.5px solid #4b9fff; }
    .neutral-badge { background: #ffffff18; color: #cccccc; border: 1.5px solid #555; }
    div[data-testid="stMetric"] { background: #1e2130; border-radius: 10px; padding: 0.8rem 1rem; border: 1px solid #2d3249; }
    .stButton > button { width: 100%; border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants (mirrored from thermal_detector_pose.py)
# ---------------------------------------------------------------------------
KP_NOSE, KP_LEYE, KP_REYE, KP_LEAR, KP_REAR = 0, 1, 2, 3, 4
KP_LSHO, KP_RSHO = 5, 6
KP_LELB, KP_RELB = 7, 8
KP_LWRI, KP_RWRI = 9, 10
KP_LHIP, KP_RHIP = 11, 12

BBox = Tuple[float, float, float, float]

CLOTH_WEIGHTS = {"scarf": 2, "sweater": 2, "cardigan": 1, "jacket": 1}

OBJECT_WEIGHTS = {
    "backpack":    ("cold", 1),
    "handbag":     ("cold", 1),
    "refrigerator":("cold", 1),
    "bottle":      ("hot",  1),
    "cup":         ("hot",  1),
    "sandwich":    ("hot",  1),
    "hot dog":     ("hot",  1),
    "pizza":       ("hot",  1),
    "laptop":      ("hot",  1),
    "microwave":   ("hot",  1),
    "oven":        ("hot",  1),
    "toaster":     ("hot",  1),
    "hair drier":  ("hot",  1),
}

STATE_COLORS = {"Hot": "#ff4b4b", "Cold": "#4b9fff", "Neutral": "#aaaaaa"}

# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------
@dataclass
class PersonState:
    history:         Deque[str]
    last_kp:         Optional[np.ndarray] = None
    wrist_motion:    Deque[float]         = field(default_factory=lambda: deque(maxlen=10))
    shoulder_motion: Deque[float]         = field(default_factory=lambda: deque(maxlen=10))
    debug:           Dict[str, float]     = field(default_factory=dict)
    run_counter:     Counter              = field(default_factory=Counter)

# ---------------------------------------------------------------------------
# Helper functions (mirrored from thermal_detector_pose.py)
# ---------------------------------------------------------------------------
def bbox_iou(a: BBox, b: BBox) -> float:
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih   = max(0.,ix2-ix1), max(0.,iy2-iy1)
    inter   = iw*ih
    if inter <= 0: return 0.
    area_a  = max(0.,ax2-ax1)*max(0.,ay2-ay1)
    area_b  = max(0.,bx2-bx1)*max(0.,by2-by1)
    denom   = area_a+area_b-inter
    return float(inter/denom) if denom > 0 else 0.

def bbox_center(b: BBox) -> Tuple[float,float]:
    x1,y1,x2,y2 = b; return (x1+x2)/2., (y1+y2)/2.

def point_in_bbox(pt: Tuple[float,float], b: BBox) -> bool:
    x,y=pt; x1,y1,x2,y2=b; return x1<=x<=x2 and y1<=y<=y2

def shoulder_width(kp: np.ndarray) -> float:
    return float(np.linalg.norm(kp[KP_LSHO]-kp[KP_RSHO]))

def norm_dist(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    return float(np.linalg.norm(a-b)/max(scale,1e-6))

def smooth(history: Deque[str]) -> str:
    return Counter(history).most_common(1)[0][0] if history else "Neutral"

def uniq_limit(labels: List[str], limit: int) -> List[str]:
    seen, out = set(), []
    for x in labels:
        if x in seen: continue
        seen.add(x); out.append(x)
        if len(out) >= limit: break
    return out

def state_color_bgr(state: str) -> Tuple[int,int,int]:
    if state == "Hot":  return (0,0,255)
    if state == "Cold": return (255,0,0)
    return (0,0,0)

def extract_pose_people(result) -> List[Tuple[int,BBox,np.ndarray]]:
    out, boxes, kps = [], getattr(result,"boxes",None), getattr(result,"keypoints",None)
    if boxes is None or kps is None: return out
    ids = getattr(boxes,"id",None)
    if ids is None: return out
    xyxy   = boxes.xyxy.detach().cpu().numpy()
    ids_np = ids.detach().cpu().numpy()
    kp_xy  = kps.xy.detach().cpu().numpy()
    for i in range(len(ids_np)):
        tid = int(ids_np[i])
        x1,y1,x2,y2 = map(float,xyxy[i])
        out.append((tid,(x1,y1,x2,y2),kp_xy[i]))
    return out

def extract_boxes(frame, model, conf, imgsz, *, exclude_labels=None):
    r = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    if r.boxes is None: return []
    names = r.names
    xyxy  = r.boxes.xyxy.detach().cpu().numpy()
    cls   = r.boxes.cls.detach().cpu().numpy()
    cf    = r.boxes.conf.detach().cpu().numpy()
    out   = []
    for i in range(len(xyxy)):
        if float(cf[i]) < conf: continue
        label = str(names[int(cls[i])])
        if exclude_labels and label in exclude_labels: continue
        x1,y1,x2,y2 = map(float,xyxy[i])
        out.append((label,(x1,y1,x2,y2)))
    return out

def assign_labels_to_people(people, objects, min_iou):
    assignments = {pid: [] for pid,_ in people}
    if not people or not objects: return assignments
    pid_to_bbox = {pid: bb for pid,bb in people}
    for label,ob in objects:
        oc = bbox_center(ob)
        best_pid, best_score = None, -1.
        for pid,pb in pid_to_bbox.items():
            iou   = bbox_iou(ob,pb)
            score = (1.+iou) if point_in_bbox(oc,pb) else iou
            if score > best_score: best_score,best_pid = score,pid
        if best_pid is None: continue
        if best_score >= 1. or bbox_iou(ob,pid_to_bbox[best_pid]) >= min_iou:
            assignments[best_pid].append(label)
    return assignments

def pose_scores(kp, st: PersonState, cfg: dict) -> Tuple[int,int,str]:
    sw = shoulder_width(kp)
    if sw < 5:
        for k in ["sh_mean","sh_jit","wr_mean","wr_jit","face_dist","hand_up"]:
            st.debug[k] = 0.
        return 0, 0, "Neutral"

    if st.last_kp is not None:
        prev = st.last_kp
        wrist_v    = float(np.linalg.norm(kp[KP_LWRI]-prev[KP_LWRI])+np.linalg.norm(kp[KP_RWRI]-prev[KP_RWRI]))
        shoulder_v = float(np.linalg.norm(kp[KP_LSHO]-prev[KP_LSHO])+np.linalg.norm(kp[KP_RSHO]-prev[KP_RSHO]))
        st.wrist_motion.append(wrist_v/sw)
        st.shoulder_motion.append(shoulder_v/sw)
    st.last_kp = kp

    cold, hot, labels = 0, 0, []
    lw,rw = kp[KP_LWRI], kp[KP_RWRI]
    lsho,rsho = kp[KP_LSHO], kp[KP_RSHO]
    lelb,relb = kp[KP_LELB], kp[KP_RELB]
    lhip,rhip = kp[KP_LHIP], kp[KP_RHIP]
    torso_top = min(lsho[1],rsho[1])
    torso_bot = max(lhip[1],rhip[1])
    torso_h   = max(1., torso_bot-torso_top)

    # Folded arms
    crossed = (lw[0]>rsho[0]) and (rw[0]<lsho[0])
    near_opp = (norm_dist(lw,relb,sw)<0.7) and (norm_dist(rw,lelb,sw)<0.7)
    if crossed and near_opp:
        cold += 2; labels.append("FoldedArms")

    # Rubbing hands
    wrists_close   = norm_dist(lw,rw,sw) < 0.40
    wrists_mid_y   = (lw[1]+rw[1])/2.
    wrists_in_torso= torso_top < wrists_mid_y < torso_bot
    if wrists_close and wrists_in_torso:
        cold += 2; labels.append("RubbingHands")

    # Motion stats
    if len(st.shoulder_motion) >= 8:
        sm = np.array(st.shoulder_motion,dtype=np.float32)
        sh_mean,sh_jit = float(np.mean(sm)), float(np.std(sm))
    else:
        sh_mean=sh_jit=0.

    if len(st.wrist_motion) >= 6:
        wm = np.array(st.wrist_motion,dtype=np.float32)
        wr_mean,wr_jit = float(np.mean(wm)), float(np.std(wm))
    else:
        wr_mean=wr_jit=0.

    # Shivering
    if sh_jit > cfg["shiver_jitter"] and sh_mean > cfg["shiver_mean"]:
        cold += 1; labels.append("Shivering")

    # Fanning
    face_pts = [kp[i] for i in [KP_NOSE,KP_LEYE,KP_REYE,KP_LEAR,KP_REAR]]
    min_face_dist = min(
        min(norm_dist(lw,fp,sw) for fp in face_pts),
        min(norm_dist(rw,fp,sw) for fp in face_pts),
    )
    near_face = min_face_dist < cfg["fan_face_dist"]
    shoulder_y= min(lsho[1],rsho[1])
    wrist_y   = min(lw[1],rw[1])
    hand_up   = wrist_y < (shoulder_y + cfg["fan_up_torso_frac"]*torso_h)
    if (near_face or hand_up) and (wr_mean > cfg["fan_mean"] or wr_jit > cfg["fan_jitter"]):
        hot += 2; labels.append("Fanning")

    st.debug.update(sh_mean=sh_mean, sh_jit=sh_jit, wr_mean=wr_mean,
                    wr_jit=wr_jit, face_dist=float(min_face_dist),
                    hand_up=1. if hand_up else 0.)

    return cold, hot, ("+".join(labels) if labels else "Neutral")

# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models(pose_model_name: str, clothes_model_path: str, objects_model_name: str):
    pose_model    = YOLO(pose_model_name)
    clothes_model = YOLO(clothes_model_path)
    objects_model = YOLO(objects_model_name)
    return pose_model, clothes_model, objects_model

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/thermometer.png", width=60)
st.sidebar.title("⚙️ Configuration")

st.sidebar.subheader("Models")
pose_model_name    = st.sidebar.selectbox("Pose model", ["yolov8n-pose.pt","yolov8s-pose.pt","yolov8m-pose.pt","yolov8l-pose.pt"], index=0)
clothes_model_path = st.sidebar.text_input("Clothing model path", "best.pt")
objects_model_name = st.sidebar.selectbox("Objects model", ["yolov8n.pt","yolov8s.pt"], index=0)
imgsz              = st.sidebar.selectbox("Image size (imgsz)", [640, 1280], index=0)

st.sidebar.subheader("Inference")
pose_conf    = st.sidebar.slider("Pose confidence",    0.1, 0.9, 0.35, 0.05)
clothes_conf = st.sidebar.slider("Clothing confidence",0.1, 0.9, 0.35, 0.05)
objects_conf = st.sidebar.slider("Objects confidence", 0.1, 0.9, 0.25, 0.05)
history_len  = st.sidebar.slider("Smoothing window (frames)", 5, 60, 25, 5)

st.sidebar.subheader("Pose Thresholds")
fan_mean        = st.sidebar.slider("Fan mean",         0.01, 0.15, 0.04,  0.01)
fan_jitter      = st.sidebar.slider("Fan jitter",       0.01, 0.15, 0.05,  0.01)
fan_face_dist   = st.sidebar.slider("Fan face dist",    0.3,  1.5,  0.70,  0.05)
fan_up_torso    = st.sidebar.slider("Fan up torso frac",0.1,  0.5,  0.25,  0.05)
shiver_jitter   = st.sidebar.slider("Shiver jitter",    0.005,0.1,  0.020, 0.005)
shiver_mean     = st.sidebar.slider("Shiver mean",      0.001,0.05, 0.010, 0.001)

st.sidebar.subheader("Room Conclusion")
threshold_pct   = st.sidebar.slider("Dissatisfaction threshold (%)", 10, 50, 20, 5)
min_frames_pct  = st.sidebar.slider("Min track length (%)", 0.3, 1.0, 0.90, 0.05)

cfg = dict(
    fan_mean=fan_mean, fan_jitter=fan_jitter, fan_face_dist=fan_face_dist,
    fan_up_torso_frac=fan_up_torso, shiver_jitter=shiver_jitter, shiver_mean=shiver_mean,
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🌡️ Thermal Comfort Detector")
st.caption("Vision-based occupant comfort analysis using YOLOv8 pose, clothing, and object detection.")

uploaded = st.file_uploader("Upload a video file", type=["mp4","avi","mov","mkv"])

run_col, _ = st.columns([1,3])
run_btn = run_col.button("▶ Run Analysis", type="primary", disabled=uploaded is None)

if not run_btn or uploaded is None:
    st.info("Upload a video and press **Run Analysis** to begin.")
    st.stop()

# Save upload to temp file
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tmp.write(uploaded.read())
tmp.close()
video_path = tmp.name

# Load models
with st.spinner("Loading models..."):
    pose_model, clothes_model, objects_model = load_models(pose_model_name, clothes_model_path, objects_model_name)

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
tracks:      Dict[int, PersonState] = {}
room_states: List[str]              = []
frame_log:   List[dict]             = []   # per-frame data for timeline
pose_label_counts: Counter          = Counter()
frame_idx = 0

progress_bar  = st.progress(0, text="Analysing video...")
status_text   = st.empty()

# Get total frame count for progress
cap_temp = cv2.VideoCapture(video_path)
total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
cap_temp.release()

for pose_result in pose_model.track(
    source=video_path,
    stream=True,
    persist=True,
    tracker="bytetrack.yaml",
    conf=pose_conf,
    imgsz=imgsz,
    verbose=False,
):
    frame = pose_result.orig_img
    if frame is None: continue

    people        = extract_pose_people(pose_result)
    people_bboxes = [(pid,bb) for pid,bb,_ in people]

    clothes = extract_boxes(frame, clothes_model, conf=clothes_conf, imgsz=imgsz)
    objects = extract_boxes(frame, objects_model, conf=objects_conf, imgsz=imgsz, exclude_labels={"person"})

    clothes_assign = assign_labels_to_people(people_bboxes, clothes, min_iou=0.05)
    objects_assign = assign_labels_to_people(people_bboxes, objects, min_iou=0.05)

    per_person_smooth: Dict[int,str] = {}

    for pid, bb, kp in people:
        st_p = tracks.get(pid)
        if st_p is None:
            st_p = PersonState(history=deque(maxlen=history_len))
            tracks[pid] = st_p

        clothes_labels = uniq_limit(clothes_assign.get(pid,[]), 4)
        obj_labels     = uniq_limit(objects_assign.get(pid,[]), 5)

        clothing_score = sum(CLOTH_WEIGHTS.get(lb,0) for lb in clothes_labels)
        pose_cold, pose_hot, pose_label = pose_scores(kp, st_p, cfg)

        obj_cold = sum(OBJECT_WEIGHTS[lb][1] for lb in obj_labels if OBJECT_WEIGHTS.get(lb,(None,))[0]=="cold")
        obj_hot  = sum(OBJECT_WEIGHTS[lb][1] for lb in obj_labels if OBJECT_WEIGHTS.get(lb,(None,))[0]=="hot")

        cold_score = clothing_score + pose_cold + obj_cold
        hot_score  = pose_hot + obj_hot

        if hot_score >= 2 and hot_score > cold_score: raw = "Hot"
        elif cold_score >= 2:                          raw = "Cold"
        else:                                          raw = "Neutral"

        st_p.history.append(raw)
        st_p.run_counter[raw] += 1
        per_person_smooth[pid] = smooth(st_p.history)

        pose_label_counts[pose_label] += 1

    room_counts = Counter(per_person_smooth.values())
    room_label  = room_counts.most_common(1)[0][0] if room_counts else "Neutral"
    room_states.append(room_label)

    frame_log.append(dict(
        frame=frame_idx,
        room=room_label,
        n_people=len(people),
        hot=room_counts.get("Hot",0),
        neutral=room_counts.get("Neutral",0),
        cold=room_counts.get("Cold",0),
    ))

    frame_idx += 1
    pct = min(frame_idx / total_frames, 1.0)
    progress_bar.progress(pct, text=f"Frame {frame_idx} / {total_frames} — Room: {room_label}")

progress_bar.empty()
status_text.empty()

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
all_counts = [sum(st_p.run_counter.values()) for st_p in tracks.values()]
# Only include tracks with at least 10 frames in the average to avoid ghost track dilution
meaningful = [c for c in all_counts if c >= 10]
avg_frames = sum(meaningful) / max(len(meaningful), 1)
MIN_FRAMES = int(min_frames_pct * avg_frames)

person_finals = {
    pid: st_p.run_counter.most_common(1)[0][0]
    for pid, st_p in tracks.items()
    if sum(st_p.run_counter.values()) >= MIN_FRAMES
}

state_counts = Counter(person_finals.values())
total_people = len(person_finals)

hot_frac  = state_counts.get("Hot",0)  / max(total_people,1)
cold_frac = state_counts.get("Cold",0) / max(total_people,1)
thr = threshold_pct / 100.0

if hot_frac >= thr and hot_frac >= cold_frac: overall = "Hot"
elif cold_frac >= thr:                         overall = "Cold"
else:                                          overall = "Neutral"

room_dist = Counter(room_states)

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Results")

# Room conclusion badge
badge_cls = {"Hot":"hot-badge","Cold":"cold-badge","Neutral":"neutral-badge"}[overall]
emoji     = {"Hot":"🔴","Cold":"🔵","Neutral":"⚪"}[overall]
st.markdown(f"""
<div style="text-align:center; padding: 1rem 0;">
  <p style="color:#888; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.4rem;">Overall Room Conclusion</p>
  <span class="room-badge {badge_cls}">{emoji} {overall}</span>
  <p style="color:#666; font-size:0.8rem; margin-top:0.4rem;">Threshold: {threshold_pct}% dissatisfied (ASHRAE 55 / ISO 7730)</p>
</div>
""", unsafe_allow_html=True)

# Metrics row
c1,c2,c3,c4 = st.columns(4)
c1.metric("🔴 Feeling Hot",     f"{state_counts.get('Hot',0)} people")
c2.metric("⚪ Feeling Neutral", f"{state_counts.get('Neutral',0)} people")
c3.metric("🔵 Feeling Cold",    f"{state_counts.get('Cold',0)} people")
c4.metric("👥 Total Tracked",   f"{total_people} people")

st.markdown("---")

# ---------------------------------------------------------------------------
# Charts row 1: donut + room timeline
# ---------------------------------------------------------------------------
col_a, col_b = st.columns([1, 2])

with col_a:
    st.subheader("Person State Distribution")
    labels_d = list(state_counts.keys())
    values_d = list(state_counts.values())
    colors_d = [STATE_COLORS[l] for l in labels_d]
    fig_donut = go.Figure(go.Pie(
        labels=labels_d, values=values_d,
        hole=0.55,
        marker=dict(colors=colors_d, line=dict(color="#0f1117", width=2)),
        textinfo="label+percent",
        textfont=dict(size=13),
    ))
    fig_donut.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font_color="white", showlegend=False,
        margin=dict(t=20,b=20,l=20,r=20), height=300,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_b:
    st.subheader("Room State Over Time")
    df_log = pd.DataFrame(frame_log)
    state_map = {"Hot":1,"Neutral":0,"Cold":-1}
    df_log["room_val"] = df_log["room"].map(state_map)

    fig_timeline = go.Figure()
    for state, color in STATE_COLORS.items():
        mask = df_log["room"] == state
        fig_timeline.add_trace(go.Scatter(
            x=df_log.loc[mask,"frame"], y=df_log.loc[mask,"room_val"],
            mode="markers", marker=dict(color=color, size=3),
            name=state,
        ))
    fig_timeline.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
        font_color="white", height=300,
        xaxis=dict(title="Frame", gridcolor="#2d3249"),
        yaxis=dict(title="State", tickvals=[-1,0,1], ticktext=["Cold","Neutral","Hot"], gridcolor="#2d3249"),
        legend=dict(bgcolor="#1e2130"),
        margin=dict(t=20,b=40,l=60,r=20),
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

# ---------------------------------------------------------------------------
# Charts row 2: people per frame + pose breakdown
# ---------------------------------------------------------------------------
col_c, col_d = st.columns([2,1])

with col_c:
    st.subheader("Hot / Neutral / Cold Count Over Time")
    fig_stack = go.Figure()
    FILL_COLORS = {
        "Hot":     "rgba(255,75,75,0.5)",
        "Cold":    "rgba(75,159,255,0.5)",
        "Neutral": "rgba(170,170,170,0.5)",
    }

    for state, color in STATE_COLORS.items():
        fig_stack.add_trace(go.Scatter(
            x=df_log["frame"], y=df_log[state.lower()],
            stackgroup="one", name=state,
            line=dict(width=0), fillcolor=FILL_COLORS[state],
            mode="lines",
        ))
    fig_stack.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
        font_color="white", height=280,
        xaxis=dict(title="Frame", gridcolor="#2d3249"),
        yaxis=dict(title="People", gridcolor="#2d3249"),
        legend=dict(bgcolor="#1e2130"),
        margin=dict(t=20,b=40,l=60,r=20),
    )
    st.plotly_chart(fig_stack, use_container_width=True)

with col_d:
    st.subheader("Detected Pose Behaviours")
    top_poses = pose_label_counts.most_common(8)
    if top_poses:
        pose_df = pd.DataFrame(top_poses, columns=["Pose","Count"])
        fig_pose = px.bar(
            pose_df, x="Count", y="Pose", orientation="h",
            color="Count", color_continuous_scale="Blues",
        )
        fig_pose.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
            font_color="white", height=280,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#2d3249"),
            yaxis=dict(gridcolor="#2d3249"),
            margin=dict(t=20,b=40,l=10,r=20),
        )
        st.plotly_chart(fig_pose, use_container_width=True)

# ---------------------------------------------------------------------------
# Room distribution bar
# ---------------------------------------------------------------------------
st.subheader("Room-Level State Distribution (frame count)")
rd_df = pd.DataFrame({"State":list(room_dist.keys()), "Frames":list(room_dist.values())})
fig_rd = px.bar(
    rd_df, x="State", y="Frames",
    color="State",
    color_discrete_map=STATE_COLORS,
)
fig_rd.update_layout(
    paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
    font_color="white", height=260, showlegend=False,
    xaxis=dict(gridcolor="#2d3249"),
    yaxis=dict(title="Frame count", gridcolor="#2d3249"),
    margin=dict(t=20,b=40,l=60,r=20),
)
st.plotly_chart(fig_rd, use_container_width=True)

# ---------------------------------------------------------------------------
# Raw data expander
# ---------------------------------------------------------------------------
with st.expander("📋 Per-person final state table"):
    person_rows = []
    for pid, st_p in tracks.items():
        total = sum(st_p.run_counter.values())
        if total < MIN_FRAMES:
            continue
        final = st_p.run_counter.most_common(1)[0][0]
        person_rows.append(dict(
            PersonID=pid,
            FinalState=final,
            Hot=st_p.run_counter.get("Hot",0),
            Neutral=st_p.run_counter.get("Neutral",0),
            Cold=st_p.run_counter.get("Cold",0),
            TotalFrames=total,
        ))
    if person_rows:
        person_df = pd.DataFrame(person_rows).sort_values("PersonID")
        st.dataframe(person_df, use_container_width=True, hide_index=True)

with st.expander("📋 Frame-by-frame log"):
    st.dataframe(df_log, use_container_width=True, hide_index=True)

# Cleanup
os.unlink(video_path)
