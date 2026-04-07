# file: thermal_detector_pose.py
from __future__ import annotations

import argparse
import os
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

BBox = Tuple[float, float, float, float]

# COCO pose keypoint indices
KP_NOSE = 0
KP_LEYE = 1
KP_REYE = 2
KP_LEAR = 3
KP_REAR = 4
KP_LSHO = 5
KP_RSHO = 6
KP_LELB = 7
KP_RELB = 8
KP_LWRI = 9
KP_RWRI = 10
KP_LHIP = 11
KP_RHIP = 12


@dataclass
class PersonState:
    history: Deque[str]
    last_kp: Optional[np.ndarray] = None  # (17,2)
    wrist_motion: Deque[float] = None
    shoulder_motion: Deque[float] = None
    debug: Dict[str, float] = None
    run_counter: Counter = None

    def __post_init__(self) -> None:
        if self.wrist_motion is None:
            self.wrist_motion = deque(maxlen=10)
        if self.shoulder_motion is None:
            self.shoulder_motion = deque(maxlen=10)
        if self.debug is None:
            self.debug = {}
        if self.run_counter is None:  
            self.run_counter = Counter()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Pose + Clothing + Objects comfort monitor (scaffold).")
    p.add_argument("--source", default="0", help="Camera index (e.g. 0) or video path/URL.")
    p.add_argument("--duration", type=float, default=20.0, help="Seconds to run.")
    p.add_argument("--show", action="store_true", help="Show live window.")
    p.add_argument("--save-video", default="", help="Optional output video path (mp4).")
    p.add_argument("--fps", type=float, default=20.0, help="FPS for saved video.")
    p.add_argument("--imgsz", type=int, default=640)

    p.add_argument("--pose-model", default="yolov8n-pose.pt", help="Ultralytics pose model.")
    p.add_argument("--clothes-model", default="best.pt", help="Your trained clothing model weights (4 classes).")
    p.add_argument("--objects-model", default="yolov8n.pt", help="General YOLO detector for all objects (COCO).")
    p.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config.")
    p.add_argument("--min-frames-pct", type=float, default=0.90, help="Minimum fraction of average track length for a person to be included in the final summary.")
    
    p.add_argument("--pose-conf", type=float, default=0.35)
    p.add_argument("--clothes-conf", type=float, default=0.35)
    p.add_argument("--objects-conf", type=float, default=0.25)  # lower so you see more than just 'person'

    p.add_argument("--history", type=int, default=25, help="Per-person smoothing window (frames).")
    p.add_argument("--min-iou-assign", type=float, default=0.05)

    # Display controls
    p.add_argument("--max-objects", type=int, default=5, help="Max object labels to show per person.")
    p.add_argument("--max-clothes", type=int, default=4, help="Max clothing labels to show per person.")
    p.add_argument("--max-pose-label", type=int, default=32, help="Max chars of pose label to show.")

    # Pose thresholds (tunable)
    p.add_argument("--shiver-jitter", type=float, default=0.020, help="Shivering: shoulder motion std threshold.")
    p.add_argument("--shiver-mean", type=float, default=0.010, help="Shivering: shoulder motion mean threshold.")

    p.add_argument("--fan-face-dist", type=float, default=0.70, help="Fanning: wrist-to-face normalized dist.")
    p.add_argument("--fan-mean", type=float, default=0.040, help="Fanning: wrist motion mean threshold.")
    p.add_argument("--fan-jitter", type=float, default=0.050, help="Fanning: wrist motion std threshold.")
    p.add_argument("--fan-up-torso-frac", type=float, default=0.25, help="Fanning: wrist above shoulders by torso frac.")

    # Debugging
    p.add_argument("--debug-metrics", action="store_true", help="Overlay shiver/fan numeric metrics for tuning.")
    p.add_argument("--auto-save", action="store_true", help="Auto-save output video to --save-video-dir with timestamp.")
    p.add_argument("--save-video-dir", default=r"recordings\tests\outputs", help="Folder for auto-saved videos.")
    p.add_argument("--device", default="cpu", help="Device to run models on: 'cuda' for GPU, 'cpu' for CPU.")

    return p.parse_args()


def coerce_source(src: str):
    try:
        return int(src)
    except ValueError:
        return src


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def bbox_center(b: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_bbox(pt: Tuple[float, float], b: BBox) -> bool:
    x, y = pt
    x1, y1, x2, y2 = b
    return x1 <= x <= x2 and y1 <= y <= y2


def state_color_bgr(state: str) -> Tuple[int, int, int]:
    if state == "Hot":
        return (0, 0, 255)
    if state == "Cold":
        return (255, 0, 0)
    return (0, 0, 0)


CLOTH_WEIGHTS = {
    "scarf": 1,
    "sweater": 1,
    "cardigan": 1,
    "jacket": 1,
}

OBJECT_WEIGHTS = {
    # --- Cold indicators ---
    "backpack": ("cold", 1),     # Extra layer carried on body; weak insulation signal
    "handbag": ("cold", 1),      # Extra item held close to body; weak insulation signal
    "refrigerator": ("cold", 1), # Cold appliance in environment; indicator of cooler surroundings

    # --- Hot indicators ---
    "bottle": ("hot", 1),        # Drinking water or cold drink; often used when feeling warm
    "cup": ("hot", 1),           # Suggests active thermal regulation
    "sandwich": ("hot", 1),      # Heavier food; weak hot indicator during warm breaks
    "hot dog": ("hot", 1),       # Hot food; weak hot indicator
    "pizza": ("hot", 1),         # Hot food; weak hot indicator
    "laptop": ("hot", 1),        # Generates local heat when in use; raises ambient temperature slightly
    "microwave": ("hot", 1),     # Appliance for heating food; indicates warm local environment
    "oven": ("hot", 1),          # Strong heat source appliance; raises local temperature significantly
    "toaster": ("hot", 1),       # Small appliance generating brief bursts of heat
    "hair drier": ("hot", 1),    # Blows hot air directly; strong hot environment indicator
}


def extract_pose_people(result) -> List[Tuple[int, BBox, np.ndarray]]:
    out = []
    boxes = getattr(result, "boxes", None)
    kps = getattr(result, "keypoints", None)
    if boxes is None or kps is None:
        return out

    ids = getattr(boxes, "id", None)
    if ids is None:
        return out

    xyxy = boxes.xyxy.detach().cpu().numpy()
    ids_np = ids.detach().cpu().numpy()
    kp_xy = kps.xy.detach().cpu().numpy()  # (n,17,2)

    for i in range(len(ids_np)):
        tid = int(ids_np[i])
        x1, y1, x2, y2 = map(float, xyxy[i])
        out.append((tid, (x1, y1, x2, y2), kp_xy[i]))
    return out


def extract_boxes(
    frame: np.ndarray,
    model: YOLO,
    conf: float,
    imgsz: int,
    *,
    device: str = "cuda",
    exclude_labels: Optional[set[str]] = None,
) -> List[Tuple[str, BBox]]:
    r = model(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    if r.boxes is None:
        return []
    names = r.names
    xyxy = r.boxes.xyxy.detach().cpu().numpy()
    cls = r.boxes.cls.detach().cpu().numpy()
    cf = r.boxes.conf.detach().cpu().numpy()

    out: List[Tuple[str, BBox]] = []
    for i in range(len(xyxy)):
        if float(cf[i]) < conf:
            continue
        label = str(names[int(cls[i])])
        if exclude_labels and label in exclude_labels:
            continue
        x1, y1, x2, y2 = map(float, xyxy[i])
        out.append((label, (x1, y1, x2, y2)))
    return out


def assign_labels_to_people(
    people: List[Tuple[int, BBox]],
    objects: List[Tuple[str, BBox]],
    min_iou: float,
) -> Dict[int, List[str]]:
    assignments: Dict[int, List[str]] = {pid: [] for pid, _ in people}
    if not people or not objects:
        return assignments

    pid_to_bbox = {pid: bb for pid, bb in people}

    for label, ob in objects:
        oc = bbox_center(ob)
        best_pid: Optional[int] = None
        best_score = -1.0

        for pid, pb in pid_to_bbox.items():
            iou = bbox_iou(ob, pb)
            score = (1.0 + iou) if point_in_bbox(oc, pb) else iou
            if score > best_score:
                best_score = score
                best_pid = pid

        if best_pid is None:
            continue

        if best_score >= 1.0 or bbox_iou(ob, pid_to_bbox[best_pid]) >= min_iou:
            assignments[best_pid].append(label)

    return assignments


def shoulder_width(kp: np.ndarray) -> float:
    return float(np.linalg.norm(kp[KP_LSHO] - kp[KP_RSHO]))


def norm_dist(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    return float(np.linalg.norm(a - b) / max(scale, 1e-6))


def pose_scores(kp: np.ndarray, st: PersonState, args: argparse.Namespace) -> Tuple[int, int, str]:
    """
    Returns (cold_score, hot_score, pose_label).

    Uses:
    - FoldedArms (cold)
    - RubbingHands (cold)
    - Shivering proxy via shoulder-motion stats (cold)
    - Fanning proxy via wrist-motion + hand position (hot)

    Also writes numeric debug metrics into st.debug:
      sh_mean, sh_jit, wr_mean, wr_jit, face_dist, hand_up
    """
    sw = shoulder_width(kp)
    if sw < 5:
        st.debug["sh_mean"] = 0.0
        st.debug["sh_jit"] = 0.0
        st.debug["wr_mean"] = 0.0
        st.debug["wr_jit"] = 0.0
        st.debug["face_dist"] = 0.0
        st.debug["hand_up"] = 0.0
        return 0, 0, "Neutral"

    # --- Temporal motion (normalized by shoulder width) ---
    if st.last_kp is not None:
        prev = st.last_kp
        # Total wrist displacement since last frame (left + right wrist)
        wrist_v = float(np.linalg.norm(kp[KP_LWRI] - prev[KP_LWRI]) + np.linalg.norm(kp[KP_RWRI] - prev[KP_RWRI]))
        # Total shoulder displacement since last frame (left + right shoulder)
        shoulder_v = float(np.linalg.norm(kp[KP_LSHO] - prev[KP_LSHO]) + np.linalg.norm(kp[KP_RSHO] - prev[KP_RSHO]))
        # Normalize by shoulder width so values are scale-invariant (same threshold works near/far from camera)
        st.wrist_motion.append(wrist_v / sw)
        st.shoulder_motion.append(shoulder_v / sw)
    # Store current keypoints for comparison in the next frame
    st.last_kp = kp

    cold = 0
    hot = 0
    labels: List[str] = []

    # Unpack keypoint coordinates for readability
    lw, rw = kp[KP_LWRI], kp[KP_RWRI]
    lsho, rsho = kp[KP_LSHO], kp[KP_RSHO]
    lelb, relb = kp[KP_LELB], kp[KP_RELB]
    lhip, rhip = kp[KP_LHIP], kp[KP_RHIP]

    # Define torso region using shoulder (top) and hip (bottom) y-coordinates
    torso_top = min(lsho[1], rsho[1])
    torso_bot = max(lhip[1], rhip[1])
    # Clamp to 1.0 to avoid division by zero if hip/shoulder keypoints overlap
    torso_h = max(1.0, torso_bot - torso_top)

    # --- 1) Folded arms / self-hug ---
    crossed = (lw[0] > rsho[0]) and (rw[0] < lsho[0])
    near_opposite_elbows = (norm_dist(lw, relb, sw) < 0.7) and (norm_dist(rw, lelb, sw) < 0.7)
    if crossed and near_opposite_elbows:
        cold += 2
        labels.append("FoldedArms")

    # --- 2) Rubbing hands ---
    wrists_close = norm_dist(lw, rw, sw) < 0.40 # wrists within 40% of shoulder width apart
    wrists_mid_y = (lw[1] + rw[1]) / 2.0
    wrists_in_torso = torso_top < wrists_mid_y < torso_bot
    if wrists_close and wrists_in_torso:
        cold += 2
        labels.append("RubbingHands")

    # --- Compute motion stats (used by both shiver & fanning) ---

    # Only compute shoulder stats if we have at least 8 frames of history
    # (avoids noisy stats from too few samples at the start)
    if len(st.shoulder_motion) >= 8:
        # Convert the rolling shoulder motion buffer to a numpy array for math
        sm = np.array(st.shoulder_motion, dtype=np.float32)
        # Average shoulder displacement per frame over the buffer (sustained movement)
        sh_mean = float(np.mean(sm))
        # Standard deviation of shoulder displacement (erratic/trembling movement)
        sh_jit = float(np.std(sm))
    else:
        # Not enough frames yet — default to zero so no false triggers at startup
        sh_mean = 0.0
        sh_jit = 0.0

    # Only compute wrist stats if we have at least 6 frames of history
    # (wrist buffer is smaller so threshold is lower)
    if len(st.wrist_motion) >= 6:
        # Convert the rolling wrist motion buffer to a numpy array for math
        wm = np.array(st.wrist_motion, dtype=np.float32)
        # Average wrist displacement per frame over the buffer (sustained movement)
        wr_mean = float(np.mean(wm))
        # Standard deviation of wrist displacement (erratic/back-and-forth movement)
        wr_jit = float(np.std(wm))
    else:
        # Not enough frames yet — default to zero so no false triggers at startup
        wr_mean = 0.0
        wr_jit = 0.0

    # --- 3) Shivering (stricter) ---
    if sh_jit > args.shiver_jitter and sh_mean > args.shiver_mean:
        cold += 1
        labels.append("Shivering")

    # --- 4) Fanning (easier) ---
    face_idxs = [KP_NOSE, KP_LEYE, KP_REYE, KP_LEAR, KP_REAR]
    face_pts = [kp[i] for i in face_idxs]
    min_face_dist = min(
        min(norm_dist(lw, fp, sw) for fp in face_pts),
        min(norm_dist(rw, fp, sw) for fp in face_pts),
    )
    near_face = min_face_dist < args.fan_face_dist

    shoulder_y = min(lsho[1], rsho[1])
    wrist_y = min(lw[1], rw[1])
    hand_up = wrist_y < (shoulder_y + args.fan_up_torso_frac * torso_h)

    # Trigger if (near_face OR hand_up) AND (wrist motion mean OR jitter)
    if (near_face or hand_up) and (wr_mean > args.fan_mean or wr_jit > args.fan_jitter):
        hot += 2
        labels.append("Fanning")

    # --- Debug metrics for tuning ---
    st.debug["sh_mean"] = sh_mean
    st.debug["sh_jit"] = sh_jit
    st.debug["wr_mean"] = wr_mean
    st.debug["wr_jit"] = wr_jit
    st.debug["face_dist"] = float(min_face_dist)
    st.debug["hand_up"] = 1.0 if hand_up else 0.0

    pose_label = "+".join(labels) if labels else "Neutral"
    return cold, hot, pose_label


def smooth(history: Deque[str]) -> str:
    return Counter(history).most_common(1)[0][0] if history else "Neutral"


def uniq_limit(labels: List[str], limit: int) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in labels:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
        if len(out) >= limit:
            break
    return out

def auto_video_path(source, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # source can be int webcam index or string filepath
    if isinstance(source, int):
        base = f"webcam{source}"
    else:
        base = os.path.splitext(os.path.basename(str(source)))[0] or "video"

    return os.path.join(save_dir, f"{base}_{ts}.mp4")

def main() -> None:
    args = parse_args()
    source = coerce_source(args.source)
    if args.auto_save and not args.save_video:
        args.save_video = auto_video_path(source, args.save_video_dir)
        print("Auto-saving to:", args.save_video)

    pose_model = YOLO(args.pose_model).to(args.device)
    clothes_model = YOLO(args.clothes_model).to(args.device)
    objects_model = YOLO(args.objects_model).to(args.device)

    writer = None
    if args.save_video:
        os.makedirs(os.path.dirname(args.save_video) or ".", exist_ok=True)

    tracks: Dict[int, PersonState] = {}
    room_states: List[str] = []

    t0 = time.time()
    for pose_result in pose_model.track(
        source=source,
        stream=True,
        persist=True,
        tracker=args.tracker,
        conf=args.pose_conf,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    ):
        # Only enforce --duration for live sources (webcam).
        # For video files, run until EOF so you don't cut the clip early on slow machines.
        if isinstance(source, int) and args.duration > 0 and (time.time() - t0 >= args.duration):
            break

        frame = pose_result.orig_img
        if frame is None:
            continue

        people = extract_pose_people(pose_result)  # (id, bbox, kp)
        people_bboxes = [(pid, bb) for pid, bb, _ in people]

        clothes = extract_boxes(frame, clothes_model, conf=args.clothes_conf, imgsz=args.imgsz, device=args.device)
        objects = extract_boxes(
            frame,
            objects_model,
            conf=args.objects_conf,
            imgsz=args.imgsz,
            device=args.device,
            exclude_labels={"person"},  # exclude person from object list
        )

        clothes_assign = assign_labels_to_people(people_bboxes, clothes, min_iou=args.min_iou_assign)
        objects_assign = assign_labels_to_people(people_bboxes, objects, min_iou=args.min_iou_assign)

        per_person_raw: Dict[int, str] = {}
        per_person_smooth: Dict[int, str] = {}
        per_person_debug: Dict[int, str] = {}

        for pid, bb, kp in people:
            st = tracks.get(pid)
            if st is None:
                st = PersonState(history=deque(maxlen=args.history))
                tracks[pid] = st

            clothes_labels = uniq_limit(clothes_assign.get(pid, []), args.max_clothes)
            obj_labels = uniq_limit(objects_assign.get(pid, []), args.max_objects)

            clothing_score = sum(CLOTH_WEIGHTS.get(lb, 0) for lb in clothes_labels)
            pose_cold, pose_hot, pose_label = pose_scores(kp, st, args)

            # Object context scoring
            obj_cold = sum(OBJECT_WEIGHTS[lb][1] for lb in obj_labels if OBJECT_WEIGHTS.get(lb, (None,))[0] == "cold")
            obj_hot  = sum(OBJECT_WEIGHTS[lb][1] for lb in obj_labels if OBJECT_WEIGHTS.get(lb, (None,))[0] == "hot")

            cold_score = clothing_score + pose_cold + obj_cold
            hot_score = pose_hot + obj_hot

            if hot_score >= 2 and hot_score > cold_score:
                raw = "Hot"
            elif cold_score >= 2:
                raw = "Cold"
            else:
                raw = "Neutral"

            st.history.append(raw)
            st.run_counter[raw] += 1
            per_person_raw[pid] = raw
            per_person_smooth[pid] = smooth(st.history)

            pose_short = pose_label if len(pose_label) <= args.max_pose_label else pose_label[: args.max_pose_label] + "…"
            metrics_txt = ""
            if args.debug_metrics:
                sh_mean = tracks[pid].debug.get("sh_mean", 0.0)
                sh_jit = tracks[pid].debug.get("sh_jit", 0.0)
                wr_mean = tracks[pid].debug.get("wr_mean", 0.0)
                wr_jit = tracks[pid].debug.get("wr_jit", 0.0)
                face_dist = tracks[pid].debug.get("face_dist", 0.0)
                hand_up = int(tracks[pid].debug.get("hand_up", 0.0))
                metrics_txt = f" sh(m={sh_mean:.3f},j={sh_jit:.3f}) wr(m={wr_mean:.3f},j={wr_jit:.3f}) face={face_dist:.2f} up={hand_up}"

            per_person_debug[pid] = (
                f"clothes={','.join(clothes_labels) if clothes_labels else '-'} "
                f"objs={','.join(obj_labels) if obj_labels else '-'} "
                f"pose={pose_short}"
                f"{metrics_txt}"
            )

        room_counts = Counter(per_person_smooth.values())
        room_label = room_counts.most_common(1)[0][0] if room_counts else "Neutral"
        room_states.append(room_label)

        vis = frame.copy()
        for pid, bb, _ in people:
            raw = per_person_raw.get(pid, "Neutral")
            sm = per_person_smooth.get(pid, "Neutral")
            color = state_color_bgr(raw)

            x1, y1, x2, y2 = map(int, bb)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

            txt = f"ID:{pid} {sm} ({per_person_debug.get(pid,'')})"
            y_text = max(20, y1 - 8)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 4), (255, 255, 255), -1)
            cv2.putText(vis, txt, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.rectangle(vis, (0, 0), (vis.shape[1], 32), (255, 255, 255), -1)
        cv2.putText(
            vis,
            f"Room: {room_label} | {dict(room_counts)}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            state_color_bgr(room_label),
            2,
            cv2.LINE_AA,
        )

        if args.save_video:
            if writer is None:
                h, w = vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save_video, fourcc, float(args.fps), (w, h))
            writer.write(vis)

        if args.show:
            cv2.imshow("Comfort (Pose+Clothes+Objects)", vis)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

        print(f"Room={room_label} People={len(people)} Counts={dict(room_counts)}")

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print("\n=== Final Room Summary (mode over run) ===")

    all_counts = [sum(st_p.run_counter.values()) for st_p in tracks.values()]

    # Only include tracks with at least 10 frames in the average to avoid ghost track dilution
    meaningful = [c for c in all_counts if c >= 10]

    avg_frames = sum(meaningful) / max(len(meaningful), 1)
    MIN_FRAMES = int(args.min_frames_pct * avg_frames)

    person_finals = {
        pid: st.run_counter.most_common(1)[0][0]
        for pid, st in tracks.items()
        if sum(st.run_counter.values()) >= MIN_FRAMES
    }
    state_counts = Counter(person_finals.values())
    total_people = len(person_finals)

    for state in ["Hot", "Neutral", "Cold"]:
        count = state_counts.get(state, 0)
        unit = "person" if count == 1 else "people"
        print(f"Feeling {state.lower()}: {count} {unit}")

    # ASHRAE 55 / ISO 7730: >20% dissatisfied = uncomfortable environment
    THRESHOLD = 0.20
    hot_frac = state_counts.get("Hot", 0) / max(total_people, 1)
    cold_frac = state_counts.get("Cold", 0) / max(total_people, 1)

    if hot_frac >= THRESHOLD and hot_frac >= cold_frac:
        overall = "Hot"
    elif cold_frac >= THRESHOLD:
        overall = "Cold"
    else:
        overall = "Neutral"

    print(f"Overall room summary: {overall}")
    print(f"(Threshold: {int(THRESHOLD*100)}% of people feeling hot or cold to label the room as uncomfortable)")


if __name__ == "__main__":
    main()