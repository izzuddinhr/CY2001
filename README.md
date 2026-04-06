# Thermal Comfort Detector (Vision-Based)

A real-time, vision-based thermal comfort monitor that uses YOLOv8 pose estimation, clothing detection, and object context to classify indoor occupants as **Hot**, **Neutral**, or **Cold**. Room-level comfort is concluded using a **20% dissatisfaction threshold** aligned with ASHRAE 55 / ISO 7730 standards.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Files Required](#files-required)
- [Basic Usage](#basic-usage)
- [Key Parameters](#key-parameters)
- [Pose Detection Thresholds](#pose-detection-thresholds)
- [Model Selection Guide](#model-selection-guide)
- [GPU Usage](#gpu-usage-ssh-into-remote-server)
- [Room Conclusion Logic](#room-conclusion-logic)
- [Output](#output)

---

## Overview

The pipeline runs three models in parallel on every frame:

| Model | Purpose | Default |
|---|---|---|
| `yolov8n-pose.pt` | Detects people and their keypoints for pose analysis | Auto-downloads |
| `best.pt` | Custom clothing detector (scarf, sweater, cardigan, jacket) | Provided |
| `yolov8n.pt` | General COCO object detector (80+ classes, excludes person) | Auto-downloads |

Detections from all three are combined per person to produce a comfort score each frame.

---

## Requirements

```bash
pip install ultralytics opencv-python numpy
```

---

## Files Required

- `thermal_detector_pose.py` — Main script
- `best.pt` — Trained clothing model (4 classes: scarf, sweater, cardigan, jacket)
- `context.csv` — Object-to-comfort mapping reference
- `yolov8n-pose.pt` — Pose model *(auto-downloads if missing)*
- `yolov8n.pt` — General object model *(auto-downloads if missing)*

---

## Basic Usage

```bash
# Run on a video file
python thermal_detector_pose.py --source recordings/tests/your_video.mp4 --show --duration 999 --auto-save

# Run on webcam (default 60 seconds)
python thermal_detector_pose.py --source 0 --show --duration 60 --auto-save

# Run without display (headless server)
python thermal_detector_pose.py --source recordings/tests/your_video.mp4 --duration 999 --auto-save
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--source` | `0` | Camera index (e.g. `0`) or path to video file |
| `--duration` | `20.0` | Seconds to run. Use `999` to process full video |
| `--show` | `False` | Display live window. Omit on headless servers |
| `--auto-save` | `False` | Auto-save annotated output video with timestamp |
| `--save-video-dir` | `recordings/tests/outputs` | Output folder for saved videos. Works on both Windows and Linux |
| `--debug-metrics` | `False` | Overlay raw numeric metrics for threshold tuning |
| `--device` | `cpu` | Device to run models on. Use `--device cuda` when running on a GPU server |
| `--pose-model` | `yolov8n-pose.pt` | Pose model to use. See [Model Selection Guide](#model-selection-guide) |
| `--imgsz` | `640` | Input image size. Use `1280` with larger models on GPU |
| `--min-frames-pct` | `0.90` | Fraction of average track length a person must be tracked for to be included in the final summary. Lower to include more people, raise to filter more ghost tracks |

---

## Pose Detection Thresholds

These have been tuned and validated against recorded test videos. Override via command-line flags if needed.

| Parameter | Default | Triggers |
|---|---|---|
| `--fan-mean` | `0.04` | Fanning: wrist motion mean threshold |
| `--fan-jitter` | `0.050` | Fanning: wrist motion std threshold |
| `--fan-face-dist` | `0.70` | Fanning: wrist-to-face normalised distance |
| `--shiver-jitter` | `0.020` | Shivering: shoulder motion std threshold |
| `--shiver-mean` | `0.010` | Shivering: shoulder motion mean threshold |

---

## Model Selection Guide

Larger models maintain tracking more consistently across frames, reducing phantom ID reassignments for the same person. Use the nano model for quick local testing and the large model for accurate multi-person analysis on GPU.

| Pose Model | `--imgsz` | Device | Best For |
|---|---|---|---|
| `yolov8n-pose.pt` | `640` (default) | CPU / GPU | Close-range, single person, fast testing |
| `yolov8s-pose.pt` | `640` or `1280` | GPU | Slightly better than nano; light GPU load |
| `yolov8m-pose.pt` | `640` or `1280` | GPU | Small groups, moderate crowd density |
| `yolov8l-pose.pt` | `1280` | GPU | Meeting rooms, crowds, highest accuracy |

> **Note:** Larger models and higher `--imgsz` require significantly more compute. Do not use `yolov8l-pose.pt` on CPU.

---

## GPU Usage (SSH into Remote Server)

### Upload files from local machine

```bash
scp thermal_detector_pose.py user@server:~/
scp best.pt user@server:~/
scp -r recordings user@server:~/
```

### Run with GPU and large model

```bash
python3 thermal_detector_pose.py \
  --source recordings/tests/your_video.mp4 \
  --duration 999 \
  --auto-save \
  --device cuda \
  --pose-model yolov8l-pose.pt \
  --imgsz 1280
```

### Adjusting ghost track filtering

If the final people count seems too high, increase `--min-frames-pct`:

```bash
python3 thermal_detector_pose.py --source your_video.mp4 --duration 999 --auto-save --device cuda --min-frames-pct 0.95
```

If valid people are being excluded, lower it:

```bash
python3 thermal_detector_pose.py --source your_video.mp4 --duration 999 --auto-save --device cuda --min-frames-pct 0.60
```

### Download output video (run from local machine)

```bash
# Windows — do NOT include a trailing backslash at the end of the destination path
scp user@server:~/recordings/tests/outputs/your_video_TIMESTAMP.mp4 "C:\your\local\path\outputs"
```

---

## Room Conclusion Logic

1. Each person is classified frame-by-frame as **Hot**, **Neutral**, or **Cold** based on pose, clothing, and object scores
2. Each person's **final state** is the most common state across the entire video (not just the last few frames)
3. Ghost tracks are filtered using `--min-frames-pct` (default `0.90`) of average track length to exclude brief misdetections
4. Room is labelled **Hot** or **Cold** if ≥ 20% of tracked people feel that way, aligned with the ASHRAE 55 / ISO 7730 dissatisfaction threshold

---

## Output

Terminal output per frame:
```
Room=Neutral  People=4  Counts={'Neutral': 3, 'Cold': 1}
```

Final summary:
```
=== Final Room Summary (mode over run) ===
Feeling hot: 0 people
Feeling neutral: 3 people
Feeling cold: 2 people
Overall room summary: Cold
(Threshold: 20% of people feeling hot or cold to label the room as uncomfortable)
```

Saved video: annotated with bounding boxes, person IDs, comfort states, clothing/object detections, and optional debug metrics overlay.
