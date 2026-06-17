"""Evaluate the trained JumpRope MLP on a video file, frame by frame.

Usage:
    cd JumpRope-Model
    python -m training.jumprope_classifier.eval_video \
        --video VID_20260615_092921.mp4 \
        --model yolo11n-pose.pt \
        --mlp training/jumprope_classifier/exports/jumprope_mlp.bin \
        --output output_video.mp4 \
        --threshold 0.5

The pipeline mirrors the Android runtime:
    1. YOLO11-pose inference per frame (PyTorch, same model weights).
    2. Single-person tracking (largest bbox, grace frames).
    3. JumpRopeCounter state machine (C++ logic ported to Python).
    4. 12-dim feature extraction per cycle.
    5. MLP forward pass → score → gate the count.
    6. Annotated output video with count, state, and MLP score.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# YOLO model
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

# ---------------------------------------------------------------------------
# MLP config (must match training/config.py)
# ---------------------------------------------------------------------------
FEATURE_DIM = 12
FEATURE_NAMES = [
    "duration_frames", "amplitude_ratio", "rise_time_ratio",
    "fall_time_ratio", "rise_fall_symmetry", "knee_flexion_ratio",
    "ankle_elevation_ratio", "avg_confidence", "left_right_symmetry",
    "peak_velocity_ratio", "amplitude_pixels", "body_height_pixels",
]
NORMALIZE_STATS = [
    (22.0, 10.0), (0.08, 0.04), (0.40, 0.12), (0.60, 0.12),
    (0.80, 0.40), (0.10, 0.06), (0.05, 0.04), (0.50, 0.20),
    (0.03, 0.02), (0.04, 0.02), (30.0, 20.0), (350.0, 150.0),
]
MLP_HIDDEN1 = 16
MLP_HIDDEN2 = 8
MLP_TOTAL_PARAMS = 353

# COCO keypoint indices
KP_NOSE = 0
KP_LEFT_EYE = 1; KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3; KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5; KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7; KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9; KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11; KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13; KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15; KP_RIGHT_ANKLE = 16

# ---------------------------------------------------------------------------
# Counter constants (mirrors jumprope_counter.cpp)
# ---------------------------------------------------------------------------
MISSING_GRACE_FRAMES = 15
COUNT_COOLDOWN_FRAMES = 10
MIN_AIRBORNE_FRAMES = 2
SMOOTHING_ALPHA_MIN = 0.28
SMOOTHING_ALPHA_MAX = 0.48
MIN_AMPLITUDE_RATIO = 0.045
MIN_AMPLITUDE_PIXELS = 10.0
RETURN_AMPLITUDE_RATIO = 0.55
POSE_KP_CONFIDENCE = 0.20
MIN_FRAME_CONFIDENCE = 0.18
TORSO_TO_BODY_SCALE = 2.8


# ---------------------------------------------------------------------------
# MLP forward pass (pure NumPy, same weights layout as C++)
# ---------------------------------------------------------------------------
class JumpRopeMLP:
    def __init__(self, bin_path: str):
        blob = np.fromfile(bin_path, dtype=np.float32)
        if len(blob) != MLP_TOTAL_PARAMS:
            raise ValueError(f"Expected {MLP_TOTAL_PARAMS} params, got {len(blob)}")
        o = 0
        self.w1 = blob[o:o + FEATURE_DIM * MLP_HIDDEN1].reshape(FEATURE_DIM, MLP_HIDDEN1); o += FEATURE_DIM * MLP_HIDDEN1
        self.b1 = blob[o:o + MLP_HIDDEN1]; o += MLP_HIDDEN1
        self.w2 = blob[o:o + MLP_HIDDEN1 * MLP_HIDDEN2].reshape(MLP_HIDDEN1, MLP_HIDDEN2); o += MLP_HIDDEN1 * MLP_HIDDEN2
        self.b2 = blob[o:o + MLP_HIDDEN2]; o += MLP_HIDDEN2
        self.w3 = blob[o:o + MLP_HIDDEN2]; o += MLP_HIDDEN2
        self.b3 = blob[o]

    def predict(self, features: np.ndarray) -> float:
        """features: (12,) raw values → probability in [0,1]."""
        means = np.array([s[0] for s in NORMALIZE_STATS], dtype=np.float32)
        stds = np.array([s[1] for s in NORMALIZE_STATS], dtype=np.float32)
        x = (features - means) / stds
        h1 = np.maximum(0, x @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        logit = float(h2 @ self.w3 + self.b3)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))


# ---------------------------------------------------------------------------
# Pose → JumpRopeFrame (mirrors make_jumprope_frame_from_pose)
# ---------------------------------------------------------------------------
def make_frame_from_pose(kpts: np.ndarray, bbox_height: float) -> Dict:
    """kpts: (17, 3) [x, y, prob]"""
    invalid = {"has_person": False, "center_y": 0.0, "body_height": 0.0, "confidence": 0.0}

    def avg_y(indices):
        vals = []
        confs = []
        for i in indices:
            if kpts[i, 2] >= POSE_KP_CONFIDENCE:
                vals.append(kpts[i, 1])
                confs.append(kpts[i, 2])
        if not vals:
            return None, 0.0
        return float(np.mean(vals)), float(np.mean(confs))

    sh_y, sh_conf = avg_y([KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER])
    hi_y, hi_conf = avg_y([KP_LEFT_HIP, KP_RIGHT_HIP])
    an_y, an_conf = avg_y([KP_LEFT_ANKLE, KP_RIGHT_ANKLE])

    center_y = 0.0
    confidence = 0.0
    if sh_y is not None and hi_y is not None:
        center_y = sh_y * 0.35 + hi_y * 0.65
        confidence = (sh_conf + hi_conf) * 0.5
    elif hi_y is not None:
        center_y = hi_y
        confidence = hi_conf
    elif sh_y is not None and an_y is not None:
        center_y = (sh_y + an_y) * 0.5
        confidence = (sh_conf + an_conf) * 0.5
    elif sh_y is not None and bbox_height > 1:
        center_y = sh_y
        confidence = sh_conf * 0.80
    else:
        return invalid

    # Estimate body height from pose
    pose_height = 0.0
    if sh_y is not None and an_y is not None and an_y > sh_y:
        pose_height = an_y - sh_y
    elif sh_y is not None and hi_y is not None:
        torso = abs(hi_y - sh_y)
        if torso > 10:
            pose_height = torso * TORSO_TO_BODY_SCALE

    body_height = bbox_height
    if pose_height > 1:
        if body_height <= 1:
            body_height = pose_height
        else:
            lo = pose_height * 0.70
            hi = pose_height * 1.65
            if body_height < lo or body_height > hi:
                body_height = pose_height
            else:
                body_height = body_height * 0.5 + pose_height * 0.5

    if body_height <= 1:
        return invalid

    return {"has_person": True, "center_y": center_y, "body_height": body_height, "confidence": confidence}


# ---------------------------------------------------------------------------
# JumpRopeCounter (Python port of jumprope_counter.cpp)
# ---------------------------------------------------------------------------
class JumpRopeCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = 0  # 0=inactive, 1=no_person, 2=ready, 3=counting
        self.count = 0
        self.missing_frames = 0
        self.cooldown_frames = 0
        self.has_smoothed_y = False
        self.is_airborne = False
        self.airborne_frames = 0
        self.smoothed_y = 0.0
        self.previous_smoothed_y = 0.0
        self.baseline_y = 0.0
        self.peak_y = 0.0
        self.body_height = 0.0

        # Feature tracking
        self.last_features = None
        self.cycle_just_completed = False
        self._reset_cycle_tracking()
        self._reset_cooldown_tracking()

    def _reset_cycle_tracking(self):
        self.ft_cycle_frames = 0
        self.ft_rise_frames = 0
        self.ft_fall_frames = 0
        self.ft_min_y = 1e9
        self.ft_max_y = -1e9
        self.ft_amplitude_px = 0.0
        self.ft_peak_velocity = 0.0
        self.ft_max_knee_flexion = 0.0
        self.ft_max_ankle_elev = 0.0
        self.ft_conf_sum = 0.0
        self.ft_conf_count = 0
        self.ft_lr_diff_sum = 0.0
        self.ft_lr_count = 0
        self.ft_body_height_avg = 0.0
        self.ft_body_height_count = 0
        self.ft_start_frame = None
        self.ft_start_kpts = None

    def _reset_cooldown_tracking(self):
        self.cd_cycle_active = False
        self.cd_cycle_frames = 0
        self.cd_min_y = 1e9
        self.cd_peak_vel = 0.0
        self.cd_rise_frames = 0
        self.cd_fall_frames = 0

    def _accumulate_stats(self, frame, kpts):
        self.ft_cycle_frames += 1
        if frame["center_y"] < self.ft_min_y:
            self.ft_min_y = frame["center_y"]
        if frame["center_y"] > self.ft_max_y:
            self.ft_max_y = frame["center_y"]

        vel = abs(self.smoothed_y - self.previous_smoothed_y)
        if vel > self.ft_peak_velocity:
            self.ft_peak_velocity = vel

        if self.is_airborne:
            self.ft_rise_frames += 1
        else:
            self.ft_fall_frames += 1

        if frame["body_height"] > 1:
            self.ft_body_height_avg += frame["body_height"]
            self.ft_body_height_count += 1
        if frame["confidence"] > 0:
            self.ft_conf_sum += frame["confidence"]
            self.ft_conf_count += 1

        if kpts is not None and kpts.shape[0] >= 17:
            for side in range(2):
                hip_idx = 11 + side
                knee_idx = 13 + side
                ankle_idx = 15 + side
                if kpts[hip_idx, 2] > 0.2 and kpts[knee_idx, 2] > 0.2:
                    dist = abs(kpts[knee_idx, 1] - kpts[hip_idx, 1])
                    if dist > self.ft_max_knee_flexion:
                        self.ft_max_knee_flexion = dist
                if self.ft_start_frame is not None and kpts[ankle_idx, 2] > 0.2:
                    elev = self.ft_start_frame["center_y"] - kpts[ankle_idx, 1]
                    if elev > self.ft_max_ankle_elev:
                        self.ft_max_ankle_elev = elev
            if kpts[5, 2] > 0.2 and kpts[6, 2] > 0.2:
                self.ft_lr_diff_sum += abs(kpts[5, 1] - kpts[6, 1])
                self.ft_lr_count += 1
            if kpts[11, 2] > 0.2 and kpts[12, 2] > 0.2:
                self.ft_lr_diff_sum += abs(kpts[11, 1] - kpts[12, 1])
                self.ft_lr_count += 1

    def _compute_features(self):
        avg_bh = (self.ft_body_height_avg / max(self.ft_body_height_count, 1)) if self.ft_body_height_count > 0 else self.body_height
        if avg_bh <= 1:
            avg_bh = 1.0
        duration = max(1, self.ft_cycle_frames)
        amp_px = max(0.0, self.ft_max_y - self.ft_min_y)

        f = np.zeros(FEATURE_DIM, dtype=np.float32)
        f[0] = float(duration)
        f[1] = amp_px / avg_bh
        f[2] = self.ft_rise_frames / duration
        f[3] = self.ft_fall_frames / duration
        f[4] = (self.ft_rise_frames / self.ft_fall_frames) if self.ft_fall_frames > 0 else 1.0
        f[5] = self.ft_max_knee_flexion / avg_bh
        f[6] = self.ft_max_ankle_elev / avg_bh
        f[7] = (self.ft_conf_sum / self.ft_conf_count) if self.ft_conf_count > 0 else 0.0
        f[8] = ((self.ft_lr_diff_sum / self.ft_lr_count) / avg_bh) if self.ft_lr_count > 0 else 0.0
        f[9] = self.ft_peak_velocity / avg_bh
        f[10] = amp_px
        f[11] = avg_bh
        return f

    def update(self, frame: Dict, kpts: Optional[np.ndarray] = None) -> Tuple[int, int, bool, Optional[np.ndarray]]:
        """Returns (state, count, cycle_completed, features_or_None)."""
        self.cycle_just_completed = False
        self.last_features = None

        if not frame["has_person"] or frame["body_height"] <= 1 or frame["confidence"] < MIN_FRAME_CONFIDENCE:
            self.missing_frames = min(self.missing_frames + 1, MISSING_GRACE_FRAMES + 1)
            if self.missing_frames <= MISSING_GRACE_FRAMES and self.state != 0 and self.count > 0:
                return self.state, self.count, False, None
            self.state = 1  # no_person
            self.has_smoothed_y = False
            self.is_airborne = False
            self.airborne_frames = 0
            self.cooldown_frames = 0
            self._reset_cycle_tracking()
            self._reset_cooldown_tracking()
            return self.state, self.count, False, None

        self.missing_frames = 0
        self.body_height = frame["body_height"]

        self._accumulate_stats(frame, kpts)

        if not self.has_smoothed_y:
            self.smoothed_y = frame["center_y"]
            self.previous_smoothed_y = self.smoothed_y
            self.baseline_y = frame["center_y"]
            self.peak_y = frame["center_y"]
            self.has_smoothed_y = True
            self._reset_cycle_tracking()
            self._reset_cooldown_tracking()
            self.ft_start_frame = frame
            self.ft_min_y = frame["center_y"]
            self.ft_max_y = frame["center_y"]
            self.ft_cycle_frames = 1
            self.state = 3 if self.count > 0 else 2
            return self.state, self.count, False, None

        self.previous_smoothed_y = self.smoothed_y
        conf_alpha = max(0.0, min(frame["confidence"], 1.0))
        alpha = SMOOTHING_ALPHA_MIN + (SMOOTHING_ALPHA_MAX - SMOOTHING_ALPHA_MIN) * conf_alpha
        self.smoothed_y = self.smoothed_y * (1 - alpha) + frame["center_y"] * alpha
        velocity_y = self.smoothed_y - self.previous_smoothed_y

        min_amp = max(MIN_AMPLITUDE_PIXELS, self.body_height * MIN_AMPLITUDE_RATIO)
        return_margin = max(4.0, min_amp * RETURN_AMPLITUDE_RATIO)

        # Cooldown
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            abs_vel = abs(velocity_y)
            if not self.cd_cycle_active:
                if velocity_y < -0.5 and self.smoothed_y < self.baseline_y - min_amp * 0.6:
                    self.cd_cycle_active = True
                    self.cd_cycle_frames = 1
                    self.cd_min_y = self.smoothed_y
                    self.cd_peak_vel = abs_vel
                    self.cd_rise_frames = 1
                    self.cd_fall_frames = 0
            else:
                self.cd_cycle_frames += 1
                if self.smoothed_y < self.cd_min_y:
                    self.cd_min_y = self.smoothed_y
                if abs_vel > self.cd_peak_vel:
                    self.cd_peak_vel = abs_vel
                if velocity_y < 0:
                    self.cd_rise_frames += 1
                else:
                    self.cd_fall_frames += 1
                cd_amp = self.baseline_y - self.cd_min_y
                cd_return = max(4.0, cd_amp * RETURN_AMPLITUDE_RATIO)
                if (self.cd_cycle_frames >= 4 and cd_amp >= min_amp * 0.6 and
                        velocity_y > -0.5 and self.smoothed_y >= self.baseline_y - cd_return):
                    # Negative sample (cooldown blocked)
                    self.ft_cycle_frames = self.cd_cycle_frames
                    self.ft_rise_frames = self.cd_rise_frames
                    self.ft_fall_frames = self.cd_fall_frames
                    self.ft_min_y = self.cd_min_y
                    self.ft_max_y = self.baseline_y
                    self.ft_amplitude_px = cd_amp
                    self.ft_peak_velocity = self.cd_peak_vel
                    self.last_features = self._compute_features()
                    self.cycle_just_completed = True
                    self._reset_cooldown_tracking()

        # Airborne detection
        if not self.is_airborne:
            if self.cooldown_frames == 0 and velocity_y < -0.5 and self.smoothed_y <= self.baseline_y - min_amp:
                self.is_airborne = True
                self.airborne_frames = 1
                self.peak_y = self.smoothed_y
            else:
                if self.smoothed_y > self.baseline_y:
                    self.baseline_y = self.baseline_y * 0.86 + self.smoothed_y * 0.14
                else:
                    self.baseline_y = self.baseline_y * 0.995 + self.smoothed_y * 0.005
        else:
            self.airborne_frames += 1
            if self.smoothed_y < self.peak_y:
                self.peak_y = self.smoothed_y
            airborne_amp = self.baseline_y - self.peak_y
            if (self.airborne_frames >= MIN_AIRBORNE_FRAMES and airborne_amp >= min_amp and
                    velocity_y > -0.5 and self.smoothed_y >= self.baseline_y - return_margin):
                # Positive cycle completed
                self.ft_amplitude_px = max(0.0, self.ft_max_y - self.ft_min_y)
                self.last_features = self._compute_features()
                self.cycle_just_completed = True
                self.count += 1
                self.cooldown_frames = COUNT_COOLDOWN_FRAMES
                self.is_airborne = False
                self.airborne_frames = 0
                self.baseline_y = self.baseline_y * 0.35 + self.smoothed_y * 0.65
                self.peak_y = self.smoothed_y
                self._reset_cycle_tracking()
                self.ft_start_frame = frame
                self.ft_min_y = frame["center_y"]
                self.ft_max_y = frame["center_y"]

        self.state = 3 if self.count > 0 else 2
        return self.state, self.count, self.cycle_just_completed, self.last_features


# ---------------------------------------------------------------------------
# YOLO inference helper
# ---------------------------------------------------------------------------
def get_pose_results(model, frame_bgr: np.ndarray, imgsz: int = 640, conf: float = 0.25):
    """Run YOLO11-pose and return list of (bbox_xyxy, kpts_17x3)."""
    results = model(frame_bgr, imgsz=imgsz, conf=conf, verbose=False, device=model.device)
    out = []
    if results[0].boxes is None:
        return out
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if results[0].keypoints is None:
        return out
    kpts_all = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)
    for i in range(len(boxes)):
        out.append((boxes[i], kpts_all[i]))
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_overlay(img, state, count, mlp_score, mlp_verdict, cycle_just_completed, threshold):
    h, w = img.shape[:2]
    # State text
    state_names = {0: "INACTIVE", 1: "NO PERSON", 2: "READY", 3: "COUNTING"}
    state_str = state_names.get(state, "?")
    color = (0, 255, 0) if state == 3 else (0, 165, 255) if state == 2 else (0, 0, 255)
    cv2.putText(img, f"State: {state_str}  Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # MLP info
    if mlp_score is not None:
        verdict_str = "KEEP" if mlp_verdict else "REJECT"
        v_color = (0, 255, 0) if mlp_verdict else (0, 0, 255)
        cv2.putText(img, f"MLP: {mlp_score:.3f} ({verdict_str})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, v_color, 2)

    # Flash on cycle completion
    if cycle_just_completed:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate JumpRope MLP on video.")
    p.add_argument("--video", type=Path, default="VID_20260615_092921.mp4", help="Input video file.")
    p.add_argument("--model", type=str, default="yolo11n-pose.pt", help="YOLO11-pose weights.")
    p.add_argument("--mlp", type=Path, default=Path("training/jumprope_classifier/exports/jumprope_mlp.bin"))
    p.add_argument("--output", type=Path, default=None, help="Output annotated video.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--threshold", type=float, default=0.995)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no-mlp", action="store_true", help="Disable MLP gating (counter-only mode).")
    p.add_argument("--save-csv", type=Path, default=None, help="Save per-cycle features + labels to CSV.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not HAS_ULTRALYTICS:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return 1

    # Load models
    print(f"Loading YOLO: {args.model}")
    model = YOLO(args.model)
    model.to(args.device)

    mlp = None
    if not args.no_mlp:
        if not args.mlp.exists():
            print(f"ERROR: MLP weights not found: {args.mlp}")
            return 1
        print(f"Loading MLP: {args.mlp}")
        mlp = JumpRopeMLP(str(args.mlp))
    else:
        print("MLP disabled — counter-only mode")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))  # 0, 90, 180, 270
    print(f"Video: {w}x{h} @ {fps:.1f} fps, {total_frames} frames, rotation={rotation}")

    # Handle rotation metadata from mobile video recordings.
    # OpenCV does NOT auto-rotate based on the metadata tag; we must do it manually.
    def rotate_frame(frame):
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    # After rotation the effective dimensions may swap
    rot_w, rot_h = (h, w) if rotation in (90, 270) else (w, h)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (rot_w, rot_h))
        print(f"Output: {args.output} ({rot_w}x{rot_h})")

    csv_file = None
    csv_writer = None
    if args.save_csv:
        csv_file = open(str(args.save_csv), "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "label", "mlp_score", "mlp_verdict"] + FEATURE_NAMES)

    counter = JumpRopeCounter()
    primary_bbox = None
    primary_missing = 0

    frame_idx = 0
    # Stats
    total_cycles = 0
    mlp_kept = 0
    mlp_rejected = 0

    print("\nProcessing frames...")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Apply rotation so the person appears upright
        frame_bgr = rotate_frame(frame_bgr)
        h_orig, w_orig = frame_bgr.shape[:2]

        # YOLO inference
        detections = get_pose_results(model, frame_bgr, imgsz=args.imgsz, conf=args.conf)

        # Single-person tracking (largest bbox)
        best_kpts = None
        best_bbox_h = 0.0
        best_bbox_xyxy = None
        for bbox_xyxy, kpts in detections:
            bh = bbox_xyxy[3] - bbox_xyxy[1]
            if bh > best_bbox_h:
                best_bbox_h = bh
                best_kpts = kpts
                best_bbox_xyxy = bbox_xyxy

        if best_kpts is not None:
            primary_bbox = best_bbox_xyxy
            primary_missing = 0
            # Ultralytics returns keypoints in original image coordinates.
            # The Android NCNN pipeline uses model-input-size normalized
            # coordinates (typically 320 or 640).  Re-scale kpts to match
            # the training-data scale so that body_height and all ratios
            # are consistent with what the MLP was trained on.
            h_orig, w_orig = frame_bgr.shape[:2]
            scale_x = args.imgsz / float(w_orig)
            scale_y = args.imgsz / float(h_orig)
            best_kpts_scaled = best_kpts.copy()
            best_kpts_scaled[:, 0] *= scale_x
            best_kpts_scaled[:, 1] *= scale_y
            best_bbox_h_scaled = best_bbox_h * scale_y
            best_bbox_xyxy_scaled = best_bbox_xyxy.copy()
            best_bbox_xyxy_scaled[0] *= scale_x; best_bbox_xyxy_scaled[2] *= scale_x
            best_bbox_xyxy_scaled[1] *= scale_y; best_bbox_xyxy_scaled[3] *= scale_y
        else:
            primary_missing += 1

        # Build frame (use scaled coords for feature extraction consistency)
        if primary_bbox is not None and primary_missing <= MISSING_GRACE_FRAMES:
            kpts_for_frame = best_kpts_scaled if best_kpts is not None else np.zeros((17, 3))
            bh_for_frame = best_bbox_h_scaled if best_kpts is not None else 0.0
            frame_data = make_frame_from_pose(kpts_for_frame, bh_for_frame)
            if not frame_data["has_person"]:
                frame_data["has_person"] = True  # keep tracking with last known bbox
                frame_data["center_y"] = counter.smoothed_y if counter.has_smoothed_y else 0
                frame_data["body_height"] = counter.body_height
                frame_data["confidence"] = 0.1
        else:
            frame_data = {"has_person": False, "center_y": 0.0, "body_height": 0.0, "confidence": 0.0}

        # Counter update (with scaled kpts for knee/ankle/symmetry features)
        prev_count = counter.count
        kpts_for_counter = best_kpts_scaled if best_kpts is not None else None
        state, count, cycle_done, features = counter.update(frame_data, kpts_for_counter)

        # MLP gating
        mlp_score = None
        mlp_verdict = None
        if cycle_done and features is not None:
            total_cycles += 1
            if mlp is not None:
                mlp_score = mlp.predict(features)
                mlp_verdict = mlp_score >= args.threshold
                if not mlp_verdict:
                    # Undo the count increment
                    counter.count = prev_count
                    count = prev_count
                    mlp_rejected += 1
                else:
                    mlp_kept += 1

            # Save to CSV
            if csv_writer is not None:
                label = 1 if (mlp_verdict is None or mlp_verdict) else 0
                csv_writer.writerow(
                    [frame_idx, label,
                     f"{mlp_score:.4f}" if mlp_score is not None else "",
                     1 if mlp_verdict else 0 if mlp_verdict is not None else "",
                     ] + [f"{v:.4f}" for v in features]
                )

        # Draw
        if best_bbox_xyxy is not None:
            x1, y1, x2, y2 = map(int, best_bbox_xyxy)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if best_kpts is not None:
            for kp in best_kpts:
                if kp[2] > 0.2:
                    cv2.circle(frame_bgr, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)

        draw_overlay(frame_bgr, state, count, mlp_score, mlp_verdict, cycle_done, args.threshold)

        if writer:
            writer.write(frame_bgr)

        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}/{total_frames}  count={count}  "
                  f"cycles={total_cycles}  kept={mlp_kept}  rejected={mlp_rejected}")

    cap.release()
    if writer:
        writer.release()
    if csv_file:
        csv_file.close()

    print(f"\n=== Done ===")
    print(f"Total frames: {frame_idx}")
    print(f"Final count (after MLP): {counter.count}")
    print(f"Total cycles: {total_cycles}")
    if mlp is not None:
        print(f"MLP kept: {mlp_kept}  MLP rejected: {mlp_rejected}")
        print(f"MLP accept rate: {mlp_kept / max(total_cycles, 1) * 100:.1f}%")
    if args.output:
        print(f"Output video: {args.output}")
    if args.save_csv:
        print(f"CSV saved: {args.save_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
