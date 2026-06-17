"""JumpRope MLP video evaluator (精简版)

Usage:
    python eval_video_simple.py --video input.mp4 [--output out.mp4] [--threshold 0.95]
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config (must match training)
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
MLP_TOTAL_PARAMS = 353

# Counter constants
MISSING_GRACE = 15
COOLDOWN = 10
MIN_AIRBORNE = 2
SMOOTH_MIN, SMOOTH_MAX = 0.28, 0.48
MIN_AMP_RATIO, MIN_AMP_PX = 0.045, 10.0
RETURN_RATIO = 0.55
KP_CONF, MIN_FRAME_CONF = 0.20, 0.18
TORSO_SCALE = 2.8

# COCO keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
class JumpRopeMLP:
    def __init__(self, bin_path: str):
        blob = np.fromfile(bin_path, dtype=np.float32)
        if len(blob) != MLP_TOTAL_PARAMS:
            raise ValueError(f"Expected {MLP_TOTAL_PARAMS} params, got {len(blob)}")
        o = 0
        self.w1 = blob[o:o+192].reshape(12,16); o+=192
        self.b1 = blob[o:o+16]; o+=16
        self.w2 = blob[o:o+128].reshape(16,8); o+=128
        self.b2 = blob[o:o+8]; o+=8
        self.w3 = blob[o:o+8]; o+=8
        self.b3 = float(blob[o])
        self.means = np.array([s[0] for s in NORMALIZE_STATS], dtype=np.float32)
        self.stds = np.array([s[1] for s in NORMALIZE_STATS], dtype=np.float32)

    def predict(self, features: np.ndarray) -> float:
        x = (features - self.means) / self.stds
        h1 = np.maximum(0, x @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        logit = float(h2 @ self.w3 + self.b3)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))


# ---------------------------------------------------------------------------
# Pose → Frame
# ---------------------------------------------------------------------------
def make_frame(kpts: np.ndarray, bbox_h: float) -> Dict:
    def avg_y(indices):
        vals = [kpts[i,1] for i in indices if kpts[i,2] >= KP_CONF]
        return np.mean(vals) if vals else None

    sh_y, hi_y, an_y = avg_y([L_SHOULDER,R_SHOULDER]), avg_y([L_HIP,R_HIP]), avg_y([L_ANKLE,R_ANKLE])
    sh_conf = np.mean([kpts[L_SHOULDER,2], kpts[R_SHOULDER,2]]) if kpts[L_SHOULDER,2] >= KP_CONF and kpts[R_SHOULDER,2] >= KP_CONF else 0

    center_y, conf = 0.0, 0.0
    if sh_y is not None and hi_y is not None:
        center_y, conf = sh_y * 0.35 + hi_y * 0.65, sh_conf
    elif hi_y is not None:
        center_y, conf = hi_y, sh_conf
    elif sh_y is not None and an_y is not None:
        center_y, conf = (sh_y + an_y) * 0.5, sh_conf
    elif sh_y is not None and bbox_h > 1:
        center_y, conf = sh_y, sh_conf * 0.80
    else:
        return {"has_person": False, "center_y": 0.0, "body_height": 0.0, "confidence": 0.0}

    pose_h = 0.0
    if sh_y is not None and an_y is not None and an_y > sh_y:
        pose_h = an_y - sh_y
    elif sh_y is not None and hi_y is not None:
        torso = abs(hi_y - sh_y)
        if torso > 10:
            pose_h = torso * TORSO_SCALE

    body_h = bbox_h
    if pose_h > 1:
        if body_h <= 1:
            body_h = pose_h
        else:
            lo, hi = pose_h * 0.70, pose_h * 1.65
            body_h = pose_h if (body_h < lo or body_h > hi) else body_h * 0.5 + pose_h * 0.5

    if body_h <= 1:
        return {"has_person": False, "center_y": 0.0, "body_height": 0.0, "confidence": 0.0}
    return {"has_person": True, "center_y": center_y, "body_height": body_h, "confidence": conf}


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------
class JumpRopeCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = self.state = self.missing = self.cooldown = 0
        self.has_smooth = self.is_airborne = False
        self.airborne_frames = 0
        self.smooth_y = self.prev_y = self.baseline_y = self.peak_y = self.body_h = 0.0
        self._reset_tracking()

    def _reset_tracking(self):
        self.ft_frames = self.ft_rise = self.ft_fall = 0
        self.ft_min_y, self.ft_max_y = 1e9, -1e9
        self.ft_peak_vel = self.ft_max_knee = self.ft_max_ankle = 0.0
        self.ft_conf_sum = self.ft_bh_sum = self.ft_lr_sum = 0.0
        self.ft_conf_n = self.ft_bh_n = self.ft_lr_n = 0
        self.ft_start = None
        self.cycle_done = False
        self.last_features = None

    def _reset_cooldown(self):
        self.cd_active = False
        self.cd_frames = 0
        self.cd_min_y = 1e9
        self.cd_peak_vel = 0.0
        self.cd_rise = self.cd_fall = 0

    def _accumulate(self, frame, kpts):
        self.ft_frames += 1
        self.ft_min_y = min(self.ft_min_y, frame["center_y"])
        self.ft_max_y = max(self.ft_max_y, frame["center_y"])
        vel = abs(self.smooth_y - self.prev_y)
        if vel > self.ft_peak_vel:
            self.ft_peak_vel = vel
        if self.is_airborne:
            self.ft_rise += 1
        else:
            self.ft_fall += 1

        if frame["body_height"] > 1:
            self.ft_bh_sum += frame["body_height"]
            self.ft_bh_n += 1
        if frame["confidence"] > 0:
            self.ft_conf_sum += frame["confidence"]
            self.ft_conf_n += 1

        if kpts is not None and kpts.shape[0] >= 17:
            for side in range(2):
                hip, knee, ankle = 11+side, 13+side, 15+side
                if kpts[hip,2] > 0.2 and kpts[knee,2] > 0.2:
                    self.ft_max_knee = max(self.ft_max_knee, abs(kpts[knee,1] - kpts[hip,1]))
                if self.ft_start is not None and kpts[ankle,2] > 0.2:
                    self.ft_max_ankle = max(self.ft_max_ankle, self.ft_start["center_y"] - kpts[ankle,1])
            for a, b in [(5,6), (11,12)]:
                if kpts[a,2] > 0.2 and kpts[b,2] > 0.2:
                    self.ft_lr_sum += abs(kpts[a,1] - kpts[b,1])
                    self.ft_lr_n += 1

    def _compute_features(self):
        bh = (self.ft_bh_sum / max(self.ft_bh_n,1)) if self.ft_bh_n > 0 else self.body_h
        if bh <= 1:
            bh = 1.0
        dur = max(1, self.ft_frames)
        amp_px = max(0.0, self.ft_max_y - self.ft_min_y)
        f = np.zeros(FEATURE_DIM, dtype=np.float32)
        f[0] = float(dur)
        f[1] = amp_px / bh
        f[2] = self.ft_rise / dur
        f[3] = self.ft_fall / dur
        f[4] = (self.ft_rise / self.ft_fall) if self.ft_fall > 0 else 1.0
        f[5] = self.ft_max_knee / bh
        f[6] = self.ft_max_ankle / bh
        f[7] = (self.ft_conf_sum / self.ft_conf_n) if self.ft_conf_n > 0 else 0.0
        f[8] = ((self.ft_lr_sum / self.ft_lr_n) / bh) if self.ft_lr_n > 0 else 0.0
        f[9] = self.ft_peak_vel / bh
        f[10] = amp_px
        f[11] = bh
        return f

    def update(self, frame, kpts=None):
        self.cycle_done = False
        self.last_features = None

        if not frame["has_person"] or frame["body_height"] <= 1 or frame["confidence"] < MIN_FRAME_CONF:
            self.missing = min(self.missing + 1, MISSING_GRACE + 1)
            if self.missing <= MISSING_GRACE and self.state != 0 and self.count > 0:
                return self.state, self.count, False, None
            self.state = 1
            self.has_smooth = self.is_airborne = False
            self.airborne_frames = 0
            self.cooldown = 0
            self._reset_tracking()
            return self.state, self.count, False, None

        self.missing = 0
        self.body_h = frame["body_height"]
        self._accumulate(frame, kpts)

        if not self.has_smooth:
            self.smooth_y = self.prev_y = self.baseline_y = self.peak_y = frame["center_y"]
            self.has_smooth = True
            self._reset_tracking()
            self._reset_cooldown()
            self.ft_start = frame
            self.ft_min_y = self.ft_max_y = frame["center_y"]
            self.ft_frames = 1
            self.state = 3 if self.count > 0 else 2
            return self.state, self.count, False, None

        self.prev_y = self.smooth_y
        alpha = SMOOTH_MIN + (SMOOTH_MAX - SMOOTH_MIN) * max(0.0, min(frame["confidence"], 1.0))
        self.smooth_y = self.smooth_y * (1 - alpha) + frame["center_y"] * alpha
        velocity = self.smooth_y - self.prev_y

        min_amp = max(MIN_AMP_PX, self.body_h * MIN_AMP_RATIO)
        ret_margin = max(4.0, min_amp * RETURN_RATIO)

        # Cooldown tracking
        if self.cooldown > 0:
            self.cooldown -= 1
            abs_vel = abs(velocity)
            if not self.cd_active:
                if velocity < -0.5 and self.smooth_y < self.baseline_y - min_amp * 0.6:
                    self.cd_active = True
                    self.cd_frames = 1
                    self.cd_min_y = self.smooth_y
                    self.cd_peak_vel = abs_vel
                    self.cd_rise, self.cd_fall = 1, 0
            else:
                self.cd_frames += 1
                self.cd_min_y = min(self.cd_min_y, self.smooth_y)
                self.cd_peak_vel = max(self.cd_peak_vel, abs_vel)
                if velocity < 0:
                    self.cd_rise += 1
                else:
                    self.cd_fall += 1
                cd_amp = self.baseline_y - self.cd_min_y
                cd_ret = max(4.0, cd_amp * RETURN_RATIO)
                if (self.cd_frames >= 4 and cd_amp >= min_amp * 0.6 and
                        velocity > -0.5 and self.smooth_y >= self.baseline_y - cd_ret):
                    self.ft_frames = self.cd_frames
                    self.ft_rise, self.ft_fall = self.cd_rise, self.cd_fall
                    self.ft_min_y, self.ft_max_y = self.cd_min_y, self.baseline_y
                    self.ft_peak_vel = self.cd_peak_vel
                    self.last_features = self._compute_features()
                    self.cycle_done = True
                    self._reset_cooldown()

        # Airborne detection
        if not self.is_airborne:
            if self.cooldown == 0 and velocity < -0.5 and self.smooth_y <= self.baseline_y - min_amp:
                self.is_airborne = True
                self.airborne_frames = 1
                self.peak_y = self.smooth_y
            else:
                if self.smooth_y > self.baseline_y:
                    self.baseline_y = self.baseline_y * 0.86 + self.smooth_y * 0.14
                else:
                    self.baseline_y = self.baseline_y * 0.995 + self.smooth_y * 0.005
        else:
            self.airborne_frames += 1
            if self.smooth_y < self.peak_y:
                self.peak_y = self.smooth_y
            if (self.airborne_frames >= MIN_AIRBORNE and
                    self.baseline_y - self.peak_y >= min_amp and
                    velocity > -0.5 and self.smooth_y >= self.baseline_y - ret_margin):
                self.last_features = self._compute_features()
                self.cycle_done = True
                self.count += 1
                self.cooldown = COOLDOWN
                self.is_airborne = False
                self.airborne_frames = 0
                self.baseline_y = self.baseline_y * 0.35 + self.smooth_y * 0.65
                self.peak_y = self.smooth_y
                self._reset_tracking()
                self.ft_start = frame
                self.ft_min_y = self.ft_max_y = frame["center_y"]

        self.state = 3 if self.count > 0 else 2
        return self.state, self.count, self.cycle_done, self.last_features


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_overlay(img, state, count, mlp_score, mlp_verdict, cycle_done):
    h, w = img.shape[:2]
    color = (0, 255, 0) if state == 3 else (0, 165, 255) if state == 2 else (0, 0, 255)
    cv2.putText(img, f"State: {'COUNTING' if state==3 else 'READY' if state==2 else 'NO PERSON'}  Count: {count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if mlp_score is not None:
        verdict = "KEEP" if mlp_verdict else "REJECT"
        v_color = (0, 255, 0) if mlp_verdict else (0, 0, 255)
        cv2.putText(img, f"MLP: {mlp_score:.3f} ({verdict})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, v_color, 2)
    if cycle_done:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="JumpRope MLP video evaluator (simple)")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default=None, help="Output video path (optional)")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt", help="YOLO model")
    parser.add_argument("--mlp", type=str, default="training/jumprope_classifier/exports/jumprope_mlp.bin")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-csv", type=str, default=None, help="Save features to CSV")
    args = parser.parse_args()

    # Load models
    print(f"Loading YOLO: {args.model}")
    model = YOLO(args.model).to(args.device)
    print(f"Loading MLP: {args.mlp}")
    mlp = JumpRopeMLP(args.mlp)

    # Open video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames, rotation={rotation}")

    # Rotation handler
    def rotate(frame):
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    rot_w, rot_h = (h, w) if rotation in (90, 270) else (w, h)

    # Output video writer
    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (rot_w, rot_h))
        print(f"Output: {args.output}")

    # CSV writer
    csv_file = None
    if args.save_csv:
        csv_file = open(args.save_csv, "w", newline="", encoding="utf-8")
        csv.writer(csv_file).writerow(["frame", "label", "mlp_score", "mlp_verdict"] + FEATURE_NAMES)

    counter = JumpRopeCounter()
    primary_bbox = None
    primary_missing = 0
    total_cycles = kept = rejected = 0

    print("\nProcessing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = rotate(frame)
        h_img, w_img = frame.shape[:2]

        # YOLO inference
        results = model(frame, imgsz=640, conf=0.25, verbose=False, device=args.device)
        detections = []
        if results[0].boxes is not None and results[0].keypoints is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            kpts = results[0].keypoints.data.cpu().numpy()
            detections = [(boxes[i], kpts[i]) for i in range(len(boxes))]

        # Track largest person
        best_kpts, best_bbox_h, best_bbox = None, 0.0, None
        for bbox, kpts in detections:
            bh = bbox[3] - bbox[1]
            if bh > best_bbox_h:
                best_bbox_h, best_kpts, best_bbox = bh, kpts, bbox

        if best_kpts is not None:
            primary_bbox = best_bbox
            primary_missing = 0
            scale_x, scale_y = 640.0 / w_img, 640.0 / h_img
            kpts_scaled = best_kpts.copy()
            kpts_scaled[:, 0] *= scale_x
            kpts_scaled[:, 1] *= scale_y
            bbox_h_scaled = best_bbox_h * scale_y
        else:
            primary_missing += 1

        # Build frame
        if primary_bbox is not None and primary_missing <= MISSING_GRACE:
            frame_data = make_frame(kpts_scaled if best_kpts is not None else np.zeros((17, 3)),
                                    bbox_h_scaled if best_kpts is not None else 0.0)
            if not frame_data["has_person"]:
                frame_data = {"has_person": True, "center_y": counter.smooth_y if counter.has_smooth else 0,
                              "body_height": counter.body_h, "confidence": 0.1}
        else:
            frame_data = {"has_person": False, "center_y": 0.0, "body_height": 0.0, "confidence": 0.0}

        # Counter + MLP
        prev_count = counter.count
        state, count, cycle_done, features = counter.update(frame_data, kpts_scaled if best_kpts is not None else None)

        mlp_score, mlp_verdict = None, None
        if cycle_done and features is not None:
            total_cycles += 1
            mlp_score = mlp.predict(features)
            mlp_verdict = mlp_score >= args.threshold
            if not mlp_verdict:
                counter.count = prev_count
                count = prev_count
                rejected += 1
            else:
                kept += 1

            if csv_file:
                csv.writer(csv_file).writerow([int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                                              1 if mlp_verdict else 0, f"{mlp_score:.4f}",
                                              1 if mlp_verdict else 0] + [f"{v:.4f}" for v in features])

        # Draw
        if best_bbox is not None:
            x1, y1, x2, y2 = map(int, best_bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if best_kpts is not None:
            for kp in best_kpts:
                if kp[2] > 0.2:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)
        draw_overlay(frame, state, count, mlp_score, mlp_verdict, cycle_done)

        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    if csv_file:
        csv_file.close()

    print(f"\n=== Result ===")
    print(f"Final count: {counter.count}")
    print(f"Total cycles: {total_cycles}")
    print(f"Kept: {kept}  Rejected: {rejected}")
    print(f"Accept rate: {kept/max(total_cycles,1)*100:.1f}%")


if __name__ == "__main__":
    main()
