"""JumpRope counter test script — ports the improved C++ algorithm to Python.

Tests both with and without MLP gating against a video file.
Expected: 10 jumps, then squats (squats should NOT count).

Usage:
    python test_jumprope.py --video VID_20260615_092921.mp4
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# Try ultralytics; fall back to a stub if unavailable
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# ======================================================================
# Constants — MUST match the improved jumprope_counter.cpp
# ======================================================================
MISSING_GRACE = 15
COOLDOWN = 5
MIN_AIRBORNE = 2
SMOOTH_MIN = 0.35
SMOOTH_MAX = 0.55
MIN_AMP_RATIO = 0.025
MIN_AMP_PX = 4.0
RETURN_RATIO = 0.50  # legacy, kept for compatibility
RETURN_BODY_RATIO = 0.06  # return margin = 6% of body height
VEL_THRESH = -0.25
MIN_CYCLE = 1
MAX_CYCLE = 30
KP_CONF = 0.15
MIN_FRAME_CONF = 0.12
TORSO_SCALE = 2.8

# Ankle-lift gate
MIN_ANKLE_LIFT_RATIO = 0.018
ANKLE_KP_CONF = 0.15

# Crouch duration gate (rejects squat-stand patterns)
# At airborne entry, squat-stand has large baseline-smooth_y displacement
# because baseline drifted up during the squat hold.
# Jumps: displacement ≈ 35% of min_amp (entry threshold)
# Squat-stand: displacement ≈ 300% of min_amp (baseline far above smooth_y)
# Squat amplitude gate (rejects squat-stand at landing, not entry)
# Jump cycles: air_amp / body_h ≈ 0.15-0.20
# Squat cycles: air_amp / body_h ≈ 0.25-0.40
MAX_JUMP_AMP_RATIO = 0.23  # max plausible jump amplitude as fraction of body height

# Standing-still detector (baseline drift recovery)
STANDING_STILL_FRAMES = 15       # ~0.5s at 30fps
STANDING_STILL_VEL = 0.4         # max |velocity| to count as "still"

# COCO indices
L_SH, R_SH = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16

# MLP config
FEATURE_DIM = 12
MLP_PARAMS = 353
NORM_STATS = [
    (22.0, 10.0), (0.08, 0.04), (0.40, 0.12), (0.60, 0.12),
    (0.80, 0.40), (0.10, 0.06), (0.05, 0.04), (0.50, 0.20),
    (0.03, 0.02), (0.04, 0.02), (30.0, 20.0), (350.0, 150.0),
]


# ======================================================================
# MLP
# ======================================================================
class MLP:
    def __init__(self, bin_path):
        blob = np.fromfile(bin_path, dtype=np.float32)
        assert len(blob) == MLP_PARAMS, f"Expected {MLP_PARAMS}, got {len(blob)}"
        o = 0
        self.w1 = blob[o:o+192].reshape(12, 16); o += 192
        self.b1 = blob[o:o+16]; o += 16
        self.w2 = blob[o:o+128].reshape(16, 8); o += 128
        self.b2 = blob[o:o+8]; o += 8
        self.w3 = blob[o:o+8]; o += 8
        self.b3 = float(blob[o])
        self.means = np.array([s[0] for s in NORM_STATS], dtype=np.float32)
        self.stds = np.array([s[1] for s in NORM_STATS], dtype=np.float32)

    def predict(self, f):
        x = (f - self.means) / self.stds
        h1 = np.maximum(0, x @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        logit = float(h2 @ self.w3 + self.b3)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))


# ======================================================================
# Pose → Frame  (pure torso signal, NO ankle blending)
# ======================================================================
def make_frame(kpts, bbox_h):
    def avg_y(indices):
        vals = [kpts[i, 1] for i in indices if kpts[i, 2] >= KP_CONF]
        return np.mean(vals) if vals else None

    def avg_conf(indices):
        vals = [kpts[i, 2] for i in indices if kpts[i, 2] >= KP_CONF]
        return np.mean(vals) if vals else 0.0

    sh_y = avg_y([L_SH, R_SH])
    hi_y = avg_y([L_HIP, R_HIP])
    an_y = avg_y([L_ANK, R_ANK])

    center_y, conf = 0.0, 0.0
    if sh_y is not None and hi_y is not None:
        # Pure torso centre — NO ankle blending
        center_y = sh_y * 0.35 + hi_y * 0.65
        conf = (avg_conf([L_SH, R_SH]) + avg_conf([L_HIP, R_HIP])) * 0.5
    elif hi_y is not None:
        center_y, conf = hi_y, avg_conf([L_HIP, R_HIP])
    elif sh_y is not None and an_y is not None:
        center_y = sh_y * 0.50 + an_y * 0.50
        conf = (avg_conf([L_SH, R_SH]) + avg_conf([L_ANK, R_ANK])) * 0.5
    elif sh_y is not None and bbox_h > 1:
        center_y, conf = sh_y, avg_conf([L_SH, R_SH]) * 0.80
    else:
        return {"has": False, "cy": 0.0, "bh": 0.0, "conf": 0.0}

    pose_h = 0.0
    if sh_y is not None and an_y is not None and an_y > sh_y:
        pose_h = an_y - sh_y
    elif sh_y is not None and hi_y is not None:
        torso = abs(hi_y - sh_y)
        if torso > 10:
            pose_h = torso * TORSO_SCALE

    bh = bbox_h
    if pose_h > 1:
        if bh <= 1:
            bh = pose_h
        else:
            lo, hi = pose_h * 0.70, pose_h * 1.65
            bh = pose_h if (bh < lo or bh > hi) else bh * 0.5 + pose_h * 0.5

    if bh <= 1:
        return {"has": False, "cy": 0.0, "bh": 0.0, "conf": 0.0}
    return {"has": True, "cy": center_y, "bh": bh, "conf": conf}


# ======================================================================
# Counter — faithful port of the improved C++ JumpRopeCounter
# ======================================================================
class Counter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = self.state = self.missing = self.cooldown = 0
        self.has_smooth = self.is_airborne = False
        self.air_n = 0  # airborne frames
        self.sm_y = self.prev_y = self.baseline = self.peak_y = self.body_h = 0.0

        # ankle-lift gate
        self.has_ankle_ground = False
        self.ankle_ground_y = 0.0
        self.frame_ankle_y = 0.0
        self.has_frame_ankle = False
        self.air_ankle_seen = False
        self.peak_ankle_lift = 0.0
        self.air_first_down = False

        # crouch duration gate (diagnostic only)
        self.crouch_frames = 0

        # standing-still detector
        self.standing_still_frames = 0

        # stable body height at airborne entry (for ratio checks)
        self.entry_body_h = 0.0

        self._reset_cycle()
        self._reset_cd()

    def _reset_cycle(self):
        self.ft_n = self.ft_rise = self.ft_fall = 0
        self.ft_min = 1e9; self.ft_max = -1e9
        self.ft_peak_vel = self.ft_max_knee = self.ft_max_ankle = 0.0
        self.ft_conf_s = self.ft_bh_s = self.ft_lr_s = 0.0
        self.ft_conf_n = self.ft_bh_n = self.ft_lr_n = 0
        self.ft_start_cy = None
        self.cycle_done = False
        self.last_feat = None

    def _reset_cd(self):
        self.cd_on = False
        self.cd_n = 0; self.cd_min = 1e9; self.cd_pv = 0.0
        self.cd_rise = self.cd_fall = 0

    def _get_ankle_y(self, kpts):
        """Extract mean ankle Y from keypoints (if confident enough)."""
        if kpts is None or kpts.shape[0] < 17:
            return None
        vals = []
        for idx in [L_ANK, R_ANK]:
            if kpts[idx, 2] >= ANKLE_KP_CONF:
                vals.append(kpts[idx, 1])
        return np.mean(vals) if vals else None

    def _accumulate(self, fr, kpts):
        self.ft_n += 1
        self.ft_min = min(self.ft_min, fr["cy"])
        self.ft_max = max(self.ft_max, fr["cy"])
        vel = abs(self.sm_y - self.prev_y)
        if vel > self.ft_peak_vel:
            self.ft_peak_vel = vel
        if self.is_airborne:
            self.ft_rise += 1
        else:
            self.ft_fall += 1
        if fr["bh"] > 1:
            self.ft_bh_s += fr["bh"]; self.ft_bh_n += 1
        if fr["conf"] > 0:
            self.ft_conf_s += fr["conf"]; self.ft_conf_n += 1

        if kpts is not None and kpts.shape[0] >= 17:
            for side in range(2):
                hip, knee, ankle = 11+side, 13+side, 15+side
                if kpts[hip, 2] > 0.2 and kpts[knee, 2] > 0.2:
                    self.ft_max_knee = max(self.ft_max_knee, abs(kpts[knee, 1] - kpts[hip, 1]))
                if self.ft_start_cy is not None and kpts[ankle, 2] > 0.2:
                    self.ft_max_ankle = max(self.ft_max_ankle, self.ft_start_cy - kpts[ankle, 1])
            for a, b in [(5, 6), (11, 12)]:
                if kpts[a, 2] > 0.2 and kpts[b, 2] > 0.2:
                    self.ft_lr_s += abs(kpts[a, 1] - kpts[b, 1])
                    self.ft_lr_n += 1

    def _features(self):
        bh = (self.ft_bh_s / max(self.ft_bh_n, 1)) if self.ft_bh_n > 0 else self.body_h
        if bh <= 1: bh = 1.0
        dur = max(1, self.ft_n)
        amp = max(0.0, self.ft_max - self.ft_min)
        f = np.zeros(FEATURE_DIM, dtype=np.float32)
        f[0] = float(dur)
        f[1] = amp / bh
        f[2] = self.ft_rise / dur
        f[3] = self.ft_fall / dur
        f[4] = (self.ft_rise / self.ft_fall) if self.ft_fall > 0 else 1.0
        f[5] = self.ft_max_knee / bh
        f[6] = self.ft_max_ankle / bh
        f[7] = (self.ft_conf_s / self.ft_conf_n) if self.ft_conf_n > 0 else 0.0
        f[8] = ((self.ft_lr_s / self.ft_lr_n) / bh) if self.ft_lr_n > 0 else 0.0
        f[9] = self.ft_peak_vel / bh
        f[10] = amp
        f[11] = bh
        return f

    def update(self, fr, kpts=None):
        self.cycle_done = False
        self.last_feat = None

        # Invalid frame
        if not fr["has"] or fr["bh"] <= 1 or fr["conf"] < MIN_FRAME_CONF:
            self.missing = min(self.missing + 1, MISSING_GRACE + 1)
            if self.missing <= MISSING_GRACE and self.state != 0 and self.count > 0:
                return self.state, self.count
            self.state = 1; self.has_smooth = self.is_airborne = False
            self.air_n = 0; self.cooldown = 0
            self._reset_cycle(); self._reset_cd()
            return self.state, self.count

        self.missing = 0
        self.body_h = fr["bh"]

        # Ankle position for this frame
        ank_y = self._get_ankle_y(kpts)
        if ank_y is not None:
            self.frame_ankle_y = ank_y
            self.has_frame_ankle = True
        else:
            self.has_frame_ankle = False

        self._accumulate(fr, kpts)

        # First valid frame
        if not self.has_smooth:
            self.sm_y = self.prev_y = self.baseline = self.peak_y = fr["cy"]
            self.has_smooth = True
            if self.has_frame_ankle:
                self.ankle_ground_y = self.frame_ankle_y
                self.has_ankle_ground = True
            self._reset_cycle(); self._reset_cd()
            self.ft_start_cy = fr["cy"]
            self.ft_min = self.ft_max = fr["cy"]
            self.ft_n = 0  # accumulate already counted
            self.state = 3 if self.count > 0 else 2
            return self.state, self.count

        # Smoothing
        self.prev_y = self.sm_y
        alpha = SMOOTH_MIN + (SMOOTH_MAX - SMOOTH_MIN) * max(0.0, min(fr["conf"], 1.0))
        self.sm_y = self.sm_y * (1 - alpha) + fr["cy"] * alpha
        vel = self.sm_y - self.prev_y

        min_amp = max(MIN_AMP_PX, self.body_h * MIN_AMP_RATIO)
        ret_margin = max(2.0, max(self.body_h, self.entry_body_h) * RETURN_BODY_RATIO)

        # Baseline: robust asymmetric torso EMA
        # FREEZE baseline during airborne — otherwise it drifts upward at 0.05/frame,
        # making the return-to-baseline landing check impossible during continuous jumps.
        if not self.is_airborne:
            if self.sm_y > self.baseline:
                self.baseline = self.baseline * 0.95 + self.sm_y * 0.05
            else:
                self.baseline = self.baseline * 0.998 + self.sm_y * 0.002

        # --- Standing-still detector ---
        if not self.is_airborne:
            if abs(vel) < STANDING_STILL_VEL:
                self.standing_still_frames += 1
                if self.standing_still_frames >= STANDING_STILL_FRAMES:
                    self.baseline = self.sm_y
            else:
                self.standing_still_frames = 0
        else:
            self.standing_still_frames = 0

        # Cooldown tracking
        if self.cooldown > 0:
            self.cooldown -= 1
            abs_vel = abs(vel)
            if not self.cd_on:
                if vel < VEL_THRESH and self.sm_y < self.baseline - min_amp * 0.6:
                    self.cd_on = True; self.cd_n = 1
                    self.cd_min = self.sm_y; self.cd_pv = abs_vel
                    self.cd_rise, self.cd_fall = 1, 0
            else:
                self.cd_n += 1
                self.cd_min = min(self.cd_min, self.sm_y)
                self.cd_pv = max(self.cd_pv, abs_vel)
                if vel < 0: self.cd_rise += 1
                else: self.cd_fall += 1
                cd_amp = self.baseline - self.cd_min
                cd_ret = max(2.0, cd_amp * RETURN_RATIO)
                if (self.cd_n >= 4 and cd_amp >= min_amp * 0.6 and
                        vel > VEL_THRESH and self.sm_y >= self.baseline - cd_ret):
                    self.ft_n = self.cd_n
                    self.ft_rise, self.ft_fall = self.cd_rise, self.cd_fall
                    self.ft_min, self.ft_max = self.cd_min, self.baseline
                    self.ft_peak_vel = self.cd_pv
                    self.last_feat = self._features()
                    self.cycle_done = True
                    self._reset_cd()

        # Airborne detection
        if not self.is_airborne:
            if (self.cooldown == 0 and vel < VEL_THRESH and
                    self.sm_y <= self.baseline - max(1.0, min_amp * 1.0)):
                self.is_airborne = True
                self.air_n = 1
                self.peak_y = self.sm_y
                self.air_first_down = False
                self.entry_body_h = self.body_h  # stable body height for ratio checks

                # Ankle-lift init
                self.air_ankle_seen = False
                self.peak_ankle_lift = 0.0
                if self.has_frame_ankle:
                    self.air_ankle_seen = True
                    if self.has_ankle_ground:
                        lift = self.ankle_ground_y - self.frame_ankle_y
                        if lift > self.peak_ankle_lift:
                            self.peak_ankle_lift = lift

                self._reset_cycle()
                self.ft_start_cy = fr["cy"]
                self.ft_min = self.ft_max = fr["cy"]
                self.ft_n = 0
                self._accumulate(fr, kpts)
            else:
                # Crouch duration tracking (elevated torso = crouched/squatting)
                if self.sm_y > self.baseline + min_amp:
                    self.crouch_frames += 1
                else:
                    self.crouch_frames = 0
                # Ankle ground tracking (max-based)
                if self.has_frame_ankle:
                    if not self.has_ankle_ground:
                        self.ankle_ground_y = self.frame_ankle_y
                        self.has_ankle_ground = True
                    elif self.frame_ankle_y > self.ankle_ground_y:
                        self.ankle_ground_y = self.ankle_ground_y * 0.5 + self.frame_ankle_y * 0.5
                    else:
                        self.ankle_ground_y = self.ankle_ground_y * 0.998 + self.frame_ankle_y * 0.002
        else:
            # Airborne phase
            self.air_n += 1
            if self.sm_y < self.peak_y:
                self.peak_y = self.sm_y

            # Track ankle lift
            if self.has_frame_ankle:
                self.air_ankle_seen = True
                if self.has_ankle_ground:
                    lift = self.ankle_ground_y - self.frame_ankle_y
                    if lift > self.peak_ankle_lift:
                        self.peak_ankle_lift = lift

            # First-motion-down gate: DISABLED — ratio-based version also too
            # sensitive; relies on amplitude-ratio + ankle-lift gates instead.

            air_amp = self.baseline - self.peak_y
            plausible = MIN_CYCLE <= self.air_n <= MAX_CYCLE

            # Early amplitude check: if amplitude already exceeds max jump ratio,
            # immediately discard. This prevents multi-jump phases from accumulating
            # huge amplitudes before the landing conditions are met.
            max_amp = max(self.body_h, self.entry_body_h) * MAX_JUMP_AMP_RATIO
            if air_amp > max_amp and self.air_n >= MIN_AIRBORNE:
                self._log_reject(fr, "AMP_EARLY", air_amp)
                self._discard_airborne(fr, aggressive=True)
                self.state = 3 if self.count > 0 else 2
                return self.state, self.count

            if (self.air_n >= MIN_AIRBORNE and air_amp >= min_amp and
                    vel > VEL_THRESH and self.sm_y >= self.baseline - ret_margin):

                if not plausible:
                    self._log_reject(fr, "RHYTHM", air_amp)
                    self._discard_airborne(fr, aggressive=True)
                elif air_amp > max(self.body_h, self.entry_body_h) * MAX_JUMP_AMP_RATIO:
                    self._log_reject(fr, "AMP_RATIO", air_amp)
                    self._discard_airborne(fr, aggressive=True)
                elif (self.air_ankle_seen and self.has_ankle_ground and
                      self.peak_ankle_lift > 0.0 and
                      self.peak_ankle_lift < max(2.0, self.body_h * MIN_ANKLE_LIFT_RATIO)):
                    self._log_reject(fr, "ANKLE_LIFT", air_amp)
                    self._discard_airborne(fr)
                else:
                    # Valid jump!
                    amp_px = max(0.0, self.ft_max - self.ft_min)
                    self.last_feat = self._features()
                    self.cycle_done = True
                    self.count += 1
                    self.cooldown = COOLDOWN
                    self.is_airborne = False
                    self.air_n = 0
                    self.baseline = self.baseline * 0.35 + self.sm_y * 0.65
                    self.peak_y = self.sm_y
                    self.crouch_frames = 0
                    self._reset_cycle()
                    self.ft_start_cy = fr["cy"]
                    self.ft_min = self.ft_max = fr["cy"]

        self.state = 3 if self.count > 0 else 2
        return self.state, self.count

    def _discard_airborne(self, fr, aggressive=False):
        self.is_airborne = False
        self.air_n = 0
        if aggressive:
            # Squat rejection: aggressively pull baseline back to current position
            self.baseline = self.baseline * 0.10 + self.sm_y * 0.90
        else:
            self.baseline = self.baseline * 0.35 + self.sm_y * 0.65
        self.peak_y = self.sm_y
        self.crouch_frames = 0
        self._reset_cycle()
        self.ft_start_cy = fr["cy"]
        self.ft_min = self.ft_max = fr["cy"]

    _reject_log = []
    def _log_reject(self, fr, gate, air_amp):
        """Log airborne rejection for debugging."""
        if not hasattr(self, '_reject_log'):
            self._reject_log = []
        self._reject_log.append({
            'frame': -1,  # filled in by caller if available
            'gate': gate,
            'air_n': self.air_n,
            'air_amp': air_amp,
            'peak_ankle': self.peak_ankle_lift,
            'body_h': self.body_h,
        })


# ======================================================================
# Main test
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--mlp", default="training/jumprope_classifier/exports/jumprope_mlp.bin")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # Load YOLO
    print(f"Loading YOLO: {args.model}")
    model = YOLO(args.model).to(args.device)

    # Load MLP
    mlp_path = Path(args.mlp)
    mlp = None
    if mlp_path.exists():
        mlp = MLP(str(mlp_path))
        print(f"MLP loaded: {mlp_path}")
    else:
        print(f"MLP not found at {mlp_path}, running without MLP")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames, rotation={rotation}")

    def rotate(frame):
        if rotation == 90: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if rotation == 180: return cv2.rotate(frame, cv2.ROTATE_180)
        if rotation == 270: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    rot_w = h if rotation in (90, 270) else w
    rot_h = w if rotation in (90, 270) else h

    # Output video
    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (rot_w, rot_h))

    # CSV
    csv_f = None
    if args.csv:
        csv_f = open(args.csv, "w", newline="", encoding="utf-8")
        csv.writer(csv_f).writerow([
            "frame", "raw_count", "mlp_count", "mlp_score", "mlp_verdict",
            "is_airborne", "baseline", "smooth_y", "velocity", "amplitude",
            "ankle_ground", "ankle_lift", "airborne_frames", "crouch_frames"
        ])

    counter = Counter()
    mlp_rejected = 0
    mlp_kept = 0
    raw_count = 0
    mlp_count = 0

    # Detailed event log
    events = []

    print("\n" + "=" * 60)
    print("Processing video frame by frame...")
    print("=" * 60)

    frame_idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = rotate(img)
        hi, wi = img.shape[:2]

        # YOLO inference
        results = model(img, imgsz=640, conf=0.50, verbose=False, device=args.device)
        best_kpts, best_bh, best_bbox = None, 0.0, None

        if results[0].boxes is not None and results[0].keypoints is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            kpts_all = results[0].keypoints.data.cpu().numpy()
            for i in range(len(boxes)):
                bh = boxes[i][3] - boxes[i][1]
                if bh > best_bh:
                    best_bh = bh
                    best_kpts = kpts_all[i]
                    best_bbox = boxes[i]

        # ultralytics returns keypoints & boxes in original image coords — no scaling needed
        if best_kpts is not None:
            kpts_s = best_kpts.copy()
            bh_s = best_bh
        else:
            kpts_s = None
            bh_s = 0.0

        # Build frame data
        if kpts_s is not None:
            fr = make_frame(kpts_s, bh_s)
        else:
            fr = {"has": False, "cy": 0.0, "bh": 0.0, "conf": 0.0}

        # Update counter
        prev_raw = counter.count
        state, raw_count = counter.update(fr, kpts_s)

        # MLP check
        mlp_score = None
        mlp_verdict = None
        if counter.cycle_done and counter.last_feat is not None:
            if mlp is not None:
                mlp_score = mlp.predict(counter.last_feat)
                mlp_verdict = mlp_score >= args.threshold
                if mlp_verdict:
                    mlp_kept += 1
                    mlp_count += 1
                else:
                    mlp_rejected += 1
            else:
                mlp_count = raw_count  # no MLP = raw count

        # Log jump events
        if counter.cycle_done:
            if mlp_verdict is True:
                event_type = f"JUMP (MLP kept, score={mlp_score:.3f})"
            elif mlp_verdict is False:
                event_type = f"JUMP (MLP rejected, score={mlp_score:.3f})"
            else:
                event_type = "JUMP"
            events.append((frame_idx, event_type, raw_count, mlp_count))

        # CSV logging
        if csv_f and counter.has_smooth:
            vel = counter.sm_y - counter.prev_y if counter.has_smooth else 0
            amp = counter.baseline - counter.peak_y if counter.is_airborne else 0
            csv.writer(csv_f).writerow([
                frame_idx, raw_count, mlp_count,
                f"{mlp_score:.4f}" if mlp_score is not None else "",
                str(mlp_verdict) if mlp_verdict is not None else "",
                counter.is_airborne,
                f"{counter.baseline:.2f}",
                f"{counter.sm_y:.2f}",
                f"{vel:.4f}",
                f"{amp:.2f}",
                f"{counter.ankle_ground_y:.2f}" if counter.has_ankle_ground else "",
                f"{counter.peak_ankle_lift:.2f}" if counter.is_airborne else "",
                counter.air_n if counter.is_airborne else 0,
                counter.crouch_frames,
            ])

        # Draw overlay
        if best_bbox is not None:
            x1, y1, x2, y2 = map(int, best_bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        color = (0, 255, 0) if state == 3 else (0, 165, 255) if state == 2 else (0, 0, 255)
        cv2.putText(img, f"RAW={raw_count}  MLP={mlp_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if mlp_score is not None:
            vc = (0, 255, 0) if mlp_verdict else (0, 0, 255)
            cv2.putText(img, f"MLP: {mlp_score:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, vc, 2)

        if writer:
            writer.write(img)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if csv_f:
        csv_f.close()

    # ===== Print results =====
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total frames processed: {frame_idx}")
    print(f"Effective FPS: ~{fps:.1f}")
    print()
    print(f"--- Without MLP (raw counter) ---")
    print(f"  Final raw count: {raw_count}")
    print()
    if mlp is not None:
        print(f"--- With MLP (threshold={args.threshold}) ---")
        print(f"  Final MLP count: {mlp_count}")
        print(f"  MLP kept: {mlp_kept}  rejected: {mlp_rejected}")
    print()
    print(f"--- Event log ---")
    for fidx, etype, rc, mc in events:
        t = fidx / fps
        print(f"  Frame {fidx:4d} ({t:6.1f}s): {etype:40s}  raw={rc}  mlp={mc}")
    print()

    # Print rejection log
    if counter._reject_log:
        print(f"--- Airborne rejections ({len(counter._reject_log)}) ---")
        for r in counter._reject_log:
            print(f"  gate={r['gate']:12s}  air_n={r['air_n']:2d}  amp={r['air_amp']:.1f}  "
                  f"peak_ankle={r['peak_ankle']:.1f}  body_h={r['body_h']:.0f}")
        print()

    print(f"Raw count = {raw_count}")
    if mlp is not None:
        print(f"MLP count = {mlp_count}")


if __name__ == "__main__":
    main()
