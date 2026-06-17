#include "jumprope_counter.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace
{

const int kMissingGraceFrames = 15;

// --- Cooldown: the #1 cause of missed counts for fast jumping.
// At 20-30 FPS, 10 frames = 0.33-0.5 s.  A fast jumper can land and
// launch again within that window, so the next jump is silently ignored.
// 3 frames (~0.1-0.15 s) is enough to debounce a single landing without
// swallowing the next jump.
const int kCountCooldownFrames = 5;

// Minimum airborne frames: require 2 to reject single-frame noise spikes.
const int kMinAirborneFrames = 2;

// Smoothing: alpha is the weight on the *new* sample in the EMA
// (smoothed = smoothed*(1-alpha) + new*alpha). Higher alpha = LESS
// smoothing = the signal follows fast, short hops more closely so their
// amplitude isn't flattened away before the counter sees it.
// 0.35-0.55 is a moderate range: responsive enough for normal jumps,
// stable enough to avoid noise-triggered false detections.
const float kSmoothingAlphaMin = 0.35f;
const float kSmoothingAlphaMax = 0.55f;

// --- Amplitude thresholds (tuned for 2m distance, 3-5cm real-world jump) ---
// At 2m a person is ~200-300 px tall.  3-5 cm real jump ≈ 1.2-2.5% body height.
const float kMinAmplitudeRatio = 0.03f;    // ~3.0% body height (raised from 0.025)
const float kMinAmplitudePixels = 4.f;     // 4px absolute floor
const float kReturnBodyRatio = 0.06f;      // return margin = 6% of body height
const float kMaxJumpAmplitudeRatio = 0.20f; // max jump amplitude as fraction of body height
                                             // (tightened from 0.23) jumps: ~0.12-0.18, squats: ~0.22-0.40

// --- Velocity: how fast the body centre must move upward to trigger airborne.
// -0.25 requires a clear upward impulse — filters out slow postural sway
// while still catching normal-speed jumps.
const float kAirborneVelocityThreshold = -0.25f;

// --- Rhythm gating ---
// Lower bound dropped from 3 to 1: at 15-20 FPS a fast jump is only airborne
// for 2-4 frames, so a 3-frame minimum silently discarded many real jumps as
// "out of rhythm". 1 frame still rejects single-frame keypoint jitter via the
// amplitude/velocity checks while letting quick hops through.
const int kMinJumpCycleFrames = 1;         // allow very fast / low-FPS jumps
const int kMaxJumpCycleFrames = 30;        // ~1s upper bound (squat transitions often exceed this)

// --- Pose confidence: relaxed for 2m distance where keypoints are smaller ---
const float kPoseKeypointConfidence = 0.15f;
const float kMinFrameConfidence = 0.12f;
const float kTorsoToBodyScale = 2.8f;

// --- Ankle-lift gating (jump vs. knee-bend discrimination) ---
// A real jump lifts the feet off the floor; a squat/knee-bend keeps the
// ankles planted while the torso drops. We require the ankles to rise by a
// minimum fraction of body height before a cycle is allowed to count. This is
// only enforced when the ankles are actually visible during the airborne phase
// — if the feet leave the frame (common for tall jumpers) the counter falls
// back to torso-displacement logic so we never regress the previous behaviour.
const float kMinAnkleLiftRatio = 0.018f;   // feet must rise ~1.8% of body height
const float kAnkleKeypointConfidence = 0.15f; // match pose keypoint confidence

// --- Standing-still detector: recovers baseline drift after squats ---
// When the person is grounded and velocity is low for this many consecutive
// frames, we consider them "standing still" and snap the baseline to their
// current smoothed Y.  This erases drift accumulated during prior squats.
const int kStandingStillFrames = 15;           // ~0.5s at 30fps
const float kStandingStillVelocity = 0.4f;     // max |velocity| to count as "still"

bool is_valid_person_frame(const JumpRopeFrame& frame)
{
    return frame.has_person && frame.body_height > 1.f && frame.confidence >= kMinFrameConfidence;
}

float clamp_float(float value, float low, float high)
{
    return std::max(low, std::min(value, high));
}

struct AverageY
{
    bool has_value;
    float value;
    float confidence;
    int count;
};

AverageY average_reliable_y(const JumpRopePoseKeypoint* keypoints, int keypoint_count, const int* indices, int index_count)
{
    float total = 0.f;
    float confidence_total = 0.f;
    int count = 0;

    for (int i = 0; i < index_count; i++)
    {
        const int index = indices[i];
        if (index < 0 || index >= keypoint_count)
            continue;

        const JumpRopePoseKeypoint& keypoint = keypoints[index];
        if (keypoint.prob < kPoseKeypointConfidence)
            continue;

        total += keypoint.y;
        confidence_total += keypoint.prob;
        count++;
    }

    if (count == 0)
        return {false, 0.f, 0.f, 0};

    return {true, total / count, confidence_total / count, count};
}

} // namespace

JumpRopeFrame make_jumprope_frame_from_pose(const JumpRopePoseKeypoint* keypoints, int keypoint_count, float bbox_height)
{
    JumpRopeFrame invalid_frame = {false, 0.f, 0.f, 0.f};
    if (!keypoints || keypoint_count < 17)
        return invalid_frame;

    const int shoulder_indices[] = {5, 6};
    const int hip_indices[] = {11, 12};
    const int ankle_indices[] = {15, 16};

    const AverageY shoulders = average_reliable_y(keypoints, keypoint_count, shoulder_indices, 2);
    const AverageY hips = average_reliable_y(keypoints, keypoint_count, hip_indices, 2);
    const AverageY ankles = average_reliable_y(keypoints, keypoint_count, ankle_indices, 2);

    float center_y = 0.f;
    float confidence = 0.f;
    if (shoulders.has_value && hips.has_value)
    {
        // Pure torso centre: stable signal for jump detection.
        // Do NOT blend ankles — ankle keypoints are noisy and often
        // invisible during jumps (feet leave frame). The ankle-lift
        // gate in update() handles jump-vs-squat discrimination separately.
        center_y = shoulders.value * 0.35f + hips.value * 0.65f;
        confidence = (shoulders.confidence + hips.confidence) * 0.5f;
    }
    else if (hips.has_value)
    {
        center_y = hips.value;
        confidence = hips.confidence;
    }
    else if (shoulders.has_value && ankles.has_value)
    {
        center_y = shoulders.value * 0.50f + ankles.value * 0.50f;
        confidence = (shoulders.confidence + ankles.confidence) * 0.5f;
    }
    else if (shoulders.has_value && bbox_height > 1.f)
    {
        center_y = shoulders.value;
        confidence = shoulders.confidence * 0.80f;
    }
    else
    {
        return invalid_frame;
    }

    float pose_height = 0.f;
    if (shoulders.has_value && ankles.has_value && ankles.value > shoulders.value)
    {
        pose_height = ankles.value - shoulders.value;
    }
    else if (shoulders.has_value && hips.has_value)
    {
        const float torso_height = std::abs(hips.value - shoulders.value);
        if (torso_height > 10.f)
            pose_height = torso_height * kTorsoToBodyScale;
    }

    float body_height = bbox_height;
    if (pose_height > 1.f)
    {
        if (body_height <= 1.f)
        {
            body_height = pose_height;
        }
        else
        {
            const float min_reasonable_bbox = pose_height * 0.70f;
            const float max_reasonable_bbox = pose_height * 1.65f;
            if (body_height < min_reasonable_bbox || body_height > max_reasonable_bbox)
                body_height = pose_height;
            else
                body_height = body_height * 0.50f + pose_height * 0.50f;
        }
    }

    if (body_height <= 1.f)
        return invalid_frame;

    return {true, center_y, body_height, confidence};
}

// ---------------------------------------------------------------------------
// Constructor / Reset
// ---------------------------------------------------------------------------

JumpRopeCounter::JumpRopeCounter()
{
    reset();
}

void JumpRopeCounter::reset()
{
    state_ = JUMP_ROPE_STATE_INACTIVE;
    count_ = 0;
    missing_frames_ = 0;
    cooldown_frames_ = 0;
    has_smoothed_y_ = false;
    is_airborne_ = false;
    airborne_frames_ = 0;
    smoothed_y_ = 0.f;
    previous_smoothed_y_ = 0.f;
    baseline_y_ = 0.f;
    peak_y_ = 0.f;
    body_height_ = 0.f;

    // feature tracking
    has_last_features_ = false;
    cycle_just_completed_ = false;
    std::memset(&last_features_, 0, sizeof(last_features_));
    frame_kpts_ = nullptr;
    frame_kpt_count_ = 0;

    // ankle-lift gating
    has_ankle_ground_ = false;
    ankle_ground_y_ = 0.f;
    frame_ankle_y_ = 0.f;
    has_frame_ankle_ = false;
    airborne_ankle_seen_ = false;
    peak_ankle_lift_ = 0.f;
    airborne_first_motion_down_ = false;

    // standing-still detector
    standing_still_frames_ = 0;

    // baseline calibration
    baseline_calibrated_ = false;
    just_calibrated_ = false;
    valid_person_frames_ = 0;
    just_lost_ = false;

    // entry body height
    entry_body_height_ = 0.f;

    reset_cycle_tracking();
    reset_cooldown_tracking();
}

void JumpRopeCounter::reset_cycle_tracking()
{
    ft_cycle_frames_ = 0;
    ft_rise_frames_ = 0;
    ft_fall_frames_ = 0;
    ft_min_y_ = 1e9f;
    ft_max_y_ = -1e9f;
    ft_amplitude_px_ = 0.f;
    ft_peak_velocity_ = 0.f;
    ft_max_knee_flexion_ = 0.f;
    ft_max_ankle_elev_ = 0.f;
    ft_conf_sum_ = 0.f;
    ft_conf_count_ = 0;
    ft_lr_diff_sum_ = 0.f;
    ft_lr_count_ = 0;
    ft_body_height_avg_ = 0.f;
    ft_body_height_count_ = 0;
    ft_start_frame_ = {false, 0.f, 0.f, 0.f};
}

void JumpRopeCounter::reset_cooldown_tracking()
{
    cd_cycle_active_ = false;
    cd_cycle_frames_ = 0;
    cd_min_y_ = 1e9f;
    cd_peak_vel_ = 0.f;
    cd_rise_frames_ = 0;
    cd_fall_frames_ = 0;
}

// ---------------------------------------------------------------------------
// Feature accumulation helpers
// ---------------------------------------------------------------------------

void JumpRopeCounter::set_frame_keypoints(const JumpRopePoseKeypoint* kpts, int count)
{
    frame_kpts_ = kpts;
    frame_kpt_count_ = count;
}

void JumpRopeCounter::accumulate_stats(const JumpRopeFrame& frame,
                                       const JumpRopePoseKeypoint* kpts, int kpt_count)
{
    ft_cycle_frames_++;

    // Track min/max Y for amplitude
    if (frame.center_y < ft_min_y_)
        ft_min_y_ = frame.center_y;
    if (frame.center_y > ft_max_y_)
        ft_max_y_ = frame.center_y;

    // Peak velocity
    float vel = std::abs(smoothed_y_ - previous_smoothed_y_);
    if (vel > ft_peak_velocity_)
        ft_peak_velocity_ = vel;

    // Rise/fall frame counting
    if (is_airborne_)
        ft_rise_frames_++;
    else
        ft_fall_frames_++;

    // Body height running average
    if (frame.body_height > 1.f)
    {
        ft_body_height_avg_ += frame.body_height;
        ft_body_height_count_++;
    }

    // Confidence average
    if (frame.confidence > 0.f)
    {
        ft_conf_sum_ += frame.confidence;
        ft_conf_count_++;
    }

    // Keypoint-based features
    if (kpts && kpt_count >= 17)
    {
        // Knee flexion: distance from knee to hip (indices 13,14 vs 11,12)
        for (int side = 0; side < 2; side++)
        {
            int hip_idx = 11 + side;
            int knee_idx = 13 + side;
            int ankle_idx = 15 + side;

            if (kpts[hip_idx].prob > 0.2f && kpts[knee_idx].prob > 0.2f)
            {
                float knee_hip_dist = std::abs(kpts[knee_idx].y - kpts[hip_idx].y);
                if (knee_hip_dist > ft_max_knee_flexion_)
                    ft_max_knee_flexion_ = knee_hip_dist;
            }

            // Ankle elevation: how much the ankle moved up from its starting position
            if (ft_start_frame_.has_person && kpts[ankle_idx].prob > 0.2f)
            {
                // ft_start_frame_.center_y approximates the "ground" level
                // ankle elevation = ground_y - current_ankle_y  (positive = ankle is above ground)
                float elev = ft_start_frame_.center_y - kpts[ankle_idx].y;
                if (elev > ft_max_ankle_elev_)
                    ft_max_ankle_elev_ = elev;
            }
        }

        // Left-right symmetry: |left_y - right_y| for shoulders and hips
        if (kpts[5].prob > 0.2f && kpts[6].prob > 0.2f)
        {
            ft_lr_diff_sum_ += std::abs(kpts[5].y - kpts[6].y);
            ft_lr_count_++;
        }
        if (kpts[11].prob > 0.2f && kpts[12].prob > 0.2f)
        {
            ft_lr_diff_sum_ += std::abs(kpts[11].y - kpts[12].y);
            ft_lr_count_++;
        }
    }
}

JumpRopeFeatures JumpRopeCounter::compute_features() const
{
    JumpRopeFeatures f;
    std::memset(&f, 0, sizeof(f));

    float avg_body_height = (ft_body_height_count_ > 0)
        ? ft_body_height_avg_ / ft_body_height_count_
        : body_height_;
    if (avg_body_height <= 1.f)
        avg_body_height = 1.f;

    int duration = std::max(1, ft_cycle_frames_);
    float amplitude_px = ft_max_y_ - ft_min_y_;
    if (amplitude_px < 0.f)
        amplitude_px = 0.f;

    f.values[0]  = (float)duration;                                          // duration_frames
    f.values[1]  = amplitude_px / avg_body_height;                           // amplitude_ratio
    f.values[2]  = (float)ft_rise_frames_ / (float)duration;                 // rise_time_ratio
    f.values[3]  = (float)ft_fall_frames_ / (float)duration;                 // fall_time_ratio
    f.values[4]  = (ft_fall_frames_ > 0)
        ? (float)ft_rise_frames_ / (float)ft_fall_frames_ : 1.f;             // rise_fall_symmetry
    f.values[5]  = ft_max_knee_flexion_ / avg_body_height;                   // knee_flexion_ratio
    f.values[6]  = ft_max_ankle_elev_ / avg_body_height;                     // ankle_elevation_ratio
    f.values[7]  = (ft_conf_count_ > 0)
        ? ft_conf_sum_ / ft_conf_count_ : 0.f;                               // avg_confidence
    f.values[8]  = (ft_lr_count_ > 0)
        ? (ft_lr_diff_sum_ / ft_lr_count_) / avg_body_height : 0.f;         // left_right_symmetry
    f.values[9]  = ft_peak_velocity_ / avg_body_height;                      // peak_velocity_ratio
    f.values[10] = amplitude_px;                                             // amplitude_pixels
    f.values[11] = avg_body_height;                                          // body_height_pixels

    return f;
}

// ---------------------------------------------------------------------------
// Main update loop
// ---------------------------------------------------------------------------

JumpRopeResult JumpRopeCounter::update(const JumpRopeFrame& frame)
{
    // Clear per-update flags
    cycle_just_completed_ = false;
    has_last_features_ = false;
    just_calibrated_ = false;
    just_lost_ = false;

    if (!is_valid_person_frame(frame))
    {
        missing_frames_ = std::min(missing_frames_ + 1, kMissingGraceFrames + 1);

        if (missing_frames_ <= kMissingGraceFrames && state_ != JUMP_ROPE_STATE_INACTIVE && count_ > 0)
            return {state_, count_};

        state_ = JUMP_ROPE_STATE_NO_PERSON;
        has_smoothed_y_ = false;
        is_airborne_ = false;
        airborne_frames_ = 0;
        cooldown_frames_ = 0;
        reset_cycle_tracking();
        reset_cooldown_tracking();

        // Signal "person lost" to the UI and reset calibration so it
        // re-triggers when the person reappears.
        // Only fire if the person was previously detected (calibrated);
        // otherwise the app would spam "target lost" on startup before
        // anyone steps into the frame.
        if (baseline_calibrated_)
        {
            just_lost_ = true;
        }
        baseline_calibrated_ = false;
        valid_person_frames_ = 0;

        return {state_, count_};
    }

    missing_frames_ = 0;
    body_height_ = frame.body_height;

    // --- Calibration: fire after enough consecutive valid person frames ---
    // ~20 frames gives the EMA signal time to stabilise regardless of
    // keypoint jitter.  This is the primary calibration trigger; the
    // standing-still detector serves as a backup for post-squat recovery.
    valid_person_frames_++;
    const int kCalibrationFrames = 20;
    if (!baseline_calibrated_ && valid_person_frames_ >= kCalibrationFrames)
    {
        baseline_calibrated_ = true;
        just_calibrated_ = true;
        // Snap baseline to current smoothed position for a clean start.
        baseline_y_ = smoothed_y_;
    }

    // --- Per-frame ankle position (for jump-vs-knee-bend gating) ---
    // Mean Y of the two ankles when both are confident enough.
    has_frame_ankle_ = false;
    frame_ankle_y_ = 0.f;
    if (frame_kpts_ && frame_kpt_count_ >= 17)
    {
        const JumpRopePoseKeypoint& la = frame_kpts_[15];
        const JumpRopePoseKeypoint& ra = frame_kpts_[16];
        float sum = 0.f;
        int n = 0;
        if (la.prob >= kAnkleKeypointConfidence) { sum += la.y; n++; }
        if (ra.prob >= kAnkleKeypointConfidence) { sum += ra.y; n++; }
        if (n > 0)
        {
            frame_ankle_y_ = sum / n;
            has_frame_ankle_ = true;
        }
    }

    // Accumulate feature stats every valid frame
    accumulate_stats(frame, frame_kpts_, frame_kpt_count_);

    if (!has_smoothed_y_)
    {
        smoothed_y_ = frame.center_y;
        previous_smoothed_y_ = smoothed_y_;
        baseline_y_ = frame.center_y;
        peak_y_ = frame.center_y;
        has_smoothed_y_ = true;

        // Seed the ankle floor from the first reliable ankle reading.
        if (has_frame_ankle_)
        {
            ankle_ground_y_ = frame_ankle_y_;
            has_ankle_ground_ = true;
        }

        // Start a new cycle
        reset_cycle_tracking();
        reset_cooldown_tracking();
        ft_start_frame_ = frame;
        ft_min_y_ = frame.center_y;
        ft_max_y_ = frame.center_y;
        ft_cycle_frames_ = 0; // accumulate_stats at top already counted this frame
        state_ = count_ > 0 ? JUMP_ROPE_STATE_COUNTING : JUMP_ROPE_STATE_READY;
        return {state_, count_};
    }

    previous_smoothed_y_ = smoothed_y_;
    const float confidence_alpha = clamp_float(frame.confidence, 0.f, 1.f);
    const float smoothing_alpha = kSmoothingAlphaMin + (kSmoothingAlphaMax - kSmoothingAlphaMin) * confidence_alpha;
    smoothed_y_ = smoothed_y_ * (1.f - smoothing_alpha) + frame.center_y * smoothing_alpha;
    const float velocity_y = smoothed_y_ - previous_smoothed_y_;

    const float min_amplitude = std::max(kMinAmplitudePixels, body_height_ * kMinAmplitudeRatio);
    // Use max of current and entry body height for return margin to avoid
    // instability from per-frame body_height_ fluctuations during squats.
    const float stable_body_h = std::max(body_height_, entry_body_height_);
    const float return_margin = std::max(2.f, stable_body_h * kReturnBodyRatio);

    // --- Baseline: robust asymmetric torso EMA ---
    // FREEZE baseline during airborne — otherwise it drifts upward at 0.05/frame
    // (smoothed_y_ > baseline during jump oscillation), making the
    // return-to-baseline landing check impossible during continuous jumps.
    if (!is_airborne_)
    {
        if (smoothed_y_ > baseline_y_)
            baseline_y_ = baseline_y_ * 0.95f + smoothed_y_ * 0.05f;
        else
            baseline_y_ = baseline_y_ * 0.998f + smoothed_y_ * 0.002f;
    }

    // --- Standing-still detector ---
    // If the person is grounded (not airborne) and velocity is very low for
    // several consecutive frames, they are standing still.  Snap the baseline
    // to their current smoothed position — this erases any upward drift
    // accumulated during prior squats and restores sensitivity for the next
    // jump.  During continuous jumping the velocity is too high for this to
    // trigger, so it won't interfere with normal counting.
    if (!is_airborne_)
    {
        if (std::abs(velocity_y) < kStandingStillVelocity)
        {
            standing_still_frames_++;
            if (standing_still_frames_ >= kStandingStillFrames)
            {
                baseline_y_ = smoothed_y_;
                if (!baseline_calibrated_)
                {
                    baseline_calibrated_ = true;
                    just_calibrated_ = true;
                }
            }
        }
        else
        {
            standing_still_frames_ = 0;
        }
    }
    else
    {
        standing_still_frames_ = 0;
    }

    if (cooldown_frames_ > 0)
    {
        cooldown_frames_--;

        // --- Negative sample tracking during cooldown ---
        // Detect upward motions that WOULD have been counted if cooldown
        // were not active.  These become label=0 samples for the MLP.
        float abs_vel = std::abs(velocity_y);

        if (!cd_cycle_active_)
        {
            // Detect start of significant upward motion blocked by cooldown
            if (velocity_y < kAirborneVelocityThreshold && smoothed_y_ < baseline_y_ - min_amplitude * 0.6f)
            {
                cd_cycle_active_ = true;
                cd_cycle_frames_ = 1;
                cd_min_y_ = smoothed_y_;
                cd_peak_vel_ = abs_vel;
                cd_rise_frames_ = 1;
                cd_fall_frames_ = 0;
            }
        }
        else
        {
            cd_cycle_frames_++;
            if (smoothed_y_ < cd_min_y_)
                cd_min_y_ = smoothed_y_;
            if (abs_vel > cd_peak_vel_)
                cd_peak_vel_ = abs_vel;

            if (velocity_y < 0.f)
                cd_rise_frames_++;
            else
                cd_fall_frames_++;

            // Check if the cooldown cycle has returned to baseline
            float cd_amplitude = baseline_y_ - cd_min_y_;
            float cd_return_margin = std::max(2.f, cd_amplitude * 0.50f);

            if (cd_cycle_frames_ >= 4 &&
                cd_amplitude >= min_amplitude * 0.6f &&
                velocity_y > kAirborneVelocityThreshold &&
                smoothed_y_ >= baseline_y_ - cd_return_margin)
            {
                // Cooldown cycle completed — record as negative sample
                ft_cycle_frames_ = cd_cycle_frames_;
                ft_rise_frames_ = cd_rise_frames_;
                ft_fall_frames_ = cd_fall_frames_;
                ft_min_y_ = cd_min_y_;
                ft_max_y_ = baseline_y_;
                ft_amplitude_px_ = cd_amplitude;
                ft_peak_velocity_ = cd_peak_vel_;

                last_features_ = compute_features();
                has_last_features_ = true;
                cycle_just_completed_ = true;  // count NOT incremented → label=0

                reset_cooldown_tracking();
            }
        }
    }

    if (!is_airborne_)
    {
        // Airborne entry: use velocity signal FIRST, baseline deviation SECOND.
        // For small jumps the baseline tracks the body tightly, so baseline
        // deviation may never exceed min_amplitude.  Velocity is the real
        // indicator of a jump impulse.
        if (cooldown_frames_ == 0 && velocity_y < kAirborneVelocityThreshold &&
            smoothed_y_ <= baseline_y_ - std::max(1.f, min_amplitude * 0.60f))
        {
            is_airborne_ = true;
            airborne_frames_ = 1;
            peak_y_ = smoothed_y_;
            entry_body_height_ = body_height_;  // stable body height for ratio checks

            // The airborne entry condition requires velocity_y < threshold
            // (upward), so the FIRST motion of this cycle is upward — this
            // is a jump signature. A squat starts with downward motion
            // (center_y increases); if the baseline has drifted down during
            // the squat, the standing-up phase can then trigger a false
            // "airborne". We track whether the first significant motion
            // after entry is up or down to catch this case.
            airborne_first_motion_down_ = false;

            // Begin ankle-lift tracking for this airborne phase. The floor is
            // whatever the ankle ground level is right now; lift is measured
            // upward from it (positive = feet above the floor).
            airborne_ankle_seen_ = false;
            peak_ankle_lift_ = 0.f;
            if (has_frame_ankle_)
            {
                airborne_ankle_seen_ = true;
                if (has_ankle_ground_)
                {
                    const float lift = ankle_ground_y_ - frame_ankle_y_;
                    if (lift > peak_ankle_lift_)
                        peak_ankle_lift_ = lift;
                }
            }

            // IMPORTANT: start the jump-cycle window here.
            // The old code let ft_cycle_frames_ include all READY/standing
            // frames, so after waiting >45 frames the first real jump was
            // rejected as "out of rhythm" and never counted.
            reset_cycle_tracking();
            ft_start_frame_ = frame;
            ft_min_y_ = frame.center_y;
            ft_max_y_ = frame.center_y;
            ft_cycle_frames_ = 0; // accumulate_stats below will increment to 1
            accumulate_stats(frame, frame_kpts_, frame_kpt_count_);
        }
        else
        {
            // Ankle floor tracking (only while grounded). Track the MAXIMUM
            // (largest Y = lowest on screen) ankle position as the true floor
            // level. During squats the ankles stay planted so the floor stays
            // put. A brief noisy low-Y (high-on-screen) reading won't corrupt
            // the floor estimate. A very slow decay handles true floor changes.
            if (has_frame_ankle_)
            {
                if (!has_ankle_ground_)
                {
                    ankle_ground_y_ = frame_ankle_y_;
                    has_ankle_ground_ = true;
                }
                else if (frame_ankle_y_ > ankle_ground_y_)
                {
                    // Ankle is lower (larger Y) → adopt quickly: this is the true floor.
                    ankle_ground_y_ = ankle_ground_y_ * 0.5f + frame_ankle_y_ * 0.5f;
                }
                else
                {
                    // Ankle is higher (smaller Y) → creep up very slowly.
                    // 0.002 alpha ≈ ~17 min half-life; prevents noise from
                    // pulling the floor up during jumps or brief lifts.
                    ankle_ground_y_ = ankle_ground_y_ * 0.998f + frame_ankle_y_ * 0.002f;
                }

                // (ankle_torso_offset_ removed — baseline is now a pure
                // torso EMA, so no standing offset needs to be tracked.)
            }
        }
    }
    else
    {
        airborne_frames_++;
        if (smoothed_y_ < peak_y_)
            peak_y_ = smoothed_y_;

        // Track how far the feet have risen above the floor during this phase.
        if (has_frame_ankle_)
        {
            airborne_ankle_seen_ = true;
            if (has_ankle_ground_)
            {
                const float lift = ankle_ground_y_ - frame_ankle_y_;
                if (lift > peak_ankle_lift_)
                    peak_ankle_lift_ = lift;
            }
        }

        // First-phase direction tracking. A real jump starts with an upward
        // impulse (we already know that because the airborne entry requires
        // negative velocity). But a squat-then-stand can also enter airborne
        // during the "stand up" phase if the baseline drifted. The tell: in
        // the first few airborne frames of a squat-stand, the body is still
        // near the baseline (or even moving DOWN briefly before recovering
        // upward). For a real jump, the smoothed signal continues moving
        // upward away from baseline. Check: within the first 3 airborne
        // frames, if velocity turns positive (downward) before any
        // significant upward displacement was achieved, mark it as a
        // squat-first cycle.
        if (airborne_frames_ <= 3 && !airborne_first_motion_down_)
        {
            float displacement = baseline_y_ - smoothed_y_;
            if (velocity_y > 0.f && displacement < min_amplitude * 0.5f)
            {
                airborne_first_motion_down_ = true;
            }
        }

        const float airborne_amplitude = baseline_y_ - peak_y_;
        // Rhythm gating: reject cycles whose duration is outside the plausible
        // jump-rope window. This filters out long walking strides and tiny jitter.
        // Use airborne_frames_ for rhythm gating. ft_cycle_frames_ is for
        // feature extraction and can include extra frames depending on where
        // the cycle was reset; using it for gating caused valid jumps to be
        // discarded after the user stood still for a moment.
        const bool plausible_cycle = (airborne_frames_ >= kMinJumpCycleFrames &&
                                      airborne_frames_ <= kMaxJumpCycleFrames);

        // Early amplitude check: if amplitude already exceeds max jump ratio,
        // immediately discard. This prevents multi-jump phases from accumulating
        // huge amplitudes before the landing conditions are met.
        const float max_amp_threshold = stable_body_h * kMaxJumpAmplitudeRatio;
        if (airborne_amplitude > max_amp_threshold && airborne_frames_ >= kMinAirborneFrames)
        {
            is_airborne_ = false;
            airborne_frames_ = 0;
            baseline_y_ = baseline_y_ * 0.10f + smoothed_y_ * 0.90f;
            peak_y_ = smoothed_y_;
            reset_cycle_tracking();
            ft_start_frame_ = frame;
            ft_min_y_ = frame.center_y;
            ft_max_y_ = frame.center_y;
        }
        else if (airborne_frames_ >= kMinAirborneFrames &&
            airborne_amplitude >= min_amplitude &&
            velocity_y > kAirborneVelocityThreshold &&
            smoothed_y_ >= baseline_y_ - return_margin)
        {
            if (!plausible_cycle)
            {
                // Out-of-rhythm motion: discard without counting, reset cycle.
                is_airborne_ = false;
                airborne_frames_ = 0;
                baseline_y_ = baseline_y_ * 0.10f + smoothed_y_ * 0.90f;
                peak_y_ = smoothed_y_;
                reset_cycle_tracking();
                ft_start_frame_ = frame;
                ft_min_y_ = frame.center_y;
                ft_max_y_ = frame.center_y;
            }
            else if (airborne_amplitude > stable_body_h * kMaxJumpAmplitudeRatio)
            {
                // Amplitude too large for a real jump — squat-stand cycles
                // produce much larger amplitude relative to body height.
                // Aggressively recover baseline.
                is_airborne_ = false;
                airborne_frames_ = 0;
                baseline_y_ = baseline_y_ * 0.10f + smoothed_y_ * 0.90f;
                peak_y_ = smoothed_y_;
                reset_cycle_tracking();
                ft_start_frame_ = frame;
                ft_min_y_ = frame.center_y;
                ft_max_y_ = frame.center_y;
            }
            else if (airborne_ankle_seen_ && has_ankle_ground_ &&
                     peak_ankle_lift_ > 0.f &&
                     peak_ankle_lift_ < std::max(2.f, body_height_ * kMinAnkleLiftRatio))
            {
                // Ankles were visible but the feet never left the floor — this
                // is a knee-bend / squat, not a jump. Discard without counting.
                // (When the ankles are NOT visible we skip this gate and fall
                // through to the torso-displacement count below, so tall jumpers
                // whose feet leave the frame are unaffected.)
                is_airborne_ = false;
                airborne_frames_ = 0;
                baseline_y_ = baseline_y_ * 0.35f + smoothed_y_ * 0.65f;
                peak_y_ = smoothed_y_;
                reset_cycle_tracking();
                ft_start_frame_ = frame;
                ft_min_y_ = frame.center_y;
                ft_max_y_ = frame.center_y;
            }
            else
            {
                // ---- Cycle completed: compute features BEFORE resetting ----
                ft_amplitude_px_ = ft_max_y_ - ft_min_y_;
                if (ft_amplitude_px_ < 0.f) ft_amplitude_px_ = 0.f;
                last_features_ = compute_features();
                has_last_features_ = true;
                cycle_just_completed_ = true;

                count_++;
                cooldown_frames_ = kCountCooldownFrames;
                is_airborne_ = false;
                airborne_frames_ = 0;
                baseline_y_ = baseline_y_ * 0.35f + smoothed_y_ * 0.65f;
                peak_y_ = smoothed_y_;

                // Reset cycle tracking for next cycle
                reset_cycle_tracking();
                ft_start_frame_ = frame;
                ft_min_y_ = frame.center_y;
                ft_max_y_ = frame.center_y;
            }
        }
    }

    state_ = count_ > 0 ? JUMP_ROPE_STATE_COUNTING : JUMP_ROPE_STATE_READY;
    return {state_, count_};
}

int JumpRopeCounter::state() const
{
    return state_;
}

int JumpRopeCounter::count() const
{
    return count_;
}
