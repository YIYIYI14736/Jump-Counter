#ifndef JUMPROPE_COUNTER_H
#define JUMPROPE_COUNTER_H

#include "jumprope_feature.h"

enum JumpRopeState
{
    JUMP_ROPE_STATE_INACTIVE = 0,
    JUMP_ROPE_STATE_NO_PERSON = 1,
    JUMP_ROPE_STATE_READY = 2,
    JUMP_ROPE_STATE_COUNTING = 3
};

struct JumpRopeFrame
{
    bool has_person;
    float center_y;
    float body_height;
    float confidence;
};

struct JumpRopePoseKeypoint
{
    float x;
    float y;
    float prob;
};

struct JumpRopeResult
{
    int state;
    int count;
};

JumpRopeFrame make_jumprope_frame_from_pose(const JumpRopePoseKeypoint* keypoints, int keypoint_count, float bbox_height);

class JumpRopeCounter
{
public:
    JumpRopeCounter();

    void reset();
    JumpRopeResult update(const JumpRopeFrame& frame);

    int state() const;
    int count() const;

    // Feature extraction (pseudo-label pipeline)
    bool has_last_features() const { return has_last_features_; }
    JumpRopeFeatures last_features() const { return last_features_; }
    void clear_last_features() { has_last_features_ = false; }

    // True when the last update() completed a cycle (count may or may not
    // have incremented — the caller decides the label).
    bool cycle_just_completed() const { return cycle_just_completed_; }

private:
    int state_;
    int count_;
    int missing_frames_;
    int cooldown_frames_;
    bool has_smoothed_y_;
    bool is_airborne_;
    int airborne_frames_;
    float smoothed_y_;
    float previous_smoothed_y_;
    float baseline_y_;
    float peak_y_;
    float body_height_;

    // --- feature tracking state ---
    bool  has_last_features_;
    bool  cycle_just_completed_;
    JumpRopeFeatures last_features_;

    int   ft_cycle_frames_;
    int   ft_rise_frames_;
    int   ft_fall_frames_;
    float ft_min_y_;
    float ft_max_y_;
    float ft_amplitude_px_;
    float ft_peak_velocity_;
    float ft_max_knee_flexion_;
    float ft_max_ankle_elev_;
    float ft_conf_sum_;
    int   ft_conf_count_;
    float ft_lr_diff_sum_;
    int   ft_lr_count_;
    float ft_body_height_avg_;
    int   ft_body_height_count_;
    JumpRopeFrame ft_start_frame_;

    // --- cooldown negative sample tracking ---
    bool  cd_cycle_active_;
    int   cd_cycle_frames_;
    float cd_min_y_;
    float cd_peak_vel_;
    int   cd_rise_frames_;
    int   cd_fall_frames_;

    // --- ankle-lift gating (distinguishes a real jump from a knee-bend) ---
    // A jump means the FEET leave the floor; a squat/knee-bend moves the torso
    // a lot while the ankles stay put. We track a stable "floor" level for the
    // ankles and require them to actually rise before counting. When the ankles
    // are not visible (e.g. a tall person whose feet leave the frame) the gate
    // is skipped and the counter falls back to torso-displacement logic.
    bool  has_ankle_ground_;
    float ankle_ground_y_;        // tracked floor level of the ankles (image y)
    float frame_ankle_y_;         // current frame mean ankle y
    bool  has_frame_ankle_;       // ankle keypoints reliable this frame
    bool  airborne_ankle_seen_;   // ankles were visible during this airborne phase
    float peak_ankle_lift_;       // max (ground_y - ankle_y) during airborne phase
    bool  airborne_first_motion_down_; // true if the first significant motion was downward (squat)

    // --- standing-still detector (baseline drift recovery) ---
    // When the person is grounded with low velocity for several consecutive
    // frames, they are standing still.  We snap the baseline to their current
    // smoothed_y_ to erase any drift accumulated during prior squats.
    int   standing_still_frames_;

    // --- stable body height at airborne entry ---
    // Captured at airborne entry and used for ratio-based checks (AMP_RATIO,
    // return margin) to avoid instability from per-frame body_height_ which
    // can be wildly wrong during squats.
    float entry_body_height_;

    // --- baseline calibration flag ---
    // Set to true after the counter has received enough consecutive valid
    // person frames for the EMA signal to stabilise.
    // Used by the UI to signal "ready to jump".
    bool baseline_calibrated_;
    bool just_calibrated_;
    int  valid_person_frames_;

    // --- person lost flag ---
    // Set to true on the frame the person is declared missing (after grace
    // period).  Cleared at the start of every update() call.
    // The UI uses this to play a "target lost" prompt and reset calibration.
    bool just_lost_;

    void reset_cycle_tracking();
    void reset_cooldown_tracking();
    void accumulate_stats(const JumpRopeFrame& frame,
                          const JumpRopePoseKeypoint* kpts, int kpt_count);
    JumpRopeFeatures compute_features() const;

public:
    // Optional: supply raw keypoints each frame for richer feature extraction.
    // If not called, knee/ankle/symmetry features default to 0.
    void set_frame_keypoints(const JumpRopePoseKeypoint* kpts, int count);

    // True once the standing-still detector has snapped the baseline for the
    // first time after reset().  Indicates the system is calibrated and ready
    // for the user to start jumping.
    bool baseline_calibrated() const { return baseline_calibrated_; }

    // True only on the exact frame the baseline was first calibrated.
    // Cleared at the start of every update() call.
    bool just_calibrated() const { return just_calibrated_; }

    // True only on the exact frame the person was declared lost.
    // Cleared at the start of every update() call.
    bool just_lost() const { return just_lost_; }

private:
    const JumpRopePoseKeypoint* frame_kpts_;
    int frame_kpt_count_;
};

#endif // JUMPROPE_COUNTER_H
