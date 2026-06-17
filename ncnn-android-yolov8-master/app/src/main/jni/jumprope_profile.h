#ifndef JUMPROPE_PROFILE_H
#define JUMPROPE_PROFILE_H

#include "jumprope_counter.h"
#include <vector>

// Maximum number of body proportion features
#define PROFILE_BODY_FEAT_DIM   5

// HSV histogram bins: 16 for Hue, 8 for Saturation, 4 for Value
// Finer bins improve discrimination between people wearing similar colours.
#define PROFILE_HIST_H_BINS     16
#define PROFILE_HIST_S_BINS     8
#define PROFILE_HIST_V_BINS     4
#define PROFILE_HIST_DIM        (PROFILE_HIST_H_BINS * PROFILE_HIST_S_BINS * PROFILE_HIST_V_BINS)

// Maximum number of stored profiles
#define MAX_PROFILES            4

// Similarity threshold: if best distance < this, consider it the same person.
// Lowered from 0.45 — body proportions alone are too similar between most
// people, so the old threshold let different people match as the same profile.
#define PROFILE_MATCH_THRESHOLD 0.25f

struct PersonProfile
{
    int   id;                              // 1-based ID
    float body_feat[PROFILE_BODY_FEAT_DIM]; // body proportion features
    float color_hist[PROFILE_HIST_DIM];    // normalised HSV histogram
    int   total_jumps;                     // accumulated jump count
    int   frames_seen;                     // how many frames contributed to features
    bool  active;                          // currently matched
};

// -----------------------------------------------------------------------
// Feature extraction helpers
// -----------------------------------------------------------------------

// Compute body proportion features from 17 COCO keypoints.
// Returns true if enough reliable keypoints were found.
bool compute_body_features(const JumpRopePoseKeypoint* kpts, int kpt_count,
                           float out[PROFILE_BODY_FEAT_DIM]);

// Compute normalised HSV histogram of the torso region.
// `rgb_data` is the raw RGB pixel buffer (row-major, 3 channels),
// `rgb_w` and `rgb_h` are the image dimensions.
// `kpts` are the 17 COCO keypoints in original image coordinates.
void compute_color_features(const unsigned char* rgb_data, int rgb_w, int rgb_h,
                            const JumpRopePoseKeypoint* kpts, int kpt_count,
                            float out[PROFILE_HIST_DIM]);

// -----------------------------------------------------------------------
// Profile manager
// -----------------------------------------------------------------------

class ProfileManager
{
public:
    ProfileManager();
    void reset();

    // Try to match the given features against stored profiles.
    // Returns the profile index [0, MAX_PROFILES) if matched, or -1 if new.
    int match(const float body_feat[PROFILE_BODY_FEAT_DIM],
              const float color_hist[PROFILE_HIST_DIM]);

    // Register a new profile (or update an existing one's features via EMA).
    // Returns the profile index.
    int update_or_create(int profile_idx,
                         const float body_feat[PROFILE_BODY_FEAT_DIM],
                         const float color_hist[PROFILE_HIST_DIM]);

    // Add jumps to a profile's total.
    void add_jumps(int profile_idx, int count);

    // Directly set a profile's total jump count.
    void set_total_jumps(int profile_idx, int total);

    // Getters
    int profile_count() const;
    const PersonProfile& profile(int idx) const;

    // Currently active profile (-1 if none)
    int active_profile() const { return active_idx_; }
    void set_active_profile(int idx) { active_idx_ = idx; }

private:
    PersonProfile profiles_[MAX_PROFILES];
    int num_profiles_;
    int active_idx_;
    int next_id_;

    float compute_distance(const float body1[PROFILE_BODY_FEAT_DIM],
                           const float hist1[PROFILE_HIST_DIM],
                           const float body2[PROFILE_BODY_FEAT_DIM],
                           const float hist2[PROFILE_HIST_DIM]) const;
};

#endif // JUMPROPE_PROFILE_H
