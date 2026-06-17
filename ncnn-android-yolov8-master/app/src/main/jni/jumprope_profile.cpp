#include "jumprope_profile.h"

#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <cstring>

// -----------------------------------------------------------------------
// Body proportion features
// -----------------------------------------------------------------------

bool compute_body_features(const JumpRopePoseKeypoint* kpts, int kpt_count,
                           float out[PROFILE_BODY_FEAT_DIM])
{
    if (!kpts || kpt_count < 17)
        return false;

    // Required keypoints: shoulders (5,6), hips (11,12), ankles (15,16)
    const float conf_thresh = 0.12f;
    auto ok = [&](int idx) { return idx < kpt_count && kpts[idx].prob >= conf_thresh; };

    if (!ok(5) || !ok(6) || !ok(11) || !ok(12))
        return false;

    // Shoulder centre and hip centre
    float sh_cx = (kpts[5].x + kpts[6].x) * 0.5f;
    float sh_cy = (kpts[5].y + kpts[6].y) * 0.5f;
    float hp_cx = (kpts[11].x + kpts[12].x) * 0.5f;
    float hp_cy = (kpts[11].y + kpts[12].y) * 0.5f;

    float shoulder_w = std::sqrt(std::pow(kpts[6].x - kpts[5].x, 2) +
                                  std::pow(kpts[6].y - kpts[5].y, 2));
    float hip_w = std::sqrt(std::pow(kpts[12].x - kpts[11].x, 2) +
                             std::pow(kpts[12].y - kpts[11].y, 2));
    float torso_h = std::sqrt(std::pow(hp_cx - sh_cx, 2) +
                               std::pow(hp_cy - sh_cy, 2));

    if (torso_h < 10.f)
        return false;

    // Feature 0: shoulder_width / torso_height
    out[0] = shoulder_w / torso_h;

    // Feature 1: hip_width / shoulder_width
    out[1] = (shoulder_w > 5.f) ? hip_w / shoulder_w : 1.f;

    // Feature 2: torso_height / estimated_body_height
    float body_h = torso_h * 2.8f;  // approximate
    if (ok(15) && ok(16))
    {
        float ankle_cy = (kpts[15].y + kpts[16].y) * 0.5f;
        float real_h = std::abs(ankle_cy - sh_cy);
        if (real_h > torso_h)
            body_h = real_h;
    }
    out[2] = torso_h / body_h;

    // Feature 3: hip_width / torso_height
    out[3] = hip_w / torso_h;

    // Feature 4: shoulder_width / body_height
    out[4] = shoulder_w / body_h;

    return true;
}

// -----------------------------------------------------------------------
// Color histogram features (HSV of torso region)
// -----------------------------------------------------------------------

static void rgb_to_hsv(unsigned char r, unsigned char g, unsigned char b,
                       float& h, float& s, float& v)
{
    float rf = r / 255.f, gf = g / 255.f, bf = b / 255.f;
    float mx = std::max({rf, gf, bf});
    float mn = std::min({rf, gf, bf});
    float d = mx - mn;

    v = mx;
    s = (mx > 0.f) ? d / mx : 0.f;

    if (d < 1e-6f)
        h = 0.f;
    else if (mx == rf)
        h = 60.f * std::fmod((gf - bf) / d, 6.f);
    else if (mx == gf)
        h = 60.f * ((bf - rf) / d + 2.f);
    else
        h = 60.f * ((rf - gf) / d + 4.f);

    if (h < 0.f) h += 360.f;
}

void compute_color_features(const unsigned char* rgb_data, int rgb_w, int rgb_h,
                            const JumpRopePoseKeypoint* kpts, int kpt_count,
                            float out[PROFILE_HIST_DIM])
{
    std::memset(out, 0, sizeof(float) * PROFILE_HIST_DIM);

    if (!kpts || kpt_count < 17 || !rgb_data || rgb_w <= 0 || rgb_h <= 0)
        return;

    const float conf_thresh = 0.12f;
    auto ok = [&](int idx) { return idx < kpt_count && kpts[idx].prob >= conf_thresh; };
    if (!ok(5) || !ok(6) || !ok(11) || !ok(12))
        return;

    // Torso bounding box from keypoints (with some padding)
    float min_x = std::min(kpts[5].x, std::min(kpts[6].x, std::min(kpts[11].x, kpts[12].x)));
    float max_x = std::max(kpts[5].x, std::max(kpts[6].x, std::max(kpts[11].x, kpts[12].x)));
    float min_y = std::min(kpts[5].y, std::min(kpts[6].y, std::min(kpts[11].y, kpts[12].y)));
    float max_y = std::max(kpts[5].y, std::max(kpts[6].y, std::max(kpts[11].y, kpts[12].y)));

    // Add 15% padding to capture more of the shirt area
    float pad_x = (max_x - min_x) * 0.15f;
    float pad_y = (max_y - min_y) * 0.15f;
    int x0 = std::max(0, (int)(min_x - pad_x));
    int y0 = std::max(0, (int)(min_y - pad_y));
    int x1 = std::min(rgb_w, (int)(max_x + pad_x));
    int y1 = std::min(rgb_h, (int)(max_y + pad_y));

    if (x1 - x0 < 4 || y1 - y0 < 4)
        return;

    float h_step = 360.f / PROFILE_HIST_H_BINS;
    float s_step = 1.0f / PROFILE_HIST_S_BINS;
    float v_step = 1.0f / PROFILE_HIST_V_BINS;
    int total_pixels = 0;

    for (int y = y0; y < y1; y++)
    {
        for (int x = x0; x < x1; x++)
        {
            int idx = (y * rgb_w + x) * 3;
            float h, s, v;
            rgb_to_hsv(rgb_data[idx], rgb_data[idx + 1], rgb_data[idx + 2], h, s, v);

            // Skip very dark or desaturated pixels (skin/background)
            if (v < 0.12f || s < 0.06f)
                continue;

            int h_bin = std::min((int)(h / h_step), PROFILE_HIST_H_BINS - 1);
            int s_bin = std::min((int)(s / s_step), PROFILE_HIST_S_BINS - 1);
            int v_bin = std::min((int)(v / v_step), PROFILE_HIST_V_BINS - 1);
            out[(h_bin * PROFILE_HIST_S_BINS + s_bin) * PROFILE_HIST_V_BINS + v_bin] += 1.f;
            total_pixels++;
        }
    }

    // Normalise
    if (total_pixels > 0)
    {
        float inv = 1.f / total_pixels;
        for (int i = 0; i < PROFILE_HIST_DIM; i++)
            out[i] *= inv;
    }
}

// -----------------------------------------------------------------------
// ProfileManager
// -----------------------------------------------------------------------

ProfileManager::ProfileManager()
{
    reset();
}

void ProfileManager::reset()
{
    std::memset(profiles_, 0, sizeof(profiles_));
    num_profiles_ = 0;
    active_idx_ = -1;
    next_id_ = 1;
}

float ProfileManager::compute_distance(
    const float body1[PROFILE_BODY_FEAT_DIM],
    const float hist1[PROFILE_HIST_DIM],
    const float body2[PROFILE_BODY_FEAT_DIM],
    const float hist2[PROFILE_HIST_DIM]) const
{
    // Body proportion distance (L2 normalised by dimension)
    float body_dist = 0.f;
    for (int i = 0; i < PROFILE_BODY_FEAT_DIM; i++)
    {
        float d = body1[i] - body2[i];
        body_dist += d * d;
    }
    body_dist = std::sqrt(body_dist / PROFILE_BODY_FEAT_DIM);

    // Histogram distance (chi-squared)
    float hist_dist = 0.f;
    for (int i = 0; i < PROFILE_HIST_DIM; i++)
    {
        float sum = hist1[i] + hist2[i];
        if (sum > 1e-6f)
        {
            float d = hist1[i] - hist2[i];
            hist_dist += (d * d) / sum;
        }
    }
    hist_dist *= 0.5f;

    // Weighted combination: 30% body, 70% colour.
    // Body proportions (shoulder/hip/torso ratios) are very similar between
    // most people and provide little discrimination.  The colour histogram
    // of the torso region (shirt colour, pattern) is far more distinctive,
    // so it receives the majority weight.
    return body_dist * 0.3f + hist_dist * 0.7f;
}

int ProfileManager::match(const float body_feat[PROFILE_BODY_FEAT_DIM],
                          const float color_hist[PROFILE_HIST_DIM])
{
    if (num_profiles_ == 0)
        return -1;

    int best_idx = -1;
    float best_dist = PROFILE_MATCH_THRESHOLD;

    for (int i = 0; i < num_profiles_; i++)
    {
        float dist = compute_distance(body_feat, color_hist,
                                      profiles_[i].body_feat,
                                      profiles_[i].color_hist);
        __android_log_print(ANDROID_LOG_DEBUG, "ProfileManager",
                            "  vs P%d: dist=%.4f (threshold=%.4f) %s",
                            profiles_[i].id, dist, PROFILE_MATCH_THRESHOLD,
                            dist < best_dist ? "<- best" : "");
        if (dist < best_dist)
        {
            best_dist = dist;
            best_idx = i;
        }
    }

    if (best_idx >= 0)
        __android_log_print(ANDROID_LOG_INFO, "ProfileManager",
                            "Matched P%d (dist=%.4f)", profiles_[best_idx].id, best_dist);
    else
        __android_log_print(ANDROID_LOG_INFO, "ProfileManager",
                            "No match (best_dist=%.4f >= threshold=%.4f) -> new profile",
                            best_dist, PROFILE_MATCH_THRESHOLD);

    return best_idx;
}

int ProfileManager::update_or_create(int profile_idx,
                                     const float body_feat[PROFILE_BODY_FEAT_DIM],
                                     const float color_hist[PROFILE_HIST_DIM])
{
    if (profile_idx >= 0 && profile_idx < num_profiles_)
    {
        // EMA update of existing profile features.
        // Very slow adaptation (alpha=0.05) to prevent feature drift —
        // a matched profile should stay anchored to its original appearance
        // so that a different person arriving later is not absorbed.
        const float alpha = 0.05f;
        PersonProfile& p = profiles_[profile_idx];
        for (int i = 0; i < PROFILE_BODY_FEAT_DIM; i++)
            p.body_feat[i] = p.body_feat[i] * (1.f - alpha) + body_feat[i] * alpha;
        for (int i = 0; i < PROFILE_HIST_DIM; i++)
            p.color_hist[i] = p.color_hist[i] * (1.f - alpha) + color_hist[i] * alpha;
        p.frames_seen++;
        return profile_idx;
    }

    // Create new profile
    if (num_profiles_ >= MAX_PROFILES)
    {
        // Replace the least-seen profile
        int min_idx = 0;
        for (int i = 1; i < num_profiles_; i++)
        {
            if (profiles_[i].frames_seen < profiles_[min_idx].frames_seen)
                min_idx = i;
        }
        profile_idx = min_idx;
    }
    else
    {
        profile_idx = num_profiles_;
        num_profiles_++;
    }

    PersonProfile& p = profiles_[profile_idx];
    p.id = next_id_++;
    std::memcpy(p.body_feat, body_feat, sizeof(float) * PROFILE_BODY_FEAT_DIM);
    std::memcpy(p.color_hist, color_hist, sizeof(float) * PROFILE_HIST_DIM);
    p.total_jumps = 0;
    p.frames_seen = 1;
    p.active = true;

    return profile_idx;
}

void ProfileManager::add_jumps(int profile_idx, int count)
{
    if (profile_idx >= 0 && profile_idx < num_profiles_)
        profiles_[profile_idx].total_jumps += count;
}

void ProfileManager::set_total_jumps(int profile_idx, int total)
{
    if (profile_idx >= 0 && profile_idx < num_profiles_)
        profiles_[profile_idx].total_jumps = total;
}

int ProfileManager::profile_count() const
{
    return num_profiles_;
}

const PersonProfile& ProfileManager::profile(int idx) const
{
    return profiles_[idx];
}
