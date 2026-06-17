#ifndef JUMPROPE_FEATURE_H
#define JUMPROPE_FEATURE_H

#define JUMPROPE_FEATURE_DIM 12

struct JumpRopeFeatures
{
    float values[JUMPROPE_FEATURE_DIM];
    // Feature index mapping:
    //  0: duration_frames        Total frames in the cycle
    //  1: amplitude_ratio        Vertical amplitude / body_height
    //  2: rise_time_ratio        rise_frames / duration_frames
    //  3: fall_time_ratio        fall_frames / duration_frames
    //  4: rise_fall_symmetry     rise_frames / max(fall_frames, 1)
    //  5: knee_flexion_ratio     Max knee flexion change / body_height
    //  6: ankle_elevation_ratio  Max ankle elevation / body_height
    //  7: avg_confidence         Mean keypoint confidence during cycle
    //  8: left_right_symmetry    Mean |left_y - right_y| / body_height
    //  9: peak_velocity_ratio    Peak |velocity| / body_height
    // 10: amplitude_pixels       Raw amplitude in pixels
    // 11: body_height_pixels     Body height used for normalization
};

// CSV header for recording
const char* jumprope_features_csv_header();

// Format features as a CSV row (writes into buf, returns buf)
char* jumprope_features_to_csv(const JumpRopeFeatures& f, char* buf, int buf_size);

// Parse features from a CSV row
bool jumprope_features_from_csv(const char* line, JumpRopeFeatures& f);

#endif // JUMPROPE_FEATURE_H
