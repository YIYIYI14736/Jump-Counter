#include "jumprope_feature.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

const char* jumprope_features_csv_header()
{
    return "duration_frames,amplitude_ratio,rise_time_ratio,fall_time_ratio,"
           "rise_fall_symmetry,knee_flexion_ratio,ankle_elevation_ratio,"
           "avg_confidence,left_right_symmetry,peak_velocity_ratio,"
           "amplitude_pixels,body_height_pixels";
}

char* jumprope_features_to_csv(const JumpRopeFeatures& f, char* buf, int buf_size)
{
    std::snprintf(buf, buf_size,
        "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f",
        (int)f.values[0],
        f.values[1], f.values[2], f.values[3], f.values[4],
        f.values[5], f.values[6], f.values[7], f.values[8],
        f.values[9], f.values[10], f.values[11]);
    return buf;
}

bool jumprope_features_from_csv(const char* line, JumpRopeFeatures& f)
{
    if (!line || !*line)
        return false;

    char buf[512];
    std::strncpy(buf, line, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char* token = std::strtok(buf, ",");
    for (int i = 0; i < JUMPROPE_FEATURE_DIM && token; i++)
    {
        f.values[i] = (float)std::atof(token);
        token = std::strtok(nullptr, ",");
    }

    return token == nullptr; // consumed exactly FEATURE_DIM values
}
