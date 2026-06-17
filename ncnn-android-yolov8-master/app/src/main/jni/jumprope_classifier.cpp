#include "jumprope_classifier.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

JumpRopeClassifier::JumpRopeClassifier()
    : loaded_(false), threshold_(0.5f)
{
    init_zeros();
}

void JumpRopeClassifier::init_zeros()
{
    std::memset(w1_, 0, sizeof(w1_));
    std::memset(b1_, 0, sizeof(b1_));
    std::memset(w2_, 0, sizeof(w2_));
    std::memset(b2_, 0, sizeof(b2_));
    std::memset(w3_, 0, sizeof(w3_));
    b3_ = 0.f;
}

bool JumpRopeClassifier::load(const char* path)
{
    FILE* fp = std::fopen(path, "rb");
    if (!fp)
        return false;

    float buf[TOTAL_PARAMS];
    size_t read_count = std::fread(buf, sizeof(float), TOTAL_PARAMS, fp);
    std::fclose(fp);

    if (read_count != TOTAL_PARAMS)
        return false;

    return load_from_memory(buf, TOTAL_PARAMS);
}

bool JumpRopeClassifier::load_from_memory(const float* data, int count)
{
    if (count != TOTAL_PARAMS)
        return false;

    int offset = 0;
    std::memcpy(w1_, data + offset, sizeof(w1_)); offset += INPUT_DIM * HIDDEN1;
    std::memcpy(b1_, data + offset, sizeof(b1_)); offset += HIDDEN1;
    std::memcpy(w2_, data + offset, sizeof(w2_)); offset += HIDDEN1 * HIDDEN2;
    std::memcpy(b2_, data + offset, sizeof(b2_)); offset += HIDDEN2;
    std::memcpy(w3_, data + offset, sizeof(w3_)); offset += HIDDEN2;
    b3_ = data[offset];

    loaded_ = true;
    return true;
}

// ---------------------------------------------------------------------------
// Input normalization:  maps raw feature ranges to roughly [0, 1].
// This MUST match the normalization used in the Python training script
// (JumpRope-Model/training/jumprope_classifier/config.py NORMALIZE_STATS).
// ---------------------------------------------------------------------------
void JumpRopeClassifier::normalize_input(const float raw[JUMPROPE_FEATURE_DIM],
                                         float out[JUMPROPE_FEATURE_DIM])
{
    // Each entry: {mean, std}  —  out[i] = (raw[i] - mean) / std
    // These statistics are computed from a representative jump-rope session
    // and must be kept in sync with training.
    static const float stats[JUMPROPE_FEATURE_DIM][2] = {
        { 22.0f,  10.0f},   //  0: duration_frames
        {  0.08f,  0.04f},  //  1: amplitude_ratio
        {  0.40f,  0.12f},  //  2: rise_time_ratio
        {  0.60f,  0.12f},  //  3: fall_time_ratio
        {  0.80f,  0.40f},  //  4: rise_fall_symmetry
        {  0.10f,  0.06f},  //  5: knee_flexion_ratio
        {  0.05f,  0.04f},  //  6: ankle_elevation_ratio
        {  0.50f,  0.20f},  //  7: avg_confidence
        {  0.03f,  0.02f},  //  8: left_right_symmetry
        {  0.04f,  0.02f},  //  9: peak_velocity_ratio
        { 30.0f,  20.0f},   // 10: amplitude_pixels
        {350.0f, 150.0f},   // 11: body_height_pixels
    };

    for (int i = 0; i < JUMPROPE_FEATURE_DIM; i++)
    {
        out[i] = (raw[i] - stats[i][0]) / stats[i][1];
    }
}

float JumpRopeClassifier::predict(const JumpRopeFeatures& features) const
{
    if (!loaded_)
        return 0.5f;

    float x[JUMPROPE_FEATURE_DIM];
    normalize_input(features.values, x);

    // Layer 1: Linear(12, 16) + ReLU
    float h1[HIDDEN1];
    for (int i = 0; i < HIDDEN1; i++)
    {
        float sum = b1_[i];
        for (int j = 0; j < INPUT_DIM; j++)
            sum += w1_[j * HIDDEN1 + i] * x[j];
        h1[i] = std::max(0.f, sum);
    }

    // Layer 2: Linear(16, 8) + ReLU
    float h2[HIDDEN2];
    for (int i = 0; i < HIDDEN2; i++)
    {
        float sum = b2_[i];
        for (int j = 0; j < HIDDEN1; j++)
            sum += w2_[j * HIDDEN2 + i] * h1[j];
        h2[i] = std::max(0.f, sum);
    }

    // Layer 3: Linear(8, 1) + Sigmoid
    float logit = b3_;
    for (int j = 0; j < HIDDEN2; j++)
        logit += w3_[j] * h2[j];

    return 1.f / (1.f + std::exp(-logit));
}
