#ifndef JUMPROPE_CLASSIFIER_H
#define JUMPROPE_CLASSIFIER_H

#include "jumprope_feature.h"

// Small 3-layer MLP for jump validation.
// Architecture: 12 -> 16 (ReLU) -> 8 (ReLU) -> 1 (Sigmoid)
// Total params: 12*16 + 16 + 16*8 + 8 + 8*1 + 1 = 305
//
// When trained weights are not loaded, the classifier returns a
// neutral score of 0.5f so that all cycles pass through (the
// existing state-machine thresholds remain the sole gate).

class JumpRopeClassifier
{
public:
    JumpRopeClassifier();

    // Load trained weights from a binary file.
    // Layout (float32, little-endian):
    //   W1[12*16]  b1[16]  W2[16*8]  b2[8]  W3[8]  b3[1]
    bool load(const char* path);

    // Load weights from a memory buffer.
    bool load_from_memory(const float* data, int count);

    // Run forward pass.  Returns probability in [0, 1].
    // Returns 0.5f if weights are not loaded.
    float predict(const JumpRopeFeatures& features) const;

    bool is_loaded() const { return loaded_; }

    // Classification threshold (default 0.5).
    void set_threshold(float t) { threshold_ = t; }
    float threshold() const { return threshold_; }

private:
    static const int INPUT_DIM = JUMPROPE_FEATURE_DIM;  // 12
    static const int HIDDEN1   = 16;
    static const int HIDDEN2   = 8;
    static const int TOTAL_PARAMS = 353;

    float w1_[INPUT_DIM * HIDDEN1];
    float b1_[HIDDEN1];
    float w2_[HIDDEN1 * HIDDEN2];
    float b2_[HIDDEN2];
    float w3_[HIDDEN2];
    float b3_;

    bool  loaded_;
    float threshold_;

    void init_zeros();
    static void normalize_input(const float raw[JUMPROPE_FEATURE_DIM],
                                float out[JUMPROPE_FEATURE_DIM]);
};

#endif // JUMPROPE_CLASSIFIER_H
