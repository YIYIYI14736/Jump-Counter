// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "yolov8.h"
#include "jumprope_counter.h"
#include "jumprope_classifier.h"
#include "jumprope_profile.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdio>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat& rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

static YOLOv8* g_yolov8 = 0;
static ncnn::Mutex lock;

static JumpRopeCounter g_jumprope_counter;
static JumpRopeClassifier g_jumprope_classifier;
static ProfileManager g_profile_manager;
static std::atomic<int> g_jumprope_state(JUMP_ROPE_STATE_INACTIVE);
static std::atomic<int> g_jumprope_count(0);
// Raw count (before MLP gating) so the UI can show both counters.
static std::atomic<int> g_jumprope_raw_count(0);
// Cumulative number of cycles the MLP has rejected. The gated (mlp) count is
// raw_count - rejected. Using a persistent offset is required because the
// counter's internal count_ keeps the increment; vetoing only the completion
// frame would let the gated count snap back up on the next frame.
static std::atomic<int> g_jumprope_rejected(0);
// Set to true the first time the counter's baseline is calibrated after
// reset.  The Java layer reads and clears this via isBaselineCalibrated().
static std::atomic<bool> g_jumprope_calibrated(false);
// Set to true on the frame the person is declared lost.  The Java layer
// reads and clears this via isPersonLost().
static std::atomic<bool> g_jumprope_lost(false);
// Offset added to the counter's count for display purposes (accumulated from
// previous sessions of the currently matched profile).
static std::atomic<int> g_prev_session_total(0);
int g_current_taskid = 0;  // extern — read by yolov8_pose / yolo11_pose draw()
static bool g_has_primary_pose_person = false;
static cv::Rect_<float> g_primary_pose_rect;
static int g_primary_pose_missing_frames = 0;

// Profile color palette (BGR): cyan, green, orange, magenta
static const cv::Scalar kProfileColors[MAX_PROFILES] = {
    cv::Scalar(0, 255, 255),    // P1: cyan/yellow
    cv::Scalar(0, 255, 0),      // P2: green
    cv::Scalar(0, 165, 255),    // P3: orange
    cv::Scalar(255, 0, 255)     // P4: magenta
};

// Track detected person's bbox for colored box drawing
static cv::Rect g_person_bbox;
static bool g_person_visible = false;

// --- Feature recording (pseudo-label data collection) ---
static FILE* g_recording_file = nullptr;
static bool  g_recording_active = false;
static int   g_recording_sample_count = 0;

const int kPrimaryPoseGraceFrames = 12;
const float kPrimaryPoseMinAffinity = 0.18f;

static void reset_primary_pose_tracking()
{
    g_has_primary_pose_person = false;
    g_primary_pose_rect = cv::Rect_<float>();
    g_primary_pose_missing_frames = 0;
}

static void reset_jumprope_tracking()
{
    g_jumprope_counter.reset();
    g_jumprope_state.store(JUMP_ROPE_STATE_INACTIVE, std::memory_order_relaxed);
    g_jumprope_count.store(0, std::memory_order_relaxed);
    g_jumprope_raw_count.store(0, std::memory_order_relaxed);
    g_jumprope_rejected.store(0, std::memory_order_relaxed);
    g_jumprope_calibrated.store(false, std::memory_order_relaxed);
    g_jumprope_lost.store(false, std::memory_order_relaxed);
    g_prev_session_total.store(0, std::memory_order_relaxed);
    g_profile_manager.reset();
    g_person_bbox = cv::Rect();
    g_person_visible = false;
    reset_primary_pose_tracking();
}

static float rect_iou(const cv::Rect_<float>& a, const cv::Rect_<float>& b)
{
    const float inter_area = (a & b).area();
    const float union_area = a.area() + b.area() - inter_area;
    if (union_area <= 0.f)
        return 0.f;

    return inter_area / union_area;
}

static float rect_center_affinity(const cv::Rect_<float>& a, const cv::Rect_<float>& b)
{
    const float ax = a.x + a.width * 0.5f;
    const float ay = a.y + a.height * 0.5f;
    const float bx = b.x + b.width * 0.5f;
    const float by = b.y + b.height * 0.5f;
    const float dx = ax - bx;
    const float dy = ay - by;
    const float distance = std::sqrt(dx * dx + dy * dy);
    const float scale = std::max(1.f, std::sqrt(std::max(a.area(), b.area())));

    return std::max(0.f, 1.f - distance / scale);
}

static float pose_core_quality(const Object& obj)
{
    if (obj.keypoints.size() < 17)
        return 0.f;

    const int core_indices[] = {5, 6, 11, 12, 13, 14, 15, 16};
    float quality = 0.f;
    int used = 0;

    for (size_t i = 0; i < sizeof(core_indices) / sizeof(core_indices[0]); i++)
    {
        const float prob = obj.keypoints[core_indices[i]].prob;
        if (prob <= 0.10f)
            continue;

        quality += prob;
        used++;
    }

    if (used == 0)
        return 0.f;

    return quality / used;
}

static float pose_person_score(const Object& obj)
{
    const float area = std::max(obj.rect.area(), 1.f);
    const float quality = pose_core_quality(obj);
    return area * std::max(obj.prob, 0.01f) * (0.30f + quality);
}

static float pose_target_affinity(const Object& obj)
{
    if (!g_has_primary_pose_person)
        return 1.f;

    const float iou = rect_iou(g_primary_pose_rect, obj.rect);
    const float center_affinity = rect_center_affinity(g_primary_pose_rect, obj.rect);
    return std::max(iou, center_affinity);
}

static const Object* select_best_pose_person(const std::vector<Object>& objects)
{
    const Object* best = 0;
    float best_score = 0.f;
    bool best_matches_target = false;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        if (obj.keypoints.size() < 17)
            continue;

        const float quality = pose_core_quality(obj);
        if (quality < 0.08f)
            continue;

        const float affinity = pose_target_affinity(obj);
        const bool matches_target = !g_has_primary_pose_person || affinity >= kPrimaryPoseMinAffinity;
        if (g_has_primary_pose_person && g_primary_pose_missing_frames < kPrimaryPoseGraceFrames && !matches_target)
            continue;

        const float tracking_weight = g_has_primary_pose_person ? (0.35f + affinity) : 1.f;
        const float score = pose_person_score(obj) * tracking_weight;
        if (!best || score > best_score || (matches_target && !best_matches_target))
        {
            best = &obj;
            best_score = score;
            best_matches_target = matches_target;
        }
    }

    return best;
}

static void keep_primary_pose_person(std::vector<Object>& objects)
{
    if (g_current_taskid != 3)
    {
        reset_primary_pose_tracking();
        return;
    }

    const Object* person = select_best_pose_person(objects);
    if (!person)
    {
        g_primary_pose_missing_frames = std::min(g_primary_pose_missing_frames + 1, kPrimaryPoseGraceFrames + 1);
        if (g_primary_pose_missing_frames > kPrimaryPoseGraceFrames)
            reset_primary_pose_tracking();

        objects.clear();
        return;
    }

    Object selected = *person;
    g_primary_pose_rect = selected.rect;
    g_has_primary_pose_person = true;
    g_primary_pose_missing_frames = 0;

    objects.clear();
    objects.push_back(selected);
}

static JumpRopeFrame make_jumprope_frame_from_object(const Object& obj)
{
    JumpRopePoseKeypoint keypoints[17];
    for (int i = 0; i < 17; i++)
    {
        keypoints[i].x = obj.keypoints[i].p.x;
        keypoints[i].y = obj.keypoints[i].p.y;
        keypoints[i].prob = obj.keypoints[i].prob;
    }

    return make_jumprope_frame_from_pose(keypoints, 17, obj.rect.height);
}

static void publish_jumprope_result(const JumpRopeResult& result)
{
    g_jumprope_state.store(result.state, std::memory_order_relaxed);
    g_jumprope_count.store(result.count, std::memory_order_relaxed);
}

static void update_jumprope_state(const cv::Mat& rgb, const std::vector<Object>& objects)
{
    if (g_current_taskid != 3 || rgb.empty())
    {
        reset_jumprope_tracking();
        return;
    }

    JumpRopeFrame frame = {false, 0.f, 0.f, 0.f};
    const Object* person = select_best_pose_person(objects);

    // Track person visibility for colored bbox drawing
    if (person && person->keypoints.size() >= 17)
    {
        g_person_bbox = cv::Rect((int)person->rect.x, (int)person->rect.y,
                                 (int)person->rect.width, (int)person->rect.height);
        g_person_visible = true;
    }
    else
    {
        g_person_visible = false;
    }

    // Supply raw keypoints to the counter for feature extraction
    if (person && person->keypoints.size() >= 17)
    {
        JumpRopePoseKeypoint kpts[17];
        for (int i = 0; i < 17; i++)
        {
            kpts[i].x = person->keypoints[i].p.x;
            kpts[i].y = person->keypoints[i].p.y;
            kpts[i].prob = person->keypoints[i].prob;
        }
        g_jumprope_counter.set_frame_keypoints(kpts, 17);
        frame = make_jumprope_frame_from_pose(kpts, 17, person->rect.height);
    }
    else
    {
        g_jumprope_counter.set_frame_keypoints(nullptr, 0);
    }

    int prev_count = g_jumprope_counter.count();
    JumpRopeResult result = g_jumprope_counter.update(frame);

    // Signal baseline calibration to the Java layer (one-shot).
    if (g_jumprope_counter.just_calibrated())
    {
        g_jumprope_calibrated.store(true, std::memory_order_relaxed);

        // --- Profile matching: identify the person who just appeared ---
        if (person && person->keypoints.size() >= 17)
        {
            JumpRopePoseKeypoint kpts[17];
            for (int i = 0; i < 17; i++)
            {
                kpts[i].x = person->keypoints[i].p.x;
                kpts[i].y = person->keypoints[i].p.y;
                kpts[i].prob = person->keypoints[i].prob;
            }

            float body_feat[PROFILE_BODY_FEAT_DIM];
            float color_hist[PROFILE_HIST_DIM];
            bool has_body = compute_body_features(kpts, 17, body_feat);
            compute_color_features(rgb.data, rgb.cols, rgb.rows, kpts, 17, color_hist);

            if (has_body)
            {
                int match_idx = g_profile_manager.match(body_feat, color_hist);
                int profile_idx = g_profile_manager.update_or_create(
                    match_idx, body_feat, color_hist);

                if (match_idx >= 0)
                {
                    // Returning person: restore their accumulated count.
                    int old_total = g_profile_manager.profile(match_idx).total_jumps;
                    g_prev_session_total.store(old_total, std::memory_order_relaxed);
                }
                else
                {
                    // New person: start from 0.
                    g_prev_session_total.store(0, std::memory_order_relaxed);
                }
                g_profile_manager.set_active_profile(profile_idx);
            }
        }
    }

    // Signal person lost to the Java layer (one-shot).
    if (g_jumprope_counter.just_lost())
    {
        g_jumprope_lost.store(true, std::memory_order_relaxed);

        // Save the cumulative total to the active profile BEFORE resetting.
        int active = g_profile_manager.active_profile();
        if (active >= 0)
        {
            int rejected = g_jumprope_rejected.load(std::memory_order_relaxed);
            int session_gated = g_jumprope_counter.count() - rejected;
            if (session_gated < 0) session_gated = 0;
            int prev_total = g_prev_session_total.load(std::memory_order_relaxed);
            int final_total = prev_total + session_gated;
            g_profile_manager.set_total_jumps(active, final_total);

            // Keep g_prev_session_total at the saved total so the display
            // continues showing the departed person's final count instead of
            // dropping to 0.  The counter is reset below (count_=0), so the
            // gated formula (prev_total + 0) will hold the final value.
            g_prev_session_total.store(final_total, std::memory_order_relaxed);
        }

        // Mark no profile as active so the panel draws the saved total_jumps
        // for the departed person (greyed out) instead of the live count.
        g_profile_manager.set_active_profile(-1);

        // Reset the counter so its internal count_ returns to 0.  The counter
        // does NOT self-reset on person-lost — it keeps returning the old
        // session's count in NO_PERSON state.  Without resetting, when the
        // person returns the gated formula (prev_session_total + count) would
        // add the old session's count on top of the saved total, producing a
        // double-count.
        g_jumprope_counter.reset();

        // Reset the MLP rejection offset for the next session.
        g_jumprope_rejected.store(0, std::memory_order_relaxed);
    }

    // Publish the RAW count (counter output before MLP filtering) so the UI
    // can show both the unfiltered count and the MLP-gated count.
    g_jumprope_raw_count.store(result.count, std::memory_order_relaxed);

    // If a cycle just completed, handle classification and recording
    if (g_jumprope_counter.cycle_just_completed() && g_jumprope_counter.has_last_features())
    {
        const JumpRopeFeatures& features = g_jumprope_counter.last_features();

        // --- Recording: write features + pseudo-label to CSV ---
        if (g_recording_active && g_recording_file)
        {
            // Label: 1 = counted (positive), 0 = rejected by cooldown (negative)
            int label = (result.count > prev_count) ? 1 : 0;
            char csv_buf[512];
            jumprope_features_to_csv(features, csv_buf, sizeof(csv_buf));
            std::fprintf(g_recording_file, "%d,%s\n", label, csv_buf);
            std::fflush(g_recording_file);
            g_recording_sample_count++;
        }

        // --- Classification: gate the count with MLP if loaded ---
        // Only a cycle that actually incremented the raw count is eligible for
        // rejection (cooldown "negative sample" cycles complete without
        // incrementing and must not affect the gated count).
        if (g_jumprope_classifier.is_loaded() && result.count > prev_count)
        {
            float score = g_jumprope_classifier.predict(features);
            if (score < g_jumprope_classifier.threshold())
            {
                // MLP says this is NOT a valid jump. Record the rejection as a
                // persistent offset so the gated count stays suppressed on
                // every subsequent frame, not just this completion frame.
                g_jumprope_rejected.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    // Gated (mlp) count = raw count minus everything the MLP has rejected,
    // plus any accumulated count from previous sessions of the matched profile.
    int rejected = g_jumprope_rejected.load(std::memory_order_relaxed);
    int gated = result.count - rejected;
    if (gated < 0)
        gated = 0;
    int prev_total = g_prev_session_total.load(std::memory_order_relaxed);
    result.count = prev_total + gated;

    publish_jumprope_result(result);
}

static void draw_jumprope_status(cv::Mat& rgb)
{
    const int state = g_jumprope_state.load(std::memory_order_relaxed);

    // ---- Top-left: compact status line ----
    {
        const char* status_text;
        if (state == JUMP_ROPE_STATE_NO_PERSON)
            status_text = "No person";
        else if (state == JUMP_ROPE_STATE_READY)
            status_text = "Ready";
        else if (state == JUMP_ROPE_STATE_COUNTING)
            status_text = "Counting";
        else
            status_text = "Switch to pose";

        int baseLine = 0;
        cv::Size sz = cv::getTextSize(status_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        const int top_margin = 40;  // below system status bar
        const int pad = 4;
        cv::Rect bg_rect(0, top_margin, sz.width + pad * 2, top_margin + sz.height + baseLine + pad * 2);
        // Semi-transparent background (matches left panel)
        cv::Mat roi = rgb(bg_rect);
        cv::Mat overlay;
        roi.convertTo(overlay, -1);
        overlay.setTo(cv::Scalar(0, 0, 0));
        cv::addWeighted(roi, 0.45, overlay, 0.55, 0, roi);

        cv::putText(rgb, status_text, cv::Point(pad, top_margin + pad + sz.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    }

    // ---- Left side: compact per-person count panel ----
    int n_profiles = g_profile_manager.profile_count();
    {
        // Panel layout: per-profile rows + optional raw-count debug row
        const int row_h = 24;
        const int n_rows = (n_profiles > 0) ? n_profiles + 1 : 1;
        const int panel_w = 110;
        const int panel_h = n_rows * row_h + 8;
        const int panel_x = 0;
        const int panel_y = 70;  // below the status line

        // Semi-transparent background
        cv::Mat roi = rgb(cv::Rect(panel_x, panel_y, panel_w, panel_h));
        cv::Mat overlay;
        roi.convertTo(overlay, -1);
        overlay.setTo(cv::Scalar(0, 0, 0));
        cv::addWeighted(roi, 0.45, overlay, 0.55, 0, roi);

        int active = g_profile_manager.active_profile();
        for (int i = 0; i < n_profiles; i++)
        {
            const PersonProfile& p = g_profile_manager.profile(i);
            cv::Scalar color = kProfileColors[i % MAX_PROFILES];
            int row_y = panel_y + 4 + i * row_h;

            // Colored indicator square
            cv::rectangle(rgb,
                          cv::Rect(panel_x + 4, row_y + 3, 10, 10),
                          color, -1);

            // Text with count
            char line[24];
            int display_count = p.total_jumps;
            if (i == active)
            {
                // Active profile: show prev_session_total + current gated count
                display_count = g_jumprope_count.load(std::memory_order_relaxed);
            }
            std::snprintf(line, sizeof(line), "P%d: %d", p.id, display_count);

            cv::Scalar text_color = (i == active)
                ? cv::Scalar(255, 255, 255)
                : cv::Scalar(160, 160, 160);
            cv::putText(rgb, line, cv::Point(panel_x + 18, row_y + 14),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1);
        }

        // Raw count row: shows the counter's raw count (before MLP gating)
        // and the MLP-gated count so the developer can see the rejection rate.
        {
            int raw = g_jumprope_raw_count.load(std::memory_order_relaxed);
            int gated_count = g_jumprope_count.load(std::memory_order_relaxed);
            int row_y = panel_y + 4 + n_profiles * row_h;
            char line[32];
            std::snprintf(line, sizeof(line), "R:%d G:%d", raw, gated_count);
            cv::putText(rgb, line, cv::Point(panel_x + 4, row_y + 14),
                        cv::FONT_HERSHEY_SIMPLEX, 0.32, cv::Scalar(120, 120, 120), 1);
        }
    }

    // ---- Draw colored bounding box around detected person ----
    if (g_person_visible)
    {
        int active = g_profile_manager.active_profile();
        cv::Scalar box_color = (active >= 0)
            ? kProfileColors[active % MAX_PROFILES]
            : cv::Scalar(255, 255, 255);

        // Outer glow (thicker, semi-transparent via double draw)
        cv::rectangle(rgb, g_person_bbox, box_color, 3);

        // Small label at top-left of bbox
        if (active >= 0)
        {
            char lbl[8];
            std::snprintf(lbl, sizeof(lbl), "P%d", g_profile_manager.profile(active).id);
            cv::putText(rgb, lbl,
                        cv::Point(g_person_bbox.x, g_person_bbox.y - 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1);
        }
    }
}

class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    // yolov8
    {
        ncnn::MutexLockGuard g(lock);

        if (g_yolov8)
        {
            std::vector<Object> objects;
            g_yolov8->detect(rgb, objects);

            keep_primary_pose_person(objects);
            update_jumprope_state(rgb, objects);
            g_yolov8->draw(rgb, objects);
            draw_jumprope_status(rgb);
        }
        else
        {
            g_jumprope_state.store(JUMP_ROPE_STATE_INACTIVE, std::memory_order_relaxed);
            g_jumprope_count.store(0, std::memory_order_relaxed);
            g_jumprope_raw_count.store(0, std::memory_order_relaxed);
            g_jumprope_rejected.store(0, std::memory_order_relaxed);
            draw_unsupported(rgb);
        }
    }

    draw_fps(rgb);
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_yolov8;
        g_yolov8 = 0;
    }

    ncnn::destroy_gpu_instance();

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int taskid, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint taskid, jint modelid, jint cpugpu)
{
    if (taskid < 0 || taskid > 5 || modelid < 0 || modelid > 4 || cpugpu < 0 || cpugpu > 2)
    {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    const char* tasknames[6] =
    {
        "",
        "_oiv7",
        "_seg",
        "_pose",
        "_cls",
        "_obb"
    };

    // modelid 0-2: yolov8-n @ 320/480/640
    // modelid 3-4: yolo11-n @ 320/640
    const char* modeltypes[5] =
    {
        "n",       // modelid 0: v8-n @ 320
        "n",       // modelid 1: v8-n @ 480
        "n",       // modelid 2: v8-n @ 640
        "n_320",   // modelid 3: v11-n @ 320
        "n_640"    // modelid 4: v11-n @ 640
    };

    const char* prefix = ((int)modelid < 3) ? "yolov8" : "yolo11";

    std::string parampath = std::string(prefix) + modeltypes[(int)modelid] + tasknames[(int)taskid] + ".ncnn.param";
    std::string modelpath = std::string(prefix) + modeltypes[(int)modelid] + tasknames[(int)taskid] + ".ncnn.bin";
    bool use_gpu = (int)cpugpu == 1;
    bool use_turnip = (int)cpugpu == 2;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        {
            static int old_taskid = 0;
            static int old_modelid = 0;
            static int old_cpugpu = 0;

            g_current_taskid = (int)taskid;
            reset_jumprope_tracking();

            if (taskid != old_taskid || modelid != old_modelid || cpugpu != old_cpugpu)
            {
                // taskid or model or cpugpu changed
                delete g_yolov8;
                g_yolov8 = 0;
            }
            old_taskid = taskid;
            old_modelid = (int)modelid;
            old_cpugpu = cpugpu;

            ncnn::destroy_gpu_instance();

            if (use_turnip)
            {
                ncnn::create_gpu_instance("libvulkan_freedreno.so");
            }
            else if (use_gpu)
            {
                ncnn::create_gpu_instance();
            }

            if (!g_yolov8)
            {
                if (taskid == 0) g_yolov8 = new YOLOv8_det_coco;
                if (taskid == 1) g_yolov8 = new YOLOv8_det_oiv7;
                if (taskid == 2) g_yolov8 = new YOLOv8_seg;
                if (taskid == 3)
                {
                    if ((int)modelid < 3)
                        g_yolov8 = new YOLOv8_pose;
                    else
                        g_yolov8 = new YOLOv11_pose;
                }
                if (taskid == 4) g_yolov8 = new YOLOv8_cls;
                if (taskid == 5) g_yolov8 = new YOLOv8_obb;

                g_yolov8->load(mgr, parampath.c_str(), modelpath.c_str(), use_gpu || use_turnip);
            }
            int target_size;
            if ((int)modelid == 0)       target_size = 320;
            else if ((int)modelid == 1)  target_size = 480;
            else if ((int)modelid == 2)  target_size = 640;
            else if ((int)modelid == 3)  target_size = 320;
            else                         target_size = 640;  // modelid 4
            g_yolov8->set_det_target_size(target_size);
        }
    }

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int)facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    {
        ncnn::MutexLockGuard g(lock);
        reset_jumprope_tracking();
    }

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

// public native int getJumpRopeState();
JNIEXPORT jint JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_getJumpRopeState(JNIEnv* env, jobject thiz)
{
    return g_jumprope_state.load(std::memory_order_relaxed);
}

// public native int getJumpRopeCount();
JNIEXPORT jint JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_getJumpRopeCount(JNIEnv* env, jobject thiz)
{
    return g_jumprope_count.load(std::memory_order_relaxed);
}

// public native int getJumpRopeRawCount();
JNIEXPORT jint JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_getJumpRopeRawCount(JNIEnv* env, jobject thiz)
{
    return g_jumprope_raw_count.load(std::memory_order_relaxed);
}

// public native boolean startRecording(String path);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_startRecording(JNIEnv* env, jobject thiz, jstring path)
{
    if (g_recording_file)
    {
        std::fclose(g_recording_file);
        g_recording_file = nullptr;
    }

    const char* cpath = env->GetStringUTFChars(path, nullptr);
    if (!cpath)
        return JNI_FALSE;

    g_recording_file = std::fopen(cpath, "w");
    env->ReleaseStringUTFChars(path, cpath);

    if (!g_recording_file)
        return JNI_FALSE;

    // Write CSV header: label,feature1,feature2,...
    std::fprintf(g_recording_file, "label,%s\n", jumprope_features_csv_header());
    std::fflush(g_recording_file);
    g_recording_active = true;
    g_recording_sample_count = 0;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Recording started: sample_count=0");
    return JNI_TRUE;
}

// public native void stopRecording();
JNIEXPORT void JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_stopRecording(JNIEnv* env, jobject thiz)
{
    if (g_recording_file)
    {
        std::fflush(g_recording_file);
        std::fclose(g_recording_file);
        g_recording_file = nullptr;
    }
    g_recording_active = false;
    // Do NOT reset g_recording_sample_count here — the Java layer
    // reads it immediately after this call via getRecordingSampleCount().
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Recording stopped: total samples=%d", g_recording_sample_count);
}

// public native int getRecordingSampleCount();
JNIEXPORT jint JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_getRecordingSampleCount(JNIEnv* env, jobject thiz)
{
    return g_recording_sample_count;
}

// public native boolean loadClassifier(String path);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_loadClassifier(JNIEnv* env, jobject thiz, jstring path)
{
    const char* cpath = env->GetStringUTFChars(path, nullptr);
    if (!cpath)
        return JNI_FALSE;

    bool ok = g_jumprope_classifier.load(cpath);
    env->ReleaseStringUTFChars(path, cpath);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Classifier load: %s", ok ? "OK" : "FAIL");
    return ok ? JNI_TRUE : JNI_FALSE;
}

// public native boolean isClassifierLoaded();
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_isClassifierLoaded(JNIEnv* env, jobject thiz)
{
    return g_jumprope_classifier.is_loaded() ? JNI_TRUE : JNI_FALSE;
}

// public native void setClassifierThreshold(float threshold);
JNIEXPORT void JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_setClassifierThreshold(JNIEnv* env, jobject thiz, jfloat threshold)
{
    g_jumprope_classifier.set_threshold((float)threshold);
}

// public native boolean isBaselineCalibrated();
// Returns true once when the standing-still detector has calibrated the
// baseline for the first time.  Subsequent calls return false until the
// next reset.
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_isBaselineCalibrated(JNIEnv* env, jobject thiz)
{
    bool expected = true;
    if (g_jumprope_calibrated.compare_exchange_strong(expected, false, std::memory_order_relaxed))
    {
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

// public native boolean isPersonLost();
// Returns true once when the person is declared missing (after grace
// period).  Subsequent calls return false until the next loss event.
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_isPersonLost(JNIEnv* env, jobject thiz)
{
    bool expected = true;
    if (g_jumprope_lost.compare_exchange_strong(expected, false, std::memory_order_relaxed))
    {
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

}
