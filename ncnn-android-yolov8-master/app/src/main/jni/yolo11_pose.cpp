// YOLOv11 pose detect for NCNN
//
// v11 model exports a single merged output tensor "out0" of shape (56, N):
//   [0:4]   bbox [cx, cy, w, h] in padded input pixels
//   [4]     confidence score (already sigmoid)
//   [5:56]  51 keypoints = 17 * (x, y, conf)
//
// where N = 8400 (640x640 input) or 2100 (320x320 input).
// Key differences from v8:
//   - Single output blob (no separate "out1")
//   - DFL, anchor/stride decode, and sigmoid are done inside the model
//   - Columns are in (56, N) layout, needs transpose to (N, 56) for row access

#include "yolov8.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <float.h>
#include <stdio.h>
#include <string.h>
#include <vector>

// Tracked globally in yolov8ncnn.cpp — skip "person" label in jump-rope mode.
extern int g_current_taskid;

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    {
        if (left < j) qsort_descent_inplace(objects, left, j);
        if (i < right) qsort_descent_inplace(objects, i, right);
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

int YOLOv11_pose::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    const int target_size = det_target_size;
    const float prob_threshold = 0.50f;
    const float nms_threshold = 0.45f;

    int img_w = rgb.cols;
    int img_h = rgb.rows;

    // yolo11n_320/640_pose.ncnn.param has static reshape and anchor tables.
    // The padded input must match the exported square size exactly.
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);

    int wpad = target_size - w;
    int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov8.create_extractor();
    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    // v11 out0 shape: h=56, w=num_anchors:
    // [cx,cy,w,h(4), score(1), keypoints(51)]
    const int num_anchors = out.w;
    const int total_cols = out.h;     // 56
    const int num_points = 17;

    // Transpose from (56, num_anchors) to (num_anchors, 56)
    // ncnn::Mat constructor: Mat(w, h, elemsize) — w=cols, h=rows
    ncnn::Mat out_t(total_cols, num_anchors, sizeof(float));
    for (int i = 0; i < num_anchors; i++)
    {
        float* dst = out_t.row(i);
        for (int j = 0; j < total_cols; j++)
        {
            dst[j] = out.row(j)[i];
        }
    }

    std::vector<Object> proposals;

    for (int i = 0; i < num_anchors; i++)
    {
        const float* row = out_t.row(i);

        float score = row[4];
        if (score < prob_threshold) continue;

        const float cx = row[0];
        const float cy = row[1];
        const float bw = row[2];
        const float bh = row[3];
        const float x0 = cx - bw * 0.5f;
        const float y0 = cy - bh * 0.5f;
        const float x1 = cx + bw * 0.5f;
        const float y1 = cy + bh * 0.5f;

        std::vector<KeyPoint> keypoints;
        keypoints.reserve(num_points);
        for (int k = 0; k < num_points; k++)
        {
            KeyPoint kp;
            const float* kpt = row + 5 + k * 3;
            kp.p.x = kpt[0];
            kp.p.y = kpt[1];
            kp.prob = kpt[2];
            keypoints.push_back(kp);
        }

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.label = 0;
        obj.prob = score;
        obj.keypoints = keypoints;
        proposals.push_back(obj);
    }

    // sort by score descending
    qsort_descent_inplace(proposals);

    // NMS
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust to original image coordinates
        float ox0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float oy0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float ox1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float oy1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        for (int j = 0; j < num_points; j++)
        {
            objects[i].keypoints[j].p.x = (objects[i].keypoints[j].p.x - (wpad / 2)) / scale;
            objects[i].keypoints[j].p.y = (objects[i].keypoints[j].p.y - (hpad / 2)) / scale;
        }

        ox0 = std::max(std::min(ox0, (float)(img_w - 1)), 0.f);
        oy0 = std::max(std::min(oy0, (float)(img_h - 1)), 0.f);
        ox1 = std::max(std::min(ox1, (float)(img_w - 1)), 0.f);
        oy1 = std::max(std::min(oy1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = ox0;
        objects[i].rect.y = oy0;
        objects[i].rect.width = ox1 - ox0;
        objects[i].rect.height = oy1 - oy0;
    }

    // sort by area
    struct
    {
        bool operator()(const Object& a, const Object& b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);

    return 0;
}

int YOLOv11_pose::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"person"};

    static const cv::Scalar colors[] = {
        cv::Scalar( 67,  54, 244), cv::Scalar( 30,  99, 233), cv::Scalar( 39, 176, 156),
        cv::Scalar( 58, 183, 103), cv::Scalar( 81, 181,  63), cv::Scalar(150, 243,  33),
        cv::Scalar(169, 244,   3), cv::Scalar(188, 212,   0), cv::Scalar(150, 136,   0),
        cv::Scalar(175,  80,  76), cv::Scalar(195,  74, 139), cv::Scalar(220,  57, 205),
        cv::Scalar(235,  59, 255), cv::Scalar(193,   7, 255), cv::Scalar(152,   0, 255),
        cv::Scalar( 87,  34, 255), cv::Scalar( 85,  72, 121), cv::Scalar(158, 158, 158),
        cv::Scalar(125, 139,  96)
    };

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        const cv::Scalar& color = colors[i % 19];

        // draw bones
        static const int joint_pairs[16][2] = {
            {0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 6}, {5, 7}, {7, 9},
            {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13},
            {12, 14}, {13, 15}, {14, 16}
        };
        static const cv::Scalar bone_colors[] = {
            cv::Scalar(  0,   0, 255), cv::Scalar(  0,   0, 255), cv::Scalar(  0,   0, 255),
            cv::Scalar(  0,   0, 255), cv::Scalar(  0, 255, 128), cv::Scalar(  0, 255, 128),
            cv::Scalar(  0, 255, 128), cv::Scalar(  0, 255, 128), cv::Scalar(  0, 255, 128),
            cv::Scalar(255, 255,  51), cv::Scalar(255, 255,  51), cv::Scalar(255, 255,  51),
            cv::Scalar(255,  51, 153), cv::Scalar(255,  51, 153), cv::Scalar(255,  51, 153),
            cv::Scalar(255,  51, 153),
        };

        for (int j = 0; j < 16; j++)
        {
            const KeyPoint& p1 = obj.keypoints[joint_pairs[j][0]];
            const KeyPoint& p2 = obj.keypoints[joint_pairs[j][1]];
            if (p1.prob < 0.2f || p2.prob < 0.2f) continue;
            cv::line(rgb, p1.p, p2.p, bone_colors[j], 2);
        }

        // draw joints
        for (size_t j = 0; j < obj.keypoints.size(); j++)
        {
            const KeyPoint& keypoint = obj.keypoints[j];
            if (keypoint.prob < 0.2f) continue;
            cv::circle(rgb, keypoint.p, 3, color, -1);
        }

        cv::rectangle(rgb, obj.rect, color);

        // Skip "person XX.X%" label in jump-rope mode (taskid=3) to avoid
        // overlapping with our per-person profile labels and count panel.
        if (g_current_taskid != 3)
        {
            char text[256];
            sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = obj.rect.x;
            int y = obj.rect.y - label_size.height - baseLine;
            if (y < 0) y = 0;
            if (x + label_size.width > rgb.cols) x = rgb.cols - label_size.width;

            cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), -1);
            cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
    }

    return 0;
}
