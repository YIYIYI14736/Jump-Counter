#include "jumprope_counter.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace
{

void expect_true(bool condition, const std::string& message, int exit_code)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(exit_code);
    }
}

void expect_near(float actual, float expected, float epsilon, const std::string& message, int exit_code)
{
    if (std::fabs(actual - expected) > epsilon)
    {
        std::cerr << "FAIL: " << message << " expected " << expected << " got " << actual << std::endl;
        std::exit(exit_code);
    }
}

JumpRopeFrame person_frame(float center_y, float body_height = 300.f)
{
    JumpRopeFrame frame;
    frame.has_person = true;
    frame.center_y = center_y;
    frame.body_height = body_height;
    frame.confidence = 1.f;
    return frame;
}

JumpRopeFrame missing_frame()
{
    JumpRopeFrame frame;
    frame.has_person = false;
    frame.center_y = 0.f;
    frame.body_height = 0.f;
    frame.confidence = 0.f;
    return frame;
}

void set_keypoint(JumpRopePoseKeypoint* keypoints, int index, float y, float prob)
{
    keypoints[index].x = 100.f + index;
    keypoints[index].y = y;
    keypoints[index].prob = prob;
}

void feed_stable_ready(JumpRopeCounter& counter)
{
    for (int i = 0; i < 8; i++)
        counter.update(person_frame(500.f));
}

void test_missing_person_stays_no_person()
{
    JumpRopeCounter counter;
    JumpRopeResult result = counter.update(missing_frame());

    expect_true(result.state == JUMP_ROPE_STATE_NO_PERSON, "missing person should report NO_PERSON", 11);
    expect_true(result.count == 0, "missing person should keep count at zero", 12);
}

void test_stable_person_enters_ready_without_counting()
{
    JumpRopeCounter counter;
    feed_stable_ready(counter);

    expect_true(counter.state() == JUMP_ROPE_STATE_READY, "stable person should enter READY", 21);
    expect_true(counter.count() == 0, "stable person should not count a jump", 22);
}

void test_one_low_high_low_cycle_counts_once()
{
    JumpRopeCounter counter;
    feed_stable_ready(counter);

    const float motion[] = {
        486.f, 474.f, 462.f, 450.f, 438.f, 426.f,
        438.f, 452.f, 468.f, 486.f, 500.f, 506.f,
    };
    for (float y : motion)
        counter.update(person_frame(y));

    expect_true(counter.state() == JUMP_ROPE_STATE_COUNTING, "completed jump should enter COUNTING", 31);
    expect_true(counter.count() == 1, "one low-high-low cycle should count once", 32);
}

void test_small_jitter_does_not_count()
{
    JumpRopeCounter counter;
    feed_stable_ready(counter);

    const float jitter[] = {
        499.f, 503.f, 497.f, 501.f, 500.f, 504.f,
        498.f, 502.f, 499.f, 501.f, 500.f, 503.f,
    };
    for (float y : jitter)
        counter.update(person_frame(y));

    expect_true(counter.state() == JUMP_ROPE_STATE_READY, "small jitter should remain READY", 41);
    expect_true(counter.count() == 0, "small jitter should not count", 42);
}

void test_temporary_missing_person_preserves_count()
{
    JumpRopeCounter counter;
    feed_stable_ready(counter);

    const float motion[] = {
        486.f, 474.f, 462.f, 450.f, 438.f, 426.f,
        438.f, 452.f, 468.f, 486.f, 500.f, 506.f,
    };
    for (float y : motion)
        counter.update(person_frame(y));

    for (int i = 0; i < 5; i++)
        counter.update(missing_frame());

    expect_true(counter.state() == JUMP_ROPE_STATE_COUNTING, "short person loss should preserve COUNTING state", 51);
    expect_true(counter.count() == 1, "short person loss should preserve count", 52);
}

void test_reset_clears_state_and_count()
{
    JumpRopeCounter counter;
    feed_stable_ready(counter);

    const float motion[] = {
        486.f, 474.f, 462.f, 450.f, 438.f, 426.f,
        438.f, 452.f, 468.f, 486.f, 500.f, 506.f,
    };
    for (float y : motion)
        counter.update(person_frame(y));

    counter.reset();

    expect_true(counter.state() == JUMP_ROPE_STATE_INACTIVE, "reset should restore INACTIVE state", 61);
    expect_true(counter.count() == 0, "reset should clear count", 62);
}

void test_pose_keypoints_make_frame_from_hips()
{
    JumpRopePoseKeypoint keypoints[17] = {};
    set_keypoint(keypoints, 11, 460.f, 0.92f);
    set_keypoint(keypoints, 12, 464.f, 0.88f);

    JumpRopeFrame frame = make_jumprope_frame_from_pose(keypoints, 17, 310.f);

    expect_true(frame.has_person, "reliable hip keypoints should make a person frame", 71);
    expect_near(frame.center_y, 462.f, 0.01f, "hip center y should be averaged when only hips are reliable", 72);
    expect_near(frame.body_height, 310.f, 0.01f, "bbox height should be used as body height", 73);
}

void test_pose_keypoints_use_torso_scale_when_bbox_is_unstable()
{
    JumpRopePoseKeypoint keypoints[17] = {};
    set_keypoint(keypoints, 5, 300.f, 0.82f);
    set_keypoint(keypoints, 6, 304.f, 0.80f);
    set_keypoint(keypoints, 11, 424.f, 0.78f);
    set_keypoint(keypoints, 12, 420.f, 0.76f);

    JumpRopeFrame frame = make_jumprope_frame_from_pose(keypoints, 17, 900.f);

    expect_true(frame.has_person, "reliable torso keypoints should make a person frame", 101);
    expect_near(frame.center_y, 379.9f, 0.2f, "torso center should blend shoulders and hips", 102);
    expect_true(frame.body_height < 500.f, "unstable large bbox should not dominate body scale", 103);
    expect_true(frame.body_height > 250.f, "torso scale should remain large enough for jump thresholds", 104);
}

void test_upper_body_motion_counts_when_ankles_are_missing()
{
    JumpRopeCounter counter;
    feed_stable_ready(counter);

    const float motion[] = {
        493.f, 486.f, 478.f, 468.f, 458.f, 450.f,
        456.f, 466.f, 478.f, 490.f, 501.f, 508.f,
    };
    for (float y : motion)
        counter.update(person_frame(y, 320.f));

    expect_true(counter.state() == JUMP_ROPE_STATE_COUNTING, "upper-body jump motion should enter COUNTING", 111);
    expect_true(counter.count() == 1, "upper-body jump motion should count once", 112);
}

void test_noisy_v11_like_motion_counts_one_completed_cycle()
{
    JumpRopeCounter counter;
    feed_stable_ready(counter);

    const float motion[] = {
        501.f, 497.f, 492.f, 486.f, 477.f, 469.f, 459.f, 452.f,
        456.f, 452.f, 458.f, 466.f, 475.f, 487.f, 496.f, 504.f,
        499.f, 503.f, 501.f,
    };
    for (float y : motion)
        counter.update(person_frame(y, 330.f));

    expect_true(counter.state() == JUMP_ROPE_STATE_COUNTING, "noisy completed cycle should enter COUNTING", 121);
    expect_true(counter.count() == 1, "noisy completed cycle should count exactly once", 122);
}

void test_pose_keypoints_fallback_to_shoulders_and_ankles()
{
    JumpRopePoseKeypoint keypoints[17] = {};
    set_keypoint(keypoints, 5, 300.f, 0.84f);
    set_keypoint(keypoints, 6, 302.f, 0.81f);
    set_keypoint(keypoints, 15, 610.f, 0.78f);
    set_keypoint(keypoints, 16, 614.f, 0.76f);

    JumpRopeFrame frame = make_jumprope_frame_from_pose(keypoints, 17, 0.f);

    expect_true(frame.has_person, "shoulder and ankle keypoints should make a fallback frame", 81);
    expect_near(frame.center_y, 456.5f, 0.01f, "fallback center y should use shoulder-ankle midpoint", 82);
    expect_near(frame.body_height, 311.f, 0.01f, "fallback body height should use shoulder-ankle span", 83);
}

void test_pose_keypoints_reject_low_confidence_person()
{
    JumpRopePoseKeypoint keypoints[17] = {};
    set_keypoint(keypoints, 11, 460.f, 0.10f);
    set_keypoint(keypoints, 12, 464.f, 0.12f);
    set_keypoint(keypoints, 15, 610.f, 0.10f);
    set_keypoint(keypoints, 16, 614.f, 0.11f);

    JumpRopeFrame frame = make_jumprope_frame_from_pose(keypoints, 17, 310.f);

    expect_true(!frame.has_person, "low confidence pose should be rejected", 91);
}

} // namespace

int main()
{
    test_missing_person_stays_no_person();
    test_stable_person_enters_ready_without_counting();
    test_one_low_high_low_cycle_counts_once();
    test_small_jitter_does_not_count();
    test_temporary_missing_person_preserves_count();
    test_reset_clears_state_and_count();
    test_pose_keypoints_make_frame_from_hips();
    test_pose_keypoints_fallback_to_shoulders_and_ankles();
    test_pose_keypoints_reject_low_confidence_person();
    test_pose_keypoints_use_torso_scale_when_bbox_is_unstable();
    test_upper_body_motion_counts_when_ankles_are_missing();
    test_noisy_v11_like_motion_counts_one_completed_cycle();

    std::cout << "jumprope_counter_test: PASS" << std::endl;
    return 0;
}
