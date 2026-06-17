# JumpRope-YOLO Design

## Background

The original project implemented a mobile DeskGuard demo for tabletop object risk detection. The teacher rejected the desktop-recognition scenario because it lacked a practical application context.

The project will pivot to a student home physical education scenario: a phone camera records a student jumping rope, YOLO pose runs on-device through NCNN, and the app counts completed jumps in real time.

## Product Goal

Build a mobile-side jump rope counting demo that keeps the project's core technical theme:

- YOLO model deployment on Android.
- NCNN inference with CPU, Vulkan GPU, and Turnip options.
- Real-time camera preview and on-device post-processing.
- A practical family PE use case for students.

## Non-Goals

- Do not build a cloud service.
- Do not require a custom labeled jump rope dataset for the first usable version.
- Do not detect the rope itself in the first version; the count is based on human pose motion.
- Do not rewrite the whole Android demo structure unless required.

## Model Strategy

Use the existing YOLO pose assets first:

- `yolov8n_pose.ncnn.param`
- `yolov8n_pose.ncnn.bin`
- optional `s/m` pose variants already present in Android assets

The implementation should default to the pose task because `YOLOv8_pose` already outputs COCO-style 17 keypoints through the current NCNN post-processing path.

For the report and later model replacement, document a YOLO11 route:

1. Export `yolo11n-pose.pt` to TorchScript.
2. Use pnnx to convert TorchScript to NCNN.
3. Keep Android input/output blob names compatible with the pose post-processing path, or add a YOLO11 pose adapter if names/shapes differ.
4. Reuse the same jump counting logic because it consumes normalized keypoint positions, not model internals.

## Android Architecture

Keep the current Android camera and NCNN integration:

- `YOLOv8Ncnn.loadModel(...)` still loads assets and selects CPU/GPU/Turnip.
- `YOLOv8::load(...)` still sets `opt.use_vulkan_compute` based on the user-selected GPU mode.
- `JNI_OnLoad` and model reload continue to manage NCNN GPU instances.

Replace the DeskGuard application state with JumpRope state:

- `INACTIVE`: camera/model unavailable or not in pose task.
- `NO_PERSON`: pose task is active, but no reliable person is detected.
- `READY`: a reliable person is visible, waiting for motion.
- `COUNTING`: jumps are being counted.

Expose native methods to Java:

- `getJumpRopeState()`
- `getJumpRopeCount()`

Reset counting when:

- the camera is closed,
- the model is reloaded,
- the task changes,
- or the user switches model/GPU mode.

Short temporary person loss should not immediately reset the count while the camera session stays active.

## Counting Algorithm

Use the largest/highest-confidence person from `YOLOv8_pose` output.

Recommended keypoints:

- shoulders: 5, 6
- hips: 11, 12
- knees: 13, 14
- ankles: 15, 16

Per frame:

1. Validate that enough torso/lower-body keypoints have confidence above a threshold.
2. Compute a body center Y coordinate from hips when available; fall back to shoulders plus ankles if hips are unreliable.
3. Smooth the center Y with an exponential moving average.
4. Estimate body scale from bounding-box height or shoulder-to-ankle distance.
5. Convert vertical motion to a scale-relative signal.
6. Use a two-phase state machine:
   - `GROUND_OR_LOW`: body is near its lower position.
   - `AIR_OR_HIGH`: body center rises above the adaptive threshold.
7. Count one jump when the signal completes a low -> high -> low cycle.
8. Add a minimum frame cooldown to avoid double-counting camera or keypoint jitter.

Initial thresholds:

- keypoint confidence: `0.25`
- minimum vertical amplitude: `0.035 * body_height`
- count cooldown: about `8` frames
- person missing grace: about `15` frames

The exact constants can be tuned after Android build verification.

## User Interface

Change presentation from DeskGuard warning to JumpRope exercise feedback:

- App name: `JumpRope-YOLO`
- Status bar examples:
  - `JumpRope: ready`
  - `JumpRope: counting 12`
  - `JumpRope: no person`
  - `JumpRope: switch to pose`

Default task should be pose so the first screen already matches the project purpose.

The original alert vibration/beep should be repurposed:

- no danger alarm,
- optional short tone on each counted jump,
- no repeated long alarm cooldown.

## Documentation Changes

Replace or supersede DeskGuard documents with JumpRope documents:

- Android algorithm document: jump rope pose counting and mobile deployment.
- Model training/export document: pose model export, YOLO11 pose replacement path, and optional evaluation dataset.
- Development diary: record the pivot from DeskGuard to JumpRope-YOLO.

Avoid leaving teacher-facing docs centered on desktop/tabletop recognition.

## Verification Plan

Model-side verification:

- Update Python tests to match JumpRope naming/config where retained.
- Run the training/export unit tests.

Android verification:

- Build Android debug APK.
- Verify JNI method names match Java declarations.
- Verify the default model loads pose assets.
- Verify CPU/GPU/Turnip selections still call the existing NCNN Vulkan path.

Repository verification:

- Search for remaining user-facing `DeskGuard`, `desktop`, `tabletop`, and desk-risk wording.
- Keep internal package/library names only if renaming them adds risk without improving the demo.

## Risks

- Pose keypoints may jitter during fast jumps; smoothing and cooldown are required.
- A side-view camera can hide one leg; the algorithm must work with partial lower-body keypoints.
- Actual rope visibility is not required, so the report should clearly say the first version counts jump motion from pose, not rope rotations.
- YOLO11 pose conversion may need separate output-shape adaptation; this is documented as a replacement path, not a blocker for the first app version.
