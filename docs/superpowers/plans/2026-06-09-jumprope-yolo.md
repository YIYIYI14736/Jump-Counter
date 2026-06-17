# JumpRope-YOLO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## File Map

- `ncnn-android-yolov8-master/app/src/main/jni/jumprope_counter.h`: pure C++ counter API and state definitions.
- `ncnn-android-yolov8-master/app/src/main/jni/jumprope_counter.cpp`: pure C++ jump-count state machine.
- `ncnn-android-yolov8-master/tests/jumprope_counter_test.cpp`: host-side unit tests compiled with local `g++`.
- `ncnn-android-yolov8-master/app/src/main/jni/CMakeLists.txt`: include `jumprope_counter.cpp` in the Android shared library.
- `ncnn-android-yolov8-master/app/src/main/jni/yolov8ncnn.cpp`: replace DeskGuard runtime state with JumpRope pose state, JNI getters, and camera overlay.
- `ncnn-android-yolov8-master/app/src/main/java/com/tencent/yolov8ncnn/YOLOv8Ncnn.java`: expose JumpRope state constants and native getters.
- `ncnn-android-yolov8-master/app/src/main/java/com/tencent/yolov8ncnn/MainActivity.java`: default to pose, poll count/state, update status, and repurpose tone feedback.
- `ncnn-android-yolov8-master/app/src/main/res/values/strings.xml`: replace DeskGuard user-facing strings with JumpRope strings.
- `ncnn-android-yolov8-master/app/src/main/res/layout/main.xml`: rename the status view id/text for JumpRope.
- `ncnn-android-yolov8-master/docs/jumprope_algorithm.md`: document pose-based counting and GPU deployment.
- `ncnn-android-yolov8-master/docs/deskguard_algorithm.md`: remove the obsolete tabletop-risk algorithm document.
- `DeskGuard-Model/`: rename to `JumpRope-Model` and update package paths from `training.deskguard_yolo` to `training.jumprope_yolo`.
- `JumpRope-Model/datasets/jumprope_pose.yaml`: pose dataset config for optional YOLO11/YOLOv8 pose fine-tuning.
- `JumpRope-Model/training/jumprope_yolo/*.py`: update training/export defaults and pose-label validation.
- `JumpRope-Model/training/jumprope_yolo/tests/*.py`: update tests first for JumpRope pose behavior.
- `JumpRope-Model/docs/training.md`: explain pretrained pose first, optional pose fine-tuning, and NCNN export.
- `开发日记.md`: record the project pivot and new verification results.

## Task 1: Add Pure Jump Counter With TDD

- [ ] **Step 1: Write failing C++ tests**
  - Create `ncnn-android-yolov8-master/tests/jumprope_counter_test.cpp`.
  - Include `jumprope_counter.h`.
  - Test these behaviors:
    - no person keeps state `NO_PERSON` and count `0`;
    - stable person enters `READY` and does not count;
    - one low -> high -> low cycle increments count once;
    - small jitter does not count;
    - temporary missing-person frames preserve count;
    - reset clears count and state.
  - Run:
    ```powershell
    g++ -std=c++17 -Wall -Wextra -I ncnn-android-yolov8-master/app/src/main/jni ncnn-android-yolov8-master/tests/jumprope_counter_test.cpp -o C:\tmp\jumprope_counter_test.exe
    ```
  - Expected result: compile fails because `jumprope_counter.h` does not exist.

- [ ] **Step 2: Implement minimal counter**
  - Add `jumprope_counter.h` and `jumprope_counter.cpp`.
  - Define:
    - `enum JumpRopeState { JUMP_ROPE_STATE_INACTIVE = 0, JUMP_ROPE_STATE_NO_PERSON = 1, JUMP_ROPE_STATE_READY = 2, JUMP_ROPE_STATE_COUNTING = 3 };`
    - `struct JumpRopeFrame { bool has_person; float center_y; float body_height; };`
    - `struct JumpRopeResult { int state; int count; };`
    - `class JumpRopeCounter`.
  - Use a smoothed vertical center and a low/high/low phase cycle.
  - Keep thresholds scale-relative to body height.
  - Do not depend on OpenCV, Android, NCNN, or JNI.

- [ ] **Step 3: Verify green**
  - Run:
    ```powershell
    g++ -std=c++17 -Wall -Wextra -I ncnn-android-yolov8-master/app/src/main/jni ncnn-android-yolov8-master/tests/jumprope_counter_test.cpp ncnn-android-yolov8-master/app/src/main/jni/jumprope_counter.cpp -o C:\tmp\jumprope_counter_test.exe
    C:\tmp\jumprope_counter_test.exe
    ```
  - Expected result: all counter tests pass.

## Task 2: Replace Android DeskGuard Runtime With JumpRope

- [ ] **Step 1: Wire counter into Android build**
  - Add `jumprope_counter.cpp` to `add_library(...)` in `CMakeLists.txt`.

- [ ] **Step 2: Update JNI state**
  - In `yolov8ncnn.cpp`, remove DeskGuard constants, target labels, and risk streak.
  - Add a global `JumpRopeCounter`.
  - Add a helper that selects the best person from pose `objects` and converts keypoints to `JumpRopeFrame`.
  - Only update counting when `g_current_taskid == 3` and pose keypoints are reliable.
  - Reset the counter on model reload and camera close.
  - Expose:
    - `Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_getJumpRopeState`
    - `Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_getJumpRopeCount`

- [ ] **Step 3: Update camera overlay**
  - Draw a compact status label in the camera frame:
    - `JumpRope: switch to pose`
    - `JumpRope: no person`
    - `JumpRope: ready count=N`
    - `JumpRope: counting count=N`
  - Keep the existing FPS overlay.

## Task 3: Update Java UI And Behavior

- [ ] **Step 1: Update native Java API**
  - In `YOLOv8Ncnn.java`, replace DeskGuard constants/method with JumpRope constants and getters.

- [ ] **Step 2: Default to pose**
  - In `MainActivity.java`, initialize `current_task = 3`.
  - Set the task spinner selection to pose before attaching its listener or guard the initial callback so it does not reload back to COCO.

- [ ] **Step 3: Replace polling/status**
  - Rename DeskGuard poller fields/methods to JumpRope equivalents.
  - Poll `getJumpRopeState()` and `getJumpRopeCount()`.
  - Update the status bar text and color from JumpRope state.
  - Emit a short tone when the count increases.

- [ ] **Step 4: Update resources**
  - Change app name to `JumpRope-YOLO`.
  - Replace DeskGuard strings with JumpRope status strings.
  - Rename the status TextView id to `textJumpRopeStatus`.

## Task 4: Convert Model-Side Project To JumpRope Pose

- [ ] **Step 1: Write/update failing Python tests**
  - Move tests to `JumpRope-Model/training/jumprope_yolo/tests`.
  - Expect:
    - default class list is `("person",)`;
    - default model is `yolo11n-pose.pt`;
    - default dataset is `datasets/jumprope_pose.yaml`;
    - pose labels accept `5 + 17 * 3` values;
    - export pnnx command still includes two input shapes.
  - Run:
    ```powershell
    cd JumpRope-Model
    python -m unittest discover -s training\jumprope_yolo\tests -v
    ```
  - Expected result: tests fail until package/config are updated.

- [ ] **Step 2: Rename/update package**
  - Rename `DeskGuard-Model` to `JumpRope-Model`.
  - Rename `training/deskguard_yolo` to `training/jumprope_yolo`.
  - Update imports, module docstrings, command examples, run directories, and export directories.

- [ ] **Step 3: Update dataset validation**
  - Allow regular detection labels with 5 fields.
  - Allow pose labels with `5 + 17 * 3` fields.
  - Keep class id and normalized coordinate checks.

- [ ] **Step 4: Verify green**
  - Run the model-side unit tests again and require all tests to pass.

## Task 5: Update Documents And Diary

- [ ] **Step 1: Replace Android algorithm document**
  - Delete `docs/deskguard_algorithm.md`.
  - Add `docs/jumprope_algorithm.md` covering pose counting, states, thresholds, GPU mode, and evaluation metrics.

- [ ] **Step 2: Update model training document**
  - Explain first-version pretrained pose deployment.
  - Explain optional YOLO11 pose fine-tuning/export to NCNN.
  - Remove desktop/tabletop/danger-edge dataset language.

- [ ] **Step 3: Update development diary**
  - Add a new entry describing the teacher feedback and pivot.
  - Record the new commit hashes and verification commands after implementation commits.

## Task 6: Final Verification And Commits

- [ ] **Step 1: Run native counter tests**
  ```powershell
  g++ -std=c++17 -Wall -Wextra -I ncnn-android-yolov8-master/app/src/main/jni ncnn-android-yolov8-master/tests/jumprope_counter_test.cpp ncnn-android-yolov8-master/app/src/main/jni/jumprope_counter.cpp -o C:\tmp\jumprope_counter_test.exe
  C:\tmp\jumprope_counter_test.exe
  ```

- [ ] **Step 2: Run model tests**
  ```powershell
  cd JumpRope-Model
  python -m unittest discover -s training\jumprope_yolo\tests -v
  ```

- [ ] **Step 3: Build Android debug APK**
  ```powershell
  cd ncnn-android-yolov8-master
  .\gradlew.bat assembleDebug
  ```

- [ ] **Step 4: Search old user-facing theme**
  ```powershell
  rg -n "DeskGuard|deskguard|桌面|课桌|防掉落|danger_edge|cup|bottle|cell_phone|laptop|keyboard|mouse" -g "!ncnn-master/**" -g "!ncnn-20260526-android-vulkan/**" -g "!tools/**"
  ```
  - Remaining matches should be limited to historical design context or third-party class-name lists.

- [ ] **Step 5: Commit implementation**
  - Commit counter and Android changes.
  - Commit model/docs changes.
  - Commit diary verification update if separate.
