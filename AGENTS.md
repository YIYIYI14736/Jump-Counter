# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository shape
- This repository contains two related codebases:
  - `ncnn-android-yolov8-master/`: Android app (Java + JNI/C++) for on-device camera inference with NCNN.
  - `JumpRope-Model/`: Python training/export utilities for JumpRope pose models (Ultralytics YOLO -> NCNN export path).
- The Android app can run with bundled NCNN assets; Python training/export is the path for updating model weights.

## Common commands
### Python training/export (`JumpRope-Model`)
- Install dependencies:
  - `pip install -r JumpRope-Model/requirements-training.txt`
- Train JumpRope pose model:
  - `python -m training.jumprope_yolo.train --data datasets/jumprope_pose.yaml --model yolo11n-pose.pt --epochs 60 --imgsz 640 --device 0`
  - Run from `JumpRope-Model/`.
- Export trained weights to NCNN (Ultralytics backend):
  - `python -m training.jumprope_yolo.export --weights training/jumprope_yolo/runs/jumprope-yolo11n-pose/weights/best.pt --format ncnn --imgsz 640`
  - Run from `JumpRope-Model/`.
- Run all Python tests:
  - `python -m unittest discover -s training/jumprope_yolo/tests -p "test_*.py"`
  - Run from `JumpRope-Model/`.
- Run one test module:
  - `python -m unittest training.jumprope_yolo.tests.test_dataset`
  - Run from `JumpRope-Model/`.
- Run one specific test:
  - `python -m unittest training.jumprope_yolo.tests.test_dataset.DatasetValidationTests.test_validate_dataset_accepts_minimal_yolo_layout`
  - Run from `JumpRope-Model/`.

### Android app (`ncnn-android-yolov8-master`)
- Build debug APK:
  - `.\gradlew.bat :app:assembleDebug`
  - Run from `ncnn-android-yolov8-master/` on Windows.
- Run Android lint:
  - `.\gradlew.bat :app:lint`
  - Run from `ncnn-android-yolov8-master/`.
- Clean build artifacts:
  - `.\gradlew.bat clean`
  - Run from `ncnn-android-yolov8-master/`.

## Native dependency setup required before Android build
- Follow `ncnn-android-yolov8-master/README.md` setup:
  - Place NCNN Android Vulkan package under `app/src/main/jni/` and update `ncnn_DIR` in `app/src/main/jni/CMakeLists.txt`.
  - Place OpenCV mobile package under `app/src/main/jni/` and update `OpenCV_DIR` in `app/src/main/jni/CMakeLists.txt`.
  - Optionally add Turnip Vulkan driver `libvulkan_freedreno.so` under `app/src/main/jniLibs/arm64-v8a`.

## Architecture overview
### End-to-end runtime flow (Android app)
- Java UI entrypoint is `ncnn-android-yolov8-master/app/src/main/java/com/tencent/yolov8ncnn/MainActivity.java`.
  - Owns camera lifecycle, model/task selectors, and jump-rope status UI.
  - Polls jump-rope state/count from JNI (`getJumpRopeState`, `getJumpRopeCount`) on a timer.
- Java-to-native bridge is `ncnn-android-yolov8-master/app/src/main/java/com/tencent/yolov8ncnn/YOLOv8Ncnn.java`.
  - Exposes `loadModel`, camera open/close, output surface binding, and jump-rope state getters.
- JNI orchestration lives in `ncnn-android-yolov8-master/app/src/main/jni/yolov8ncnn.cpp`.
  - Selects detector implementation by task/model id (`YOLOv8_*` or `YOLOv11_pose`).
  - Maintains global detector/camera state and rendering loop.
  - Runs detect -> optional single-person tracking -> jump-rope update -> draw overlays for each frame.
- Task-specific inference implementations are in `ncnn-android-yolov8-master/app/src/main/jni/yolov8_*.cpp` and `yolo11_pose.cpp`.
  - `yolo11_pose.cpp` handles YOLO11 pose output shape/decoding differences from YOLOv8 pose.

## Jump-rope counting logic
- Core state machine is in `ncnn-android-yolov8-master/app/src/main/jni/jumprope_counter.cpp` with API in `jumprope_counter.h`.
- Input is reduced pose information (`JumpRopeFrame`) derived from keypoints + bbox height.
- Counter behavior is smoothing + airborne detection + cooldown:
  - Handles missing-person grace frames.
  - Switches among inactive/no-person/ready/counting states.
  - Uses vertical trajectory amplitude relative to body height to count valid jumps.
- JNI keeps only one primary pose person in pose mode before feeding the counter, so UI and count focus on a single target.

## Model assets and naming contract
- Android model loading is name-based from app assets in `ncnn-android-yolov8-master/app/src/main/assets/`.
- `loadModel` composes filenames from task suffix + model id; YOLOv8 and YOLO11 prefixes are both supported.
- When introducing new exported models, ensure filenames and expected target size mapping remain aligned with `yolov8ncnn.cpp`.

## Python training/export responsibilities
- `JumpRope-Model/training/jumprope_yolo/train.py` parses CLI options and forwards Ultralytics training overrides from `config.py`.
- `JumpRope-Model/training/jumprope_yolo/export.py` exports checkpoints via Ultralytics and can emit `pnnx` command lines for dynamic-shape NCNN workflows.
- `JumpRope-Model/training/jumprope_yolo/dataset.py` validates YOLO detection/pose label format and ranges for dataset sanity checks.
- `JumpRope-Model/training/jumprope_yolo/tests/` is unittest-based coverage for config, dataset validation, and export command generation.
