# Android UI B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the selected full-screen camera overlay UI for the Android app.

**Architecture:** Use the existing native Android layout and activity binding. `main.xml` owns the screen structure, XML drawables provide static panel styling, and `MainActivity` continues to update jump-rope status text and colors.

**Tech Stack:** Android XML layouts, XML shape drawables, Java `Activity`, Gradle Android build.

---

### Task 1: Add Overlay Styling Resources

**Files:**
- Create: `ncnn-android-yolov8-master/app/src/main/res/drawable/overlay_panel_background.xml`
- Create: `ncnn-android-yolov8-master/app/src/main/res/drawable/overlay_chip_background.xml`
- Create: `ncnn-android-yolov8-master/app/src/main/res/drawable/spinner_overlay_background.xml`
- Create: `ncnn-android-yolov8-master/app/src/main/res/drawable/status_pill_background.xml`

- [ ] Add rounded translucent shapes for overlay panels, buttons, spinners, and the status pill.
- [ ] Use only platform-supported shape drawable attributes.

### Task 2: Replace Layout With Full-Screen Overlay

**Files:**
- Modify: `ncnn-android-yolov8-master/app/src/main/res/layout/main.xml`

- [ ] Change the root view to `FrameLayout`.
- [ ] Make `@id/cameraview` fill the parent.
- [ ] Add top overlay containing title and `@id/buttonSwitchCamera`.
- [ ] Add `@id/textJumpRopeStatus` as a floating pill.
- [ ] Add bottom overlay containing existing `@id/spinnerTask`, `@id/spinnerModel`, and `@id/spinnerCPUGPU`.

### Task 3: Polish Activity Status Styling

**Files:**
- Modify: `ncnn-android-yolov8-master/app/src/main/java/com/tencent/yolov8ncnn/MainActivity.java`

- [ ] Hide the default title bar before `setContentView`.
- [ ] Set status/navigation bars to black where supported.
- [ ] Replace direct `setBackgroundColor` calls with a rounded translucent drawable helper.

### Task 4: Verify Build

**Files:**
- No source edits.

- [ ] Run `cmd /c .\gradlew.bat assembleDebug` from `ncnn-android-yolov8-master`.
- [ ] Confirm the build exits with code 0.
