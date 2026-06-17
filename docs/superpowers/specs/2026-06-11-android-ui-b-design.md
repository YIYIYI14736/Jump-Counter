# Android UI B Design

## Goal

Update the Android demo UI to the selected full-screen camera direction: maximize the camera preview and place camera controls, model selectors, and jump-rope status as overlays.

## Design

- Keep the existing native Android implementation. Do not add Material, AppCompat, or new runtime dependencies.
- Replace the current vertical `LinearLayout` with a `FrameLayout` where `SurfaceView` fills the screen.
- Add a top overlay with the app title and compact camera switch button.
- Add a compact bottom overlay containing the existing task, model, and runtime selectors.
- Keep `textJumpRopeStatus` as the Java-controlled status view, but render it as a floating translucent pill near the top of the preview.
- Keep all existing control IDs so `MainActivity` can continue binding the same views.

## Constraints

- Camera preview and JNI model logic are out of scope.
- The layout must compile with minSdk 24 and existing Android support-v4 only.
- Dynamic jump-rope state colors remain in `MainActivity`.
