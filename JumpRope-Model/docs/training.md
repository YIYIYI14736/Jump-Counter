# JumpRope YOLO Pose Training

This folder contains the optional training and export workflow for the mobile JumpRope-YOLO demo.
The first version should run with pretrained YOLO pose weights, so custom training is only needed when
the demo needs better robustness for a specific camera angle, lighting setup, or rope-jumping style.

## Classes And Pose Labels

The dataset uses one class:

```text
0: person
```

Pose labels follow the Ultralytics COCO keypoint layout with 17 keypoints and three values per
keypoint:

```text
class_id x_center y_center width height kpt_x kpt_y visibility ...
```

Box and keypoint coordinates are normalized to `0..1`. Visibility uses the usual pose convention
where `0` is not labeled, `1` is labeled but not visible, and `2` is visible.

## Dataset Layout

Put images and labels under:

```text
datasets/jumprope_pose/
  images/train/
  labels/train/
  images/val/
  labels/val/
```

The dataset config is `datasets/jumprope_pose.yaml`.

## Train

Install dependencies:

```powershell
pip install -r requirements-training.txt
```

Fine-tune a lightweight pose baseline:

```powershell
python -m training.jumprope_yolo.train --data datasets/jumprope_pose.yaml --model yolo11n-pose.pt --epochs 60 --imgsz 640 --device 0
```

For the course demo, start from pretrained `yolo11n-pose.pt` and only fine-tune after collecting
representative jumping clips from the target phone/camera position.

## Export

Export a trained checkpoint to NCNN through Ultralytics:

```powershell
python -m training.jumprope_yolo.export --weights training/jumprope_yolo/runs/jumprope-yolo11n-pose/weights/best.pt --format ncnn --imgsz 640
```

The exported `.param` and `.bin` files can be copied into `app/src/main/assets/` after matching the
Android model names. The current Android demo can also use the bundled pretrained YOLOv8 pose NCNN
assets for the first runnable version.

## Evaluation

Record both model and app metrics:

- pose quality: visible hip/shoulder/ankle stability in jumping clips;
- counting quality: absolute count error per 30-second clip;
- latency: CPU FPS, GPU FPS, and average frame delay;
- robustness: missed-person frames, side-view jumps, weak light, and partial occlusion.
