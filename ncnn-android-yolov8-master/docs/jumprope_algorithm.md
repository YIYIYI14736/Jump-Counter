# JumpRope-YOLO 姿态计数算法

## 目标

JumpRope-YOLO 的第一版目标是在 Android 端直接使用 YOLO pose 结果完成跳绳计数。应用不依赖云端服务，也不需要先训练新检测类别；手机摄像头采集画面后，由 NCNN 推理输出人体关键点，再用轻量时序状态机统计一次完整的起跳和落地周期。

输入是 pose 检测结果：

```text
person bbox + 17 keypoints(x, y, confidence)
```

输出是跳绳状态：

```text
0 = inactive
1 = no_person
2 = ready
3 = counting
```

## 人体选择与关键点转换

同一帧可能检测到多个人。第一版选择面积和置信度综合得分最高的人：

```text
score = bbox_area * max(confidence, 0.01)
```

计数不直接使用全身所有关键点，而是优先使用左右髋部的平均 y 坐标作为身体中心。如果髋部置信度不足，则回退到肩部与踝部的中点：

```text
center_y = avg(left_hip.y, right_hip.y)
fallback_center_y = (avg_shoulders_y + avg_ankles_y) / 2
body_height = bbox.height or avg_ankles_y - avg_shoulders_y
```

关键点置信度阈值当前为 `0.25`。低于阈值的关键点不会参与中心点估计，避免遮挡或误检导致计数抖动。

## 状态机

状态机只依赖 `has_person`、`center_y` 和 `body_height`：

```text
NO_PERSON:
  没有可靠人体。

READY:
  已检测到稳定人体，但还没有完成一次跳跃。

COUNTING:
  已完成至少一次低 -> 高 -> 低周期，持续显示累计次数。
```

`center_y` 使用指数平滑，降低关键点抖动：

```text
smoothed_y = old_y * 0.55 + current_y * 0.45
```

起跳判断采用相对身体高度的阈值：

```text
min_amplitude = max(8 px, body_height * 0.035)
```

当平滑后的身体中心向上移动超过阈值，进入 airborne 阶段；当身体中心回落到基线附近，计数加一：

```text
if !airborne and smoothed_y <= baseline_y - min_amplitude:
    airborne = true

if airborne and smoothed_y >= baseline_y - min_amplitude:
    count += 1
    airborne = false
```

计数后设置短冷却帧，避免同一次落地被连续计数。短时间人体丢失会保留当前次数，用于处理跳跃时遮挡、快速运动模糊或单帧 pose 失败。

## Android 端反馈

应用默认加载 `pose` 任务。摄像头画面左上角绘制紧凑状态：

```text
JumpRope: switch to pose
JumpRope: no person
JumpRope: ready count=N
JumpRope: counting count=N
```

Java 层每 150 ms 轮询 JNI 状态和次数。当次数增加时，手机触发一次短震动和提示音。界面状态条同步显示当前状态和累计次数。

## GPU 与部署

CPU、GPU、turnip 三种推理模式沿用原 NCNN 示例的加载流程。切换模型、切换任务或关闭摄像头时会重置跳绳计数状态，避免旧任务的状态残留到新的推理任务。

第一版可直接使用工程内的 YOLOv8 pose NCNN 资产。后续如果要提升特定场景效果，可以用 `JumpRope-Model` 中的 pose 微调流程训练 `person` 姿态模型，再导出到 NCNN。

## 实验指标

课程报告建议记录：

- 计数误差：每段 30 秒或 100 次跳绳视频的绝对误差。
- 漏检情况：连续跳跃中 `NO_PERSON` 状态出现次数。
- 端侧性能：CPU FPS、GPU FPS、平均单帧延迟。
- 响应延迟：完成一次跳跃到界面次数更新、震动/提示音触发的时间。
- 鲁棒性：正面、侧面、弱光、遮挡、不同身高和不同跳跃幅度。
