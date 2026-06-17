# 基于深度学习的跳绳计数系统

> 深度学习应用实践课程设计项目 — 林泰康 & 洪天鑫

## 项目概述

本系统实现了一套完整的跳绳自动计数方案，从模型训练到移动端部署，包含：

- **YOLO11 姿态估计模型**：检测人体关键点（17点）
- **MLP 跳跃分类器**：基于关键点时序特征判断跳跃有效性
- **多人档案追踪**：通过体态特征+颜色直方图实现人物重识别
- **Android 实时推理**：NCNN + Vulkan 加速，手机端 20-30 FPS

## 目录结构

```
.
├── JumpRope-Model/              # Python 训练/导出工具链
│   ├── training/
│   │   ├── jumprope_yolo/       # YOLO11 姿态模型训练 & 导出
│   │   └── jumprope_classifier/ # MLP 跳跃分类器训练 & 导出
│   ├── datasets/                # 训练数据集
│   ├── exports/                 # 导出的 NCNN 模型
│   ├── yolo11n-pose_ncnn_model/ # NCNN 模型文件 (.param + .bin)
│   ├── eval_video_simple.py     # 视频评估工具
│   ├── test_jumprope.py         # 测试脚本
│   └── requirements-training.txt
│
├── ncnn-android-yolov8-master/  # Android 应用
│   ├── app/src/main/
│   │   ├── java/.../            # Java UI + JNI 桥接
│   │   ├── jni/                 # C++ 推理 + 计数逻辑
│   │   │   ├── yolov8ncnn.cpp        # JNI 主入口 & 帧循环
│   │   │   ├── yolo11_pose.cpp       # YOLO11 姿态解码
│   │   │   ├── jumprope_counter.cpp  # 跳跃计数状态机
│   │   │   ├── jumprope_profile.cpp  # 多人档案追踪
│   │   │   └── CMakeLists.txt
│   │   ├── assets/              # 模型资源
│   │   └── res/                 # UI 布局 & 资源
│   └── build.gradle
│
├── 课程文档/                    # 课程相关文档
│   ├── 林泰康-答辩PPT-v2.pptx
│   ├── 基于深度学习的跳绳计数系统-答辩报告.pdf
│   ├── 公式符号与中文详解.pdf
│   ├── 林泰康-深度学习应用实践课程报告-v8.docx
│   ├── Android端人体跳跃计数软件.pptx
│   └── ...
│
├── AGENTS.md                    # 开发指南（WARP/CI 用）
└── .gitignore
```

## 核心算法

### 1. 跳跃计数状态机

状态转移：`INACTIVE → NO_PERSON → READY → COUNTING`

- **起跳判定**：冷却期结束 + 垂直速度 < -0.25 px/帧 + 位移超过最小振幅
- **回落判定**：腾空 ≥ 2 帧 + 速度转正 + 回到基线附近
- **蹲起过滤**：踝关节离地检测 + 振幅上限门控

### 2. MLP 跳跃分类器

网络结构：`12 → 16 → 8 → 1`（353 参数），Sigmoid 输出

- 输入特征：振幅、速度、帧数、踝关节提升等 12 维
- 输出：跳跃有效性概率，阈值 0.5
- 门控计数：`C_display = T_prev + (C_raw - N_rejected)`

### 3. 多人档案追踪

- **体态特征**：5 维身体比例向量（L2 距离）
- **颜色特征**：16H × 8S × 4V = 512 维 HSV 直方图（卡方距离）
- **融合权重**：体态 30% + 颜色 70%
- **匹配阈值**：0.25，EMA 更新 α = 0.05
- 最多支持 4 人同时追踪（P1-P4）

### 4. YOLO → NCNN 部署流水线

```
PyTorch → ONNX → NCNN (.param + .bin) → Android Assets
```

Vulkan GPU 加速推理，640×640 输入，实时 20-30 FPS。

## 构建说明

### Python 训练

```bash
cd JumpRope-Model
pip install -r requirements-training.txt
python -m training.jumprope_yolo.train --data datasets/jumprope_pose.yaml --model yolo11n-pose.pt --epochs 60 --imgsz 640 --device 0
python -m training.jumprope_yolo.export --weights training/jumprope_yolo/runs/jumprope-yolo11n-pose/weights/best.pt --format ncnn --imgsz 640
```

### Android 构建

1. 按照 `ncnn-android-yolov8-master/README.md` 配置 NCNN 和 OpenCV 本地依赖
2. 将导出的 NCNN 模型放入 `app/src/main/assets/`
3. 构建：`.\gradlew.bat :app:assembleDebug`

## 项目状态

**已完成归档** — 2026-06-17

本项目为课程设计作品，已通过答辩。代码和文档已整理归档，不再积极开发。
