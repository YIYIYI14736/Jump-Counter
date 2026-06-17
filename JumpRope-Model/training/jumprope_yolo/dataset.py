from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DETECTION_LABEL_VALUES = 5
COCO_POSE_KEYPOINTS = 17
POSE_LABEL_VALUES = DETECTION_LABEL_VALUES + COCO_POSE_KEYPOINTS * 3


@dataclass(frozen=True)
class DatasetSpec:
    path: Path
    train: str
    val: Optional[str]
    names: Tuple[str, ...]


@dataclass
class DatasetReport:
    errors: List[str] = field(default_factory=list)
    image_count: int = 0
    label_count: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)


def _iter_images(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def _label_dir_for_image_dir(image_dir: Path, root: Path) -> Path:
    relative = image_dir.relative_to(root)
    parts = list(relative.parts)
    if parts and parts[0] == "images":
        parts[0] = "labels"
        return root.joinpath(*parts)
    return root / "labels" / relative.name


def validate_yolo_dataset(spec: DatasetSpec) -> DatasetReport:
    report = DatasetReport(class_counts={name: 0 for name in spec.names})
    root = spec.path
    image_dir = root / spec.train
    label_dir = _label_dir_for_image_dir(image_dir, root)

    if not image_dir.exists():
        report.errors.append(f"Image directory does not exist: {image_dir}")
        return report
    if not label_dir.exists():
        report.errors.append(f"Label directory does not exist: {label_dir}")
        return report

    images = list(_iter_images(image_dir))
    report.image_count = len(images)

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            report.errors.append(f"Missing label for image: {image_path.name}")
            continue

        report.label_count += 1
        for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) not in (DETECTION_LABEL_VALUES, POSE_LABEL_VALUES):
                report.errors.append(
                    f"{label_path.name}:{line_number} expected 5 detection values or 56 pose values"
                )
                continue
            try:
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
            except ValueError:
                report.errors.append(f"{label_path.name}:{line_number} contains non-numeric YOLO values")
                continue

            if class_id < 0 or class_id >= len(spec.names):
                report.errors.append(f"{label_path.name}:{line_number} class id {class_id} outside 0..{len(spec.names) - 1}")
                continue
            bbox_values = values[:4]
            if any(value < 0.0 or value > 1.0 for value in bbox_values):
                report.errors.append(f"{label_path.name}:{line_number} bbox value outside 0..1")
                continue

            if len(parts) == POSE_LABEL_VALUES:
                keypoint_values = values[4:]
                invalid_keypoint = False
                for keypoint_index in range(0, len(keypoint_values), 3):
                    x = keypoint_values[keypoint_index]
                    y = keypoint_values[keypoint_index + 1]
                    visibility = keypoint_values[keypoint_index + 2]
                    if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
                        report.errors.append(f"{label_path.name}:{line_number} keypoint coordinate outside 0..1")
                        invalid_keypoint = True
                        break
                    if visibility < 0.0 or visibility > 2.0:
                        report.errors.append(f"{label_path.name}:{line_number} keypoint visibility outside 0..2")
                        invalid_keypoint = True
                        break
                if invalid_keypoint:
                    continue

            report.class_counts[spec.names[class_id]] += 1

    return report
