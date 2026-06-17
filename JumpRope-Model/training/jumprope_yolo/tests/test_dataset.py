import tempfile
import unittest
from pathlib import Path

from training.jumprope_yolo.config import DEFAULT_CLASSES
from training.jumprope_yolo.dataset import DatasetSpec, validate_yolo_dataset


class DatasetValidationTests(unittest.TestCase):
    def _make_minimal_dataset(self, root: Path, label_text: str = "0 0.5 0.5 0.2 0.3\n"):
        image_dir = root / "images" / "train"
        label_dir = root / "labels" / "train"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / "frame001.jpg").write_bytes(b"not-a-real-image-but-valid-path")
        (label_dir / "frame001.txt").write_text(label_text, encoding="utf-8")

    def test_validate_dataset_accepts_minimal_yolo_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_minimal_dataset(root)
            spec = DatasetSpec(path=root, train="images/train", val=None, names=DEFAULT_CLASSES)

            report = validate_yolo_dataset(spec)

            self.assertEqual(report.errors, [])
            self.assertEqual(report.image_count, 1)
            self.assertEqual(report.label_count, 1)
            self.assertEqual(report.class_counts["person"], 1)

    def test_validate_dataset_rejects_out_of_range_class_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_minimal_dataset(root, label_text="1 0.5 0.5 0.2 0.3\n")
            spec = DatasetSpec(path=root, train="images/train", val=None, names=DEFAULT_CLASSES)

            report = validate_yolo_dataset(spec)

            self.assertTrue(any("class id 1" in error for error in report.errors))

    def test_validate_dataset_rejects_bbox_values_outside_unit_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_minimal_dataset(root, label_text="0 1.2 0.5 0.2 0.3\n")
            spec = DatasetSpec(path=root, train="images/train", val=None, names=DEFAULT_CLASSES)

            report = validate_yolo_dataset(spec)

            self.assertTrue(any("outside 0..1" in error for error in report.errors))

    def test_validate_dataset_accepts_coco_pose_label_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keypoints = " ".join("0.5 0.5 2" for _ in range(17))
            self._make_minimal_dataset(root, label_text=f"0 0.5 0.5 0.2 0.3 {keypoints}\n")
            spec = DatasetSpec(path=root, train="images/train", val=None, names=DEFAULT_CLASSES)

            report = validate_yolo_dataset(spec)

            self.assertEqual(report.errors, [])
            self.assertEqual(report.class_counts["person"], 1)

    def test_validate_dataset_rejects_pose_keypoint_coordinates_outside_unit_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keypoints = ["0.5 0.5 2"] * 17
            keypoints[3] = "1.2 0.5 2"
            self._make_minimal_dataset(root, label_text=f"0 0.5 0.5 0.2 0.3 {' '.join(keypoints)}\n")
            spec = DatasetSpec(path=root, train="images/train", val=None, names=DEFAULT_CLASSES)

            report = validate_yolo_dataset(spec)

            self.assertTrue(any("keypoint coordinate outside 0..1" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
