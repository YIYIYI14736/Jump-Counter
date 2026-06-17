import unittest
from pathlib import Path

from training.jumprope_yolo.config import (
    DEFAULT_CLASSES,
    JUMPROPE_CLASSES,
    TrainOptions,
    build_train_overrides,
    parse_class_names,
)


class ConfigTests(unittest.TestCase):
    def test_default_classes_are_person_only_for_pose_counting(self):
        self.assertEqual(JUMPROPE_CLASSES, ("person",))
        self.assertEqual(DEFAULT_CLASSES, JUMPROPE_CLASSES)

    def test_parse_class_names_accepts_csv_and_trims_whitespace(self):
        names = parse_class_names("person, coach")
        self.assertEqual(names, ("person", "coach"))

    def test_parse_class_names_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, "Duplicate class name"):
            parse_class_names("person,coach,person")

    def test_small_dataset_overrides_keep_training_lightweight(self):
        options = TrainOptions(
            data=Path("datasets/jumprope_pose.yaml"),
            model="yolo11n-pose.pt",
            epochs=60,
            imgsz=640,
            batch=-1,
            project=Path("training/jumprope_yolo/runs"),
            name="jumprope-yolo11n-pose",
            device=None,
            workers=2,
            seed=7,
            patience=20,
            cache=False,
            amp=True,
            small_data=True,
            freeze=0,
        )

        overrides = build_train_overrides(options)

        self.assertEqual(overrides["data"], "datasets/jumprope_pose.yaml")
        self.assertEqual(overrides["model"], "yolo11n-pose.pt")
        self.assertEqual(overrides["batch"], -1)
        self.assertEqual(overrides["project"], "training/jumprope_yolo/runs")
        self.assertEqual(overrides["name"], "jumprope-yolo11n-pose")
        self.assertEqual(overrides["close_mosaic"], 10)
        self.assertEqual(overrides["mixup"], 0.0)
        self.assertEqual(overrides["copy_paste"], 0.0)
        self.assertTrue(overrides["cos_lr"])


if __name__ == "__main__":
    unittest.main()
