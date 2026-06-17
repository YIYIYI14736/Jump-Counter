import unittest
from pathlib import Path

from training.jumprope_yolo.export import ExportOptions, build_pnnx_command


class ExportCommandTests(unittest.TestCase):
    def test_build_pnnx_command_uses_two_input_shapes_for_dynamic_ncnn(self):
        options = ExportOptions(
            weights=Path("runs/detect/train/weights/best.pt"),
            imgsz=640,
            half=False,
            simplify=True,
            opset=12,
            device="cpu",
            output_dir=Path("exports"),
            format="ncnn",
            ncnn_backend="pnnx",
            dynamic=True,
        )

        command = build_pnnx_command(Path("exports/best.torchscript"), options)

        self.assertEqual(command[0], "pnnx")
        self.assertEqual(command[1], "exports/best.torchscript")
        self.assertIn("inputshape=[1,3,640,640]", command)
        self.assertIn("inputshape2=[1,3,320,320]", command)


if __name__ == "__main__":
    unittest.main()
