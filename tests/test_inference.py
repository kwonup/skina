import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.classes import CLASS_NAMES
from src.models import create_model
from src.pipeline.inference import load_inference_model


class InferenceTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = Path(self.temp_dir.name) / "cnn.pth"
        model = create_model("cnn", len(CLASS_NAMES), pretrained=False)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": "cnn",
                "class_names": list(CLASS_NAMES),
                "image_size": 32,
            },
            self.checkpoint_path,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_model_loads_and_predicts_top_three(self):
        inference_model = load_inference_model(
            self.checkpoint_path, device=torch.device("cpu")
        )
        output = inference_model.predict(Image.new("RGB", (48, 48)), top_k=3)

        self.assertEqual(inference_model.class_names, CLASS_NAMES)
        self.assertEqual(len(output.predictions), 3)
        self.assertEqual([item.rank for item in output.predictions], [1, 2, 3])
        probabilities = [item.probability for item in output.predictions]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
        self.assertTrue(all(0 <= value <= 1 for value in probabilities))

    def test_legacy_class_mapping_is_rejected(self):
        checkpoint = torch.load(self.checkpoint_path, weights_only=True)
        checkpoint["class_names"] = [*CLASS_NAMES, "legacy_class"]
        torch.save(checkpoint, self.checkpoint_path)

        with self.assertRaisesRegex(ValueError, "클래스 순서"):
            load_inference_model(self.checkpoint_path, device=torch.device("cpu"))

    def test_invalid_top_k_is_rejected(self):
        inference_model = load_inference_model(
            self.checkpoint_path, device=torch.device("cpu")
        )
        with self.assertRaisesRegex(ValueError, "top_k"):
            inference_model.predict(Image.new("RGB", (32, 32)), top_k=0)


if __name__ == "__main__":
    unittest.main()
