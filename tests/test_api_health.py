import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch
from fastapi.testclient import TestClient

from service.api.main import create_app
from src.pipeline.inference import load_inference_model
from tests.helpers import create_test_checkpoint, create_test_settings


class HealthApiTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = create_test_checkpoint(
            Path(self.temp_dir.name) / "cnn.pth"
        )
        self.settings = create_test_settings(self.checkpoint_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_health_reports_loaded_model(self):
        app = create_app(settings=self.settings)
        with TestClient(app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "model_loaded": True,
                "model_name": "cnn",
                "num_classes": 10,
                "device": str(
                    torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                ),
            },
        )

    def test_model_is_loaded_once_for_multiple_health_requests(self):
        loader = Mock(wraps=load_inference_model)
        app = create_app(settings=self.settings, model_loader=loader)
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/health").status_code, 200)

        loader.assert_called_once_with(self.checkpoint_path)

    def test_missing_checkpoint_fails_during_startup(self):
        missing_settings = create_test_settings(
            Path(self.temp_dir.name) / "missing.pth"
        )
        app = create_app(settings=missing_settings)
        with self.assertRaises(FileNotFoundError):
            with TestClient(app):
                pass


if __name__ == "__main__":
    unittest.main()
