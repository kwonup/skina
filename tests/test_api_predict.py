from io import BytesIO
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from service.api.main import create_app
from tests.helpers import create_test_checkpoint, create_test_settings


def make_image_bytes(image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (48, 48), color=(190, 130, 110)).save(
        buffer, format=image_format
    )
    return buffer.getvalue()


class PredictApiTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        checkpoint_path = create_test_checkpoint(
            Path(self.temp_dir.name) / "cnn.pth"
        )
        self.settings = create_test_settings(checkpoint_path)
        self.app = create_app(settings=self.settings)
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temp_dir.cleanup()

    def test_png_returns_top_three_predictions(self):
        response = self.client.post(
            "/predict",
            files={"image": ("lesion.png", make_image_bytes(), "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["top3"]), 3)
        self.assertEqual(body["prediction"]["class"], body["top3"][0]["class"])
        self.assertEqual(
            body["prediction"]["confidence"], body["top3"][0]["probability"]
        )
        self.assertGreaterEqual(body["inference_time_ms"], 0)
        probabilities = [item["probability"] for item in body["top3"]]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))

    def test_jpg_and_jpeg_extensions_are_accepted(self):
        for filename in ("lesion.jpg", "lesion.jpeg"):
            with self.subTest(filename=filename):
                response = self.client.post(
                    "/predict",
                    files={
                        "image": (filename, make_image_bytes("JPEG"), "image/jpeg")
                    },
                )
                self.assertEqual(response.status_code, 200)

    def test_non_image_file_is_rejected(self):
        response = self.client.post(
            "/predict",
            files={"image": ("notes.txt", b"not an image", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)

    def test_spoofed_and_corrupted_image_is_rejected(self):
        response = self.client.post(
            "/predict",
            files={"image": ("fake.png", b"not an image", "image/png")},
        )
        self.assertEqual(response.status_code, 400)

    def test_empty_file_is_rejected(self):
        response = self.client.post(
            "/predict",
            files={"image": ("empty.png", b"", "image/png")},
        )
        self.assertEqual(response.status_code, 400)

    def test_oversized_file_is_rejected(self):
        small_settings = create_test_settings(self.settings.model_path)
        object.__setattr__(small_settings, "max_upload_size_bytes", 16)
        app = create_app(settings=small_settings)
        with TestClient(app) as client:
            response = client.post(
                "/predict",
                files={"image": ("large.png", make_image_bytes(), "image/png")},
            )
        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
