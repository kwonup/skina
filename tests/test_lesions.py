import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from service.api.lesions import MEDICAL_DISCLAIMER, load_lesion_catalog
from service.api.main import create_app
from src.classes import CLASS_NAMES
from tests.helpers import create_test_checkpoint, create_test_settings


class LesionsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        checkpoint_path = create_test_checkpoint(
            Path(self.temp_dir.name) / "cnn.pth"
        )
        self.app = create_app(settings=create_test_settings(checkpoint_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_catalog_contains_exact_class_keys_in_checkpoint_order(self):
        catalog = load_lesion_catalog(CLASS_NAMES)
        self.assertEqual(tuple(catalog), CLASS_NAMES)
        self.assertTrue(all(not item.description for item in catalog.values()))

    def test_missing_catalog_key_is_rejected(self):
        catalog = {
            name: {
                "name_ko": "",
                "name_en": "",
                "category": "",
                "description": "",
                "features": [],
                "precautions": [],
            }
            for name in CLASS_NAMES[:-1]
        }
        path = Path(self.temp_dir.name) / "incomplete.json"
        path.write_text(json.dumps(catalog), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "누락"):
            load_lesion_catalog(CLASS_NAMES, path)

    def test_lesions_endpoint_returns_ten_items_in_order(self):
        with TestClient(self.app) as client:
            response = client.get("/lesions")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            [item["class"] for item in body["lesions"]], list(CLASS_NAMES)
        )
        self.assertEqual(body["disclaimer"], MEDICAL_DISCLAIMER)

    def test_prediction_is_enriched_from_same_catalog(self):
        from tests.test_api_predict import make_image_bytes

        with TestClient(self.app) as client:
            response = client.post(
                "/predict",
                files={"image": ("lesion.png", make_image_bytes(), "image/png")},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["prediction"]["class"], body["information"]["class"]
        )
        self.assertEqual(body["disclaimer"], MEDICAL_DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
