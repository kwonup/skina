import unittest

from src.classes import CLASS_NAMES, NUM_CLASSES, validate_class_names
from src.data.dataset import create_dataloaders


class ClassNamesTest(unittest.TestCase):
    def test_configured_class_names_are_the_final_ten_classes(self):
        self.assertEqual(NUM_CLASSES, 10)
        self.assertEqual(len(CLASS_NAMES), len(set(CLASS_NAMES)))

    def test_processed_dataset_matches_configured_order(self):
        *_, dataset_class_names = create_dataloaders(batch_size=1)
        self.assertEqual(tuple(dataset_class_names), CLASS_NAMES)

    def test_mismatched_order_is_rejected(self):
        reversed_names = tuple(reversed(CLASS_NAMES))
        with self.assertRaisesRegex(ValueError, "클래스 순서"):
            validate_class_names(reversed_names, source="test")


if __name__ == "__main__":
    unittest.main()
