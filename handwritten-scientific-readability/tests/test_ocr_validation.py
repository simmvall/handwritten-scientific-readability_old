import pathlib
import tempfile
import unittest

import pandas as pd

from src.ocr_validation import (
    correlation_frame,
    load_ground_truth_map,
    merge_metrics_and_validation,
)


class TestOcrValidation(unittest.TestCase):
    def test_load_ground_truth_map_uses_matching_filenames(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = pathlib.Path(tmp_dir)
            image_folder = base / "images"
            ground_truth_folder = base / "ground_truth"

            image_folder.mkdir()
            ground_truth_folder.mkdir()

            (image_folder / "neat.png").write_bytes(b"fake")
            (image_folder / "messy.png").write_bytes(b"fake")
            (ground_truth_folder / "neat.txt").write_text(
                "hello world",
                encoding="utf8",
            )
            (ground_truth_folder / "messy.txt").write_text(
                "messy text",
                encoding="utf8",
            )

            image_paths = sorted(image_folder.iterdir())
            ground_truth = load_ground_truth_map(ground_truth_folder, image_paths)

            self.assertEqual(ground_truth["neat.png"], "hello world")
            self.assertEqual(ground_truth["messy.png"], "messy text")

    def test_merge_metrics_and_validation_produces_numeric_correlation(self):
        metrics = pd.DataFrame(
            {
                "image": ["neat.png", "messy.png"],
                "baseline_consistency": [0.9, 0.2],
                "spatial_organization": [0.8, 0.3],
                "text_fragmentation": [0.85, 0.25],
                "combined_score": [0.84, 0.25],
            }
        )
        validation = pd.DataFrame(
            {
                "image": ["neat.png", "messy.png"],
                "ocr_confidence": [93.4, 62.3],
                "cer": [0.07, 0.41],
                "ocr_text": ["clean text", "rough text"],
            }
        )

        merged = merge_metrics_and_validation(metrics, validation)
        corr = correlation_frame(merged)

        self.assertEqual(set(merged["image"]), {"neat.png", "messy.png"})
        self.assertIn("cer", corr.columns)
        self.assertIn("ocr_confidence", corr.columns)


if __name__ == "__main__":
    unittest.main()