import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from src.notebook_validation import ocr_pdf_document
from src.app import main_function


class TestPdfValidation(unittest.TestCase):
    def test_ocr_pdf_document_renders_pages_and_ocr_text(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            pdf_path = tmp_file.name

        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "hello world")
        document.save(pdf_path)
        document.close()

        with patch("src.notebook_validation.pytesseract.image_to_string", return_value="hello world"):
            result = ocr_pdf_document(pdf_path)

        self.assertEqual(result["page_count"], 1)
        self.assertEqual(result["ocr_text"], "hello world")
        self.assertEqual(result["page_char_counts"], [11])
        self.assertEqual(result["total_char_count"], 11)
        self.assertEqual(len(result["rendered_paths"]), 1)
        self.assertTrue(result["rendered_paths"][0].endswith("page_0001.png"))

    def test_main_function_returns_pdf_comparison_payload(self):
        with patch("src.app.compare_image_with_pdf") as compare_mock:
            compare_mock.return_value = {
                "combined_score": 0.9,
                "pdf_cer": 0.2,
                "pdf_page_count": 3,
                "pdf_page_char_counts": [10, 20, 30],
            }

            result = main_function(
                image_path="notebookimages/neat.png",
                pdf_path="pdfs/notebook.pdf",
                ground_truth_text="hello world",
            )

        self.assertEqual(result["pdf_cer"], 0.2)
        self.assertEqual(result["pdf_page_count"], 3)
        self.assertEqual(result["pdf_page_char_counts"], [10, 20, 30])
        compare_mock.assert_called_once()

    def test_main_function_without_ground_truth_uses_review_mode(self):
        with patch("src.app.build_pdf_review_report") as review_mock:
            review_mock.return_value = {
                "combined_score": 0.9,
                "pdf_page_count": 3,
                "pdf_ocr_text": "review text",
                "pdf_page_char_counts": [5, 4, 6],
                "pdf_total_char_count": 15,
                "review_dir": "review_outputs",
            }

            result = main_function(
                image_path="notebookimages/neat.png",
                pdf_path="pdfs/notebook.pdf",
            )

        self.assertEqual(result["pdf_page_count"], 3)
        self.assertEqual(result["pdf_ocr_text"], "review text")
        self.assertEqual(result["pdf_page_char_counts"], [5, 4, 6])
        self.assertEqual(result["pdf_total_char_count"], 15)
        self.assertEqual(result["review_dir"], "review_outputs")
        review_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()