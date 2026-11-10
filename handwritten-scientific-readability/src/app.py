from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Union

try:
    from src.document_readability import analyze_page
    from src.notebook_validation import compare_image_with_pdf, build_pdf_review_report
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.document_readability import analyze_page
    from src.notebook_validation import compare_image_with_pdf, build_pdf_review_report


def main_function(
    image_path: Optional[Union[str, bytes]] = None,
    pdf_path: Optional[Union[str, bytes]] = None,
    ground_truth_text: Optional[str] = None,
    ground_truth_file: Optional[str] = None,
    output_dir: Optional[str] = "preprocessed_outputs",
    review_dir: Optional[str] = "review_outputs",
    render_dir: Optional[str] = "review_outputs/rendered_pages",
    tesseract_cmd: Optional[str] = None,
) -> Dict[str, Any]:
    if image_path is None and pdf_path is None:
        return {
            "status": "no_image_provided",
            "message": "Provide an image path to analyze document readability.",
        }

    if pdf_path is not None and image_path is None:
        return {
            "status": "image_required_for_pdf_comparison",
            "message": "Provide a representative PNG image path together with the PDF path.",
        }

    if pdf_path is not None and image_path is not None:
        if ground_truth_text is None and ground_truth_file is None:
            return build_pdf_review_report(
                image_path=image_path,
                pdf_path=pdf_path,
                output_dir=output_dir,
                review_dir=review_dir,
                render_dir=render_dir,
                tesseract_cmd=tesseract_cmd,
            )

        reference_text = ground_truth_text
        if reference_text is None and ground_truth_file is not None:
            with open(ground_truth_file, encoding="utf8") as handle:
                reference_text = handle.read()

        return compare_image_with_pdf(
            image_path=image_path,
            pdf_path=pdf_path,
            ground_truth_text=reference_text,
            output_dir=output_dir,
            tesseract_cmd=tesseract_cmd,
            render_dir=render_dir,
        )

    if image_path is not None:
        return analyze_page(image_path, ground_truth_text, output_dir=output_dir)

    return analyze_page(image_path, ground_truth_text, output_dir=output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze handwritten document readability.")
    parser.add_argument("image_path", nargs="?", help="Path to the handwritten page image")
    parser.add_argument("--pdf-path", dest="pdf_path", default=None, help="Path to the notebook PDF")
    parser.add_argument("--ground-truth", dest="ground_truth_text", default=None)
    parser.add_argument("--ground-truth-file", dest="ground_truth_file", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default="preprocessed_outputs")
    parser.add_argument("--review-dir", dest="review_dir", default="review_outputs")
    parser.add_argument("--render-dir", dest="render_dir", default="review_outputs/rendered_pages")
    parser.add_argument("--tesseract-cmd", dest="tesseract_cmd", default=None)
    args = parser.parse_args()

    result = main_function(
        image_path=args.image_path,
        pdf_path=args.pdf_path,
        ground_truth_text=args.ground_truth_text,
        ground_truth_file=args.ground_truth_file,
        output_dir=args.output_dir,
        review_dir=args.review_dir,
        render_dir=args.render_dir,
        tesseract_cmd=args.tesseract_cmd,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())