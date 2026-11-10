from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Union

try:
    from src.ocr_validation import build_validation_dataframe, write_dataframe_csv
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.ocr_validation import build_validation_dataframe, write_dataframe_csv


def main_function(
    image_folder: Union[str, os.PathLike] = "notebookimages",
    ground_truth_dir: Union[str, os.PathLike] = "ground_truth",
    output_csv: Union[str, os.PathLike] = "tesseract_validation.csv",
    tesseract_cmd: Optional[str] = None,
) -> Dict[str, Any]:
    dataframe = build_validation_dataframe(
        image_folder=image_folder,
        ground_truth_dir=ground_truth_dir,
        tesseract_cmd=tesseract_cmd,
    )
    write_dataframe_csv(dataframe, output_csv)
    return {
        "status": "ok",
        "rows": int(len(dataframe)),
        "output_csv": str(output_csv),
        "data": dataframe.to_dict(orient="records"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Tesseract OCR against handwritten ground truth."
    )
    parser.add_argument("--image-folder", default="notebookimages")
    parser.add_argument("--ground-truth-dir", default="ground_truth")
    parser.add_argument("--output-csv", default="tesseract_validation.csv")
    parser.add_argument("--tesseract-cmd", default=None)
    args = parser.parse_args()

    result = main_function(
        image_folder=args.image_folder,
        ground_truth_dir=args.ground_truth_dir,
        output_csv=args.output_csv,
        tesseract_cmd=args.tesseract_cmd,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
