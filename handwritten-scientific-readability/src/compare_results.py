from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

try:
    from src.ocr_validation import (
        build_structural_metrics_dataframe,
        build_validation_dataframe,
        correlation_frame,
        merge_metrics_and_validation,
        write_dataframe_csv,
    )
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.ocr_validation import (
        build_structural_metrics_dataframe,
        build_validation_dataframe,
        correlation_frame,
        merge_metrics_and_validation,
        write_dataframe_csv,
    )


def load_metrics_dataframe(
    metrics_csv: Union[str, os.PathLike],
    image_folder: Union[str, os.PathLike],
) -> pd.DataFrame:
    metrics_path = Path(metrics_csv)
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return build_structural_metrics_dataframe(image_folder=image_folder)


def load_validation_dataframe(
    validation_csv: Union[str, os.PathLike],
    image_folder: Union[str, os.PathLike],
    ground_truth_dir: Union[str, os.PathLike],
    tesseract_cmd: Optional[str] = None,
) -> pd.DataFrame:
    validation_path = Path(validation_csv)
    if validation_path.exists():
        return pd.read_csv(validation_path)
    return build_validation_dataframe(
        image_folder=image_folder,
        ground_truth_dir=ground_truth_dir,
        tesseract_cmd=tesseract_cmd,
    )


def main_function(
    metrics_csv: Union[str, os.PathLike] = "metrics.csv",
    validation_csv: Union[str, os.PathLike] = "tesseract_validation.csv",
    image_folder: Union[str, os.PathLike] = "notebookimages",
    ground_truth_dir: Union[str, os.PathLike] = "ground_truth",
    output_csv: Optional[Union[str, os.PathLike]] = None,
    summary_csv: Optional[Union[str, os.PathLike]] = "comparison_summary.csv",
    tesseract_cmd: Optional[str] = None,
) -> Dict[str, Any]:
    metrics = load_metrics_dataframe(metrics_csv, image_folder)
    validation = load_validation_dataframe(
        validation_csv=validation_csv,
        image_folder=image_folder,
        ground_truth_dir=ground_truth_dir,
        tesseract_cmd=tesseract_cmd,
    )
    merged = merge_metrics_and_validation(metrics, validation)
    pearson_correlations = merged.corr(method="pearson", numeric_only=True)
    spearman_correlations = merged.corr(method="spearman", numeric_only=True)
    summary = merged[["image", "combined_score", "ocr_confidence", "cer"]].copy()

    key_correlations = {
        "pearson": {
            "combined_score_vs_ocr_confidence": float(
                pearson_correlations.loc["combined_score", "ocr_confidence"]
            ),
            "combined_score_vs_cer": float(
                pearson_correlations.loc["combined_score", "cer"]
            ),
            "ocr_confidence_vs_cer": float(
                pearson_correlations.loc["ocr_confidence", "cer"]
            ),
        },
        "spearman": {
            "combined_score_vs_ocr_confidence": float(
                spearman_correlations.loc["combined_score", "ocr_confidence"]
            ),
            "combined_score_vs_cer": float(
                spearman_correlations.loc["combined_score", "cer"]
            ),
            "ocr_confidence_vs_cer": float(
                spearman_correlations.loc["ocr_confidence", "cer"]
            ),
        },
    }

    if output_csv is not None:
        write_dataframe_csv(merged, output_csv)
    if summary_csv is not None:
        write_dataframe_csv(summary, summary_csv)

    return {
        "status": "ok",
        "rows": int(len(merged)),
        "merged": merged.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "key_correlations": key_correlations,
        "pearson_matrix": pearson_correlations.to_dict(),
        "spearman_matrix": spearman_correlations.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare structural metrics with OCR validation results."
    )
    parser.add_argument("--metrics-csv", default="metrics.csv")
    parser.add_argument("--validation-csv", default="tesseract_validation.csv")
    parser.add_argument("--image-folder", default="notebookimages")
    parser.add_argument("--ground-truth-dir", default="ground_truth")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-csv", default="comparison_summary.csv")
    parser.add_argument("--tesseract-cmd", default=None)
    args = parser.parse_args()

    result = main_function(
        metrics_csv=args.metrics_csv,
        validation_csv=args.validation_csv,
        image_folder=args.image_folder,
        ground_truth_dir=args.ground_truth_dir,
        output_csv=args.output_csv,
        summary_csv=args.summary_csv,
        tesseract_cmd=args.tesseract_cmd,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
