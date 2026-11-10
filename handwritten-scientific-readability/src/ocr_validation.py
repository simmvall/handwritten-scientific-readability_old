from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import pandas as pd

try:
    from src.document_readability import compute_metrics, validate_against_ground_truth
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.document_readability import compute_metrics, validate_against_ground_truth


DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEFAULT_IMAGE_SUFFIXES = (".png",)


def configure_tesseract(tesseract_cmd: Optional[str] = None) -> None:
    try:
        import pytesseract
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency validation
        raise RuntimeError(
            "pytesseract is required for OCR validation"
        ) from exc

    pytesseract.pytesseract.tesseract_cmd = (
        tesseract_cmd
        or os.environ.get("TESSERACT_CMD")
        or DEFAULT_TESSERACT_CMD
    )


def discover_image_files(
    image_folder: Union[str, os.PathLike],
    suffixes: Sequence[str] = DEFAULT_IMAGE_SUFFIXES,
) -> List[Path]:
    folder = Path(image_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")

    normalized_suffixes = tuple(s.lower() for s in suffixes)
    image_paths = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in normalized_suffixes
    ]

    if not image_paths:
        raise FileNotFoundError(
            f"No image files with suffixes {normalized_suffixes} were found in {folder}"
        )

    return image_paths


def load_ground_truth_map(
    ground_truth_dir: Union[str, os.PathLike],
    image_paths: Iterable[Path],
) -> Mapping[str, str]:
    folder = Path(ground_truth_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Ground truth folder not found: {folder}")

    ground_truth: Dict[str, str] = {}
    missing: List[Path] = []

    for image_path in image_paths:
        text_path = folder / f"{image_path.stem}.txt"
        if not text_path.exists():
            missing.append(text_path)
            continue
        ground_truth[image_path.name] = text_path.read_text(encoding="utf8").strip()

    if missing:
        missing_display = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing ground truth files for validation: {missing_display}"
        )

    return ground_truth


def validate_image(
    image_path: Union[str, os.PathLike],
    ground_truth_text: str,
) -> Dict[str, Any]:
    payload = validate_against_ground_truth(image_path, ground_truth_text)
    payload = {
        "image": Path(image_path).name,
        "ground_truth_text": ground_truth_text,
        **payload,
    }
    return payload


def build_validation_dataframe(
    image_folder: Union[str, os.PathLike] = "notebookimages",
    ground_truth_dir: Union[str, os.PathLike] = "ground_truth",
    tesseract_cmd: Optional[str] = None,
) -> pd.DataFrame:
    configure_tesseract(tesseract_cmd)
    image_paths = discover_image_files(image_folder)
    ground_truth = load_ground_truth_map(ground_truth_dir, image_paths)

    rows = [
        validate_image(path, ground_truth[path.name])
        for path in image_paths
    ]

    return pd.DataFrame(rows).sort_values("image").reset_index(drop=True)


def build_structural_metrics_dataframe(
    image_folder: Union[str, os.PathLike] = "notebookimages",
) -> pd.DataFrame:
    image_paths = discover_image_files(image_folder)
    rows = []

    for path in image_paths:
        metrics = compute_metrics(path).to_dict()
        metrics["image"] = path.name
        rows.append(metrics)

    return pd.DataFrame(rows).sort_values("image").reset_index(drop=True)


def merge_metrics_and_validation(
    metrics: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    return metrics.merge(validation, on="image", how="inner")


def correlation_frame(merged: pd.DataFrame) -> pd.DataFrame:
    return merged.corr(numeric_only=True)


def write_dataframe_csv(dataframe: pd.DataFrame, output_csv: Union[str, os.PathLike]) -> None:
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv, index=False)
